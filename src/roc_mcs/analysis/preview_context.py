"""Resolve immutable workflow artifacts required by stage previews.

This module contains lineage resolution only.  It is deliberately independent
of Qt so preview loading can be exercised with real History/ArtifactStore tests.
"""
from __future__ import annotations

from typing import Any

from .engine import SOURCE_REF
from .workflow_history import NodeStatus


_BRANCH_OPERATIONS = {"peaks.branches", "peaks.branch_correlation", "theory.identify_hkl"}


def _require_successful_ancestor(history: Any, node_id: str, output_name: str):
    node = history.find_ancestor_output(node_id, output_name)
    if node is None:
        raise RuntimeError(
            f"Для предпросмотра узла {node_id!r} не найден предшествующий выход {output_name!r}."
        )
    if node.status != NodeStatus.SUCCEEDED:
        raise RuntimeError(
            f"Предшествующий узел {node.label!r}, содержащий {output_name!r}, "
            f"не завершён успешно (status={node.status!r})."
        )
    return node


def _load_output_subset(
    executor: Any,
    node: Any,
    output_name: str,
    *,
    arrays: tuple[str, ...] = (),
    tables: tuple[str, ...] = (),
    columns: dict[str, tuple[str, ...]] | None = None,
    group_head_limits: dict[str, tuple[str, int]] | None = None,
    missing_ok: bool = True,
):
    """Resolve one output while decoding only the payload consumed by a preview."""
    artifact_ref = node.output_refs.get(output_name)
    if artifact_ref is None:
        raise ValueError(f"У узла {node.label!r} отсутствует выход {output_name!r}")
    if artifact_ref == SOURCE_REF:
        # External HDF5 scan access is managed by the repository and may use
        # source-backed arrays; retain its established loading contract.
        return executor.load_output(node, output_name)
    reader = getattr(executor.artifact_store, "read_subset", None)
    if not callable(reader):
        return executor.load_output(node, output_name)
    if not arrays and not tables:
        raise ValueError("Для предпросмотра не задано ни одного объекта артефакта")
    return reader(
        artifact_ref,
        array_names=arrays,
        table_names=tables,
        column_names_by_table=columns,
        group_head_limits=group_head_limits,
        missing_ok=missing_ok,
        missing_columns_ok=True,
    )


def _load_input_subset(
    history: Any,
    executor: Any,
    owner_node: Any,
    input_port: str,
    **selection: Any,
):
    reference = owner_node.input_refs.get(input_port)
    if not isinstance(reference, dict) or not reference.get("node_id") or not reference.get("output"):
        raise RuntimeError(
            f"У узла {owner_node.label!r} отсутствует корректная ссылка на вход {input_port!r}."
        )
    parent = history.get(reference["node_id"])
    if parent.review_status != "accepted":
        raise ValueError(
            f"Результат {parent.label!r} ещё не принят пользователем "
            f"(review_status={parent.review_status!r})."
        )
    return _load_output_subset(
        executor, parent, str(reference["output"]), **selection
    )


def _branch_preview_context(node: Any, history: Any, executor: Any) -> dict[str, Any]:
    """Resolve a complete scientific preview from the minimum required HDF5 payload.

    This is a read-only projection: it never changes the saved artefacts or
    scientific calculations. Required signal cubes remain full resolution;
    unrelated arrays and wide table columns are not decoded a second time.
    """
    branch_catalog_columns = ("peak_id", "branch_id", "x", "y", "theta", "theta_index")
    branch_bundle_selection = {
        "tables": ("branch_catalog", "branch_edges"),
        "columns": {"branch_catalog": branch_catalog_columns},
    }
    if node.operation_id == "peaks.branches":
        branch_node = node
        position_port = "positions"
        branches_bundle = _load_output_subset(
            executor, branch_node, "branches", **branch_bundle_selection
        )
        correlation_bundle = None
        hkl_bundle = None
    else:
        branch_ref = node.input_refs.get("branches")
        if not isinstance(branch_ref, dict) or not branch_ref.get("node_id"):
            raise RuntimeError(
                f"У узла {node.operation_id!r} отсутствует явная ссылка на вход branches."
            )
        branch_node = history.get(branch_ref["node_id"])
        if branch_node.operation_id != "peaks.branches":
            raise RuntimeError(
                "Вход branches должен ссылаться на результат «Определение дифракционных ветвей»; "
                f"получен {branch_node.operation_id!r}."
            )
        position_port = "positions"
        branches_bundle = _load_input_subset(
            history, executor, node, "branches", **branch_bundle_selection
        )
        if node.operation_id == "peaks.branch_correlation":
            correlation_bundle = _load_output_subset(
                executor, node, "branch_correlation",
                tables=(
                    "branch_correlation_pairs", "branch_correlation_assignments",
                    "branch_correlation_candidates",
                ),
                columns={
                    "branch_correlation_pairs": (
                        "source_branch_id", "partner_branch_id", "source_event_count",
                        "partner_event_count", "assigned_source_count", "final_supported_count",
                        "refined_shift_x_px", "refined_shift_y_px", "theta_shift_final_arcsec",
                        "theta_residual_mad_arcsec", "refinement_status",
                    ),
                    "branch_correlation_assignments": (
                        "source_peak_id", "source_branch_id", "assignment_class",
                        "assigned_partner_branch_id", "matched_partner_peak_id",
                    ),
                    "branch_correlation_candidates": (
                        "source_peak_id", "partner_branch_id", "matched_partner_peak_id",
                        "supported", "assigned",
                    ),
                },
            )
            hkl_bundle = None
        elif node.operation_id == "theory.identify_hkl":
            correlation_bundle = _load_input_subset(
                history, executor, node, "branch_correlation",
                tables=(
                    "branch_correlation_pairs", "branch_correlation_assignments",
                    "branch_correlation_candidates",
                ),
                columns={
                    "branch_correlation_pairs": (
                        "source_branch_id", "partner_branch_id", "source_event_count",
                        "partner_event_count", "assigned_source_count", "final_supported_count",
                        "refined_shift_x_px", "refined_shift_y_px", "theta_shift_final_arcsec",
                        "theta_residual_mad_arcsec", "refinement_status",
                    ),
                    "branch_correlation_assignments": (
                        "source_peak_id", "source_branch_id", "assignment_class",
                        "assigned_partner_branch_id", "matched_partner_peak_id",
                    ),
                    "branch_correlation_candidates": (
                        "source_peak_id", "partner_branch_id", "matched_partner_peak_id",
                        "supported", "assigned",
                    ),
                },
            )
            hkl_bundle = _load_output_subset(
                executor, node, "hkl_analysis",
                tables=(
                    "hkl_theoretical_reflections", "hkl_bragg_rejected",
                    "hkl_pair_measurements", "hkl_pair_statistics", "hkl_pair_ranking",
                ),
                columns={
                    "hkl_theoretical_reflections": (
                        "hkl", "d_A", "theta_ka1_deg", "delta_theta_arcsec",
                        "FH_sq", "F_H_sq_at_ka1", "darwin_sigma_arcsec", "darwin_pi_arcsec",
                    ),
                    "hkl_bragg_rejected": ("hkl", "d_A", "reason"),
                    "hkl_pair_measurements": (
                        "pair_id", "source_branch_id", "partner_branch_id",
                        "source_peak_id", "matched_partner_peak_id",
                        "delta_theta_abs_arcsec", "delta_theta_signed_arcsec",
                        "theta_source_arcsec", "theta_partner_arcsec",
                    ),
                    "hkl_pair_statistics": (
                        "pair_id", "matched_assignment_count", "identification_status",
                        "delta_theta_abs_median_arcsec", "delta_theta_abs_mean_arcsec",
                        "delta_theta_abs_q10_arcsec", "delta_theta_abs_q90_arcsec",
                        "delta_theta_abs_mad_arcsec", "unique_source_peak_count",
                        "unique_partner_peak_count", "reused_partner_assignment_count",
                        "best_hkl", "best_abs_difference_arcsec",
                        "best_theory_delta_theta_arcsec", "theta_shift_final_arcsec",
                    ),
                    "hkl_pair_ranking": (
                        "pair_id", "rank", "hkl", "d_A", "theta_ka1_deg",
                        "delta_theta_arcsec", "difference_from_exp_median_arcsec",
                        "FH_sq", "F_H_sq_at_ka1", "darwin_sigma_arcsec", "darwin_pi_arcsec",
                    ),
                },
                group_head_limits={"hkl_pair_ranking": ("pair_id", 15)},
            )
        else:
            raise RuntimeError(f"Неподдерживаемый branch preview для {node.operation_id!r}.")

    position_ref = branch_node.input_refs.get(position_port)
    if not isinstance(position_ref, dict) or not position_ref.get("node_id"):
        raise RuntimeError(
            f"У этапа {branch_node.label!r} отсутствует ссылка на источник координат пиков."
        )
    position_node = history.get(position_ref["node_id"])
    if position_node.operation_id == "peaks.detect":
        position_selection = {
            "arrays": ("peak_count_map",),
            "tables": ("peak_catalog",),
            "columns": {"peak_catalog": (
                "peak_id", "x", "y", "height", "prominence", "width_theta",
            )},
        }
    elif position_node.operation_id == "fit.scan":
        position_selection = {
            "arrays": (
                "n_detected_map", "n_fitted_map", "fit_cube", "residual_cube",
                "valid_mask", "theta",
            ),
            "tables": ("pixel_fit_catalog", "fitted_peak_catalog", "peak_catalog"),
            "columns": {
                "pixel_fit_catalog": (
                    "x", "y", "n_detected", "n_fitted", "rmse", "redchi", "chisqr",
                    "p95_abs_residual", "max_abs_residual", "rmse_over_noise", "aic", "bic",
                ),
                "peak_catalog": ("peak_id", "x", "y", "height", "prominence", "width_theta"),
            },
        }
    else:
        # Future position-producing operations retain a correct full-artifact
        # fallback until their preview schema is explicitly registered here.
        position_selection = None
    if position_selection is None:
        # An operation added in a future release may carry an unfamiliar
        # position schema. Preserve the previous full-read behavior rather than
        # silently omitting data from a new preview type.
        resolved_inputs = executor.resolve_node_inputs(branch_node)
        position_bundle = resolved_inputs.get(position_port)
    else:
        position_bundle = _load_input_subset(
            history, executor, branch_node, position_port, **position_selection
        )

    scan_node = _require_successful_ancestor(history, branch_node.id, "scan")
    scan_bundle = _load_output_subset(
        executor, scan_node, "scan",
        arrays=(
            "data_yxt", "theta", "x_coords", "y_coords",
            "valid_mask_yxt", "analysis_valid_mask_yx",
        ),
    )

    spike_node = history.find_ancestor_output(branch_node.id, "spikes")
    spike_bundle = (
        _load_output_subset(
            executor, spike_node, "spikes",
            arrays=("spike_count_map", "clean_data_yxt", "detector_valid_mask_yxt"),
        )
        if spike_node is not None and spike_node.status == NodeStatus.SUCCEEDED
        else None
    )

    context: dict[str, Any] = {
        "node": node,
        "scan": scan_bundle,
        "spikes": spike_bundle,
        "peaks": position_bundle if position_node.operation_id == "peaks.detect" else None,
        "fit": position_bundle if position_node.operation_id == "fit.scan" else None,
        "branches": branches_bundle,
    }
    if correlation_bundle is not None:
        context["branch_correlation"] = correlation_bundle
    if hkl_bundle is not None:
        context["hkl_analysis"] = hkl_bundle
    return context


def load_node_preview_context(node: Any, history: Any, executor: Any) -> dict[str, Any]:
    """Load the exact artifact context needed to preview one successful node.

    The returned mapping is renderer-neutral and can be passed to
    ``build_workflow_preview_state``.  Large HDF5 loads are performed by the
    caller's worker thread.
    """
    if node.operation_id == "source.scan":
        return {
            "node": node,
            "scan": executor.load_output(node, "scan"),
            "spikes": None,
            "peaks": None,
            "fit": None,
        }

    if node.operation_id in {"preprocess.pool", "preprocess.detector_bad_pixels"}:
        # These operations return a complete ScanArtifact. Preview their exact
        # output, not an upstream unmodified input.
        return {
            "node": node,
            "scan": executor.load_output(node, "scan"),
            "spikes": None,
            "peaks": None,
            "fit": None,
        }

    if node.operation_id in _BRANCH_OPERATIONS:
        return _branch_preview_context(node, history, executor)

    inputs = executor.resolve_node_inputs(node)
    output_names = tuple(node.output_refs)
    if not output_names:
        raise RuntimeError(
            f"У успешно завершённого узла {node.label!r} отсутствуют ссылки на выходные артефакты."
        )
    stage_output = executor.load_output(node, output_names[0])
    spike_context = (
        stage_output if node.operation_id == "preprocess.spikes" else inputs.get("spikes")
    )
    spike_diagnostic_warning = None
    if spike_context is None and node.operation_id in {"peaks.detect", "fit.scan"}:
        # QC overlay is not an algorithm input. Resolve it only for display when
        # the operation does not consume the spike artifact.
        spike_node = history.find_ancestor_output(node.id, "spikes")
        if spike_node is not None and spike_node.status == NodeStatus.SUCCEEDED:
            try:
                spike_context = executor.load_output(spike_node, "spikes")
            except Exception as exc:  # QC is optional and must not hide core preview
                spike_diagnostic_warning = str(exc)

    return {
        "node": node,
        "scan": inputs.get("scan"),
        "spikes": spike_context,
        "spike_diagnostic_warning": spike_diagnostic_warning,
        "peaks": stage_output if node.operation_id == "peaks.detect" else inputs.get("peaks"),
        "fit": stage_output if node.operation_id == "fit.scan" else None,
    }


__all__ = ["load_node_preview_context"]
