"""Resolve immutable workflow artifacts required by stage previews.

This module contains lineage resolution only.  It is deliberately independent
of Qt so preview loading can be exercised with real History/ArtifactStore tests.
"""
from __future__ import annotations

from typing import Any

from .workflow_history import NodeStatus


_BRANCH_OPERATIONS = {"peaks.branches", "peaks.branch_correlation"}


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


def _branch_preview_context(node: Any, history: Any, executor: Any) -> dict[str, Any]:
    """Resolve scan, peak positions, optional spike QC and branch artifacts.

    A cross-correlation node consumes only the accepted branch catalogue.  The
    display Scan is intentionally resolved through that exact branch node's
    ancestry; it is not added as a fictitious scientific input to correlation.
    """
    if node.operation_id == "peaks.branches":
        branch_node = node
        branch_inputs = executor.resolve_node_inputs(branch_node)
        branches_bundle = executor.load_output(branch_node, "branches")
        correlation_bundle = None
    else:
        branch_ref = node.input_refs.get("branches")
        if not isinstance(branch_ref, dict) or not branch_ref.get("node_id"):
            raise RuntimeError(
                "У узла 3D-кросс-корреляции отсутствует явная ссылка на вход branches."
            )
        branch_node = history.get(branch_ref["node_id"])
        if branch_node.operation_id != "peaks.branches":
            raise RuntimeError(
                "Вход branches должен ссылаться на результат «Определение дифракционных ветвей»; "
                f"получен {branch_node.operation_id!r}."
            )
        inputs = executor.resolve_node_inputs(node)
        branches_bundle = inputs["branches"]
        # Re-resolve the branch stage's exact positional input to reconstruct
        # optional detected/Fit maps in the same way as its own preview.
        branch_inputs = executor.resolve_node_inputs(branch_node)
        correlation_bundle = executor.load_output(node, "branch_correlation")

    position_bundle = branch_inputs.get("positions")
    if position_bundle is None:
        raise RuntimeError(
            f"У этапа {branch_node.label!r} отсутствует разрешённый вход positions."
        )
    position_ref = branch_node.input_refs.get("positions")
    if not isinstance(position_ref, dict) or not position_ref.get("node_id"):
        raise RuntimeError(
            f"У этапа {branch_node.label!r} отсутствует ссылка на источник координат пиков."
        )
    position_node = history.get(position_ref["node_id"])

    scan_node = _require_successful_ancestor(history, branch_node.id, "scan")
    scan_bundle = executor.load_output(scan_node, "scan")

    spike_node = history.find_ancestor_output(branch_node.id, "spikes")
    spike_bundle = (
        executor.load_output(spike_node, "spikes")
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
    if node.operation_id == "peaks.branch_correlation":
        context["branch_correlation"] = correlation_bundle
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
