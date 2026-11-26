"""Renderer-neutral view model for the theoretical Kα/HKL analysis result."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class HklIdentificationPreview:
    """Detached table snapshots needed by the HKL results panel.

    By default, DataFrames are deep-copied so callers cannot mutate the
    immutable workflow artifact. The HDF5 preview loader may retain its freshly
    decoded, privately owned frames to avoid a redundant second full copy.
    """

    theoretical_reflections: pd.DataFrame
    rejected_reflections: pd.DataFrame
    measurements: pd.DataFrame
    pair_statistics: pd.DataFrame
    pair_ranking: pd.DataFrame
    metadata: dict[str, Any]


def _copy_table(
    bundle: Any, name: str, required: set[str], *, copy_data: bool = True
) -> pd.DataFrame:
    tables = getattr(bundle, "tables", None)
    if not isinstance(tables, dict) or name not in tables:
        raise ValueError(f"В артефакте HKL отсутствует таблица {name!r}")
    frame = tables[name]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Таблица {name!r} должна быть pandas.DataFrame")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Таблица {name!r} не содержит столбцы {sorted(missing)}")
    return frame.copy(deep=True) if copy_data else frame


def build_hkl_identification_preview(
    bundle: Any, *, copy_tables: bool = True
) -> HklIdentificationPreview:
    """Validate and detach all scientific result tables for GUI consumption.

    ``copy_tables=False`` is reserved for ``load_hkl_identification_preview``:
    its input is freshly decoded from an immutable HDF5 artifact and already
    owns independent memory. The default deep-copy behavior protects callers
    that build a preview from a live in-memory workflow bundle.
    """
    reflections = _copy_table(
        bundle, "hkl_theoretical_reflections",
        {"hkl", "d_A", "theta_ka1_deg", "delta_theta_arcsec"},
        copy_data=copy_tables,
    )
    rejected = _copy_table(
        bundle, "hkl_bragg_rejected", {"hkl", "d_A", "reason"},
        copy_data=copy_tables,
    )
    measurements = _copy_table(
        bundle, "hkl_pair_measurements",
        {"pair_id", "delta_theta_abs_arcsec", "source_peak_id", "matched_partner_peak_id"},
        copy_data=copy_tables,
    )
    pair_stats = _copy_table(
        bundle, "hkl_pair_statistics",
        {"pair_id", "matched_assignment_count", "delta_theta_abs_median_arcsec", "identification_status"},
        copy_data=copy_tables,
    )
    ranking = _copy_table(
        bundle, "hkl_pair_ranking",
        {"pair_id", "rank", "hkl", "delta_theta_arcsec", "difference_from_exp_median_arcsec"},
        copy_data=copy_tables,
    )
    metadata = getattr(bundle, "metadata", {}) or {}
    theory_metadata = metadata.get("theory", {})
    if not isinstance(theory_metadata, dict):
        raise ValueError("В метаданных HKL отсутствует словарь theory")
    return HklIdentificationPreview(
        theoretical_reflections=reflections,
        rejected_reflections=rejected,
        measurements=measurements,
        pair_statistics=pair_stats,
        pair_ranking=ranking,
        metadata=dict(theory_metadata),
    )


def load_hkl_identification_preview(artifact_store: Any, artifact_id: str) -> HklIdentificationPreview:
    """Load and validate one HKL result artifact without resolving its lineage.

    HKL is an analysis-only output. Its result tables do not require loading the
    upstream 3D scan, branch catalog, or the generic ROI preview context.
    """
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("Для HKL-предпросмотра отсутствует artifact_id")
    table_names = (
        "hkl_theoretical_reflections",
        "hkl_bragg_rejected",
        "hkl_pair_measurements",
        "hkl_pair_statistics",
        "hkl_pair_ranking",
    )
    # Avoid decoding potentially large peer tables and arrays. The selected
    # frames are fresh HDF5-decoded objects, so another deep copy would add
    # memory/time without improving isolation.
    if hasattr(artifact_store, "read_subset"):
        # Read only data used by the result panel and the post-registration
        # Δθ layers. Optional display columns are schema-tolerant for projects
        # created by older releases; the required scientific fields are still
        # validated by build_hkl_identification_preview below.
        projections = {
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
        }
        bundle = artifact_store.read_subset(
            artifact_id,
            table_names=table_names,
            column_names_by_table=projections,
            group_head_limits={"hkl_pair_ranking": ("pair_id", 15)},
            missing_columns_ok=True,
        )
        return build_hkl_identification_preview(bundle, copy_tables=False)
    if hasattr(artifact_store, "read_tables"):
        bundle = artifact_store.read_tables(
            artifact_id,
            table_names,
            group_head_limits={"hkl_pair_ranking": ("pair_id", 15)},
        )
        return build_hkl_identification_preview(bundle, copy_tables=False)
    # Compatibility for small test doubles and third-party stores.
    bundle = artifact_store.read(artifact_id)
    return build_hkl_identification_preview(bundle)


__all__ = [
    "HklIdentificationPreview",
    "build_hkl_identification_preview",
    "load_hkl_identification_preview",
]
