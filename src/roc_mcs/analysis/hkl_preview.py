"""Renderer-neutral view model for the theoretical Kα/HKL analysis result."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class HklIdentificationPreview:
    """Detached table snapshots needed by the HKL results panel.

    DataFrames are deep-copied at construction time so the UI cannot mutate the
    immutable scientific ArtifactBundle held by the workflow.
    """

    theoretical_reflections: pd.DataFrame
    rejected_reflections: pd.DataFrame
    measurements: pd.DataFrame
    pair_statistics: pd.DataFrame
    pair_ranking: pd.DataFrame
    metadata: dict[str, Any]


def _copy_table(bundle: Any, name: str, required: set[str]) -> pd.DataFrame:
    tables = getattr(bundle, "tables", None)
    if not isinstance(tables, dict) or name not in tables:
        raise ValueError(f"В артефакте HKL отсутствует таблица {name!r}")
    frame = tables[name]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Таблица {name!r} должна быть pandas.DataFrame")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Таблица {name!r} не содержит столбцы {sorted(missing)}")
    return frame.copy(deep=True)


def build_hkl_identification_preview(bundle: Any) -> HklIdentificationPreview:
    """Validate and detach all scientific result tables for GUI consumption."""
    reflections = _copy_table(
        bundle, "hkl_theoretical_reflections",
        {"hkl", "d_A", "theta_ka1_deg", "delta_theta_arcsec"},
    )
    rejected = _copy_table(bundle, "hkl_bragg_rejected", {"hkl", "d_A", "reason"})
    measurements = _copy_table(
        bundle, "hkl_pair_measurements",
        {"pair_id", "delta_theta_abs_arcsec", "source_peak_id", "matched_partner_peak_id"},
    )
    pair_stats = _copy_table(
        bundle, "hkl_pair_statistics",
        {"pair_id", "matched_assignment_count", "delta_theta_abs_median_arcsec", "identification_status"},
    )
    ranking = _copy_table(
        bundle, "hkl_pair_ranking",
        {"pair_id", "rank", "hkl", "delta_theta_arcsec", "difference_from_exp_median_arcsec"},
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


__all__ = ["HklIdentificationPreview", "build_hkl_identification_preview"]
