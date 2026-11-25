"""Identify candidate HKL families from matched-peak Kα doublet separations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from roc_mcs.theory.bragg import BraggDoubletConfig, calculate_doublet_reflections
from roc_mcs.theory.crystals import get_crystal_model
from roc_mcs.theory.crystalpy_caf2 import add_crystalpy_properties


@dataclass(frozen=True, slots=True)
class KAlphaIdentificationConfig:
    crystal_model_id: str = "caf2_fluorite"
    lattice_a_A: float = 5.4635
    lambda_ka1_A: float = 0.7093000
    lambda_ka2_A: float = 0.7135900
    max_miller_index: int = 10
    histogram_bin_width_arcsec: float = 7.0
    include_crystalpy: bool = False
    crystal_thickness_m: float = 1e-5


def _finite_series(frame: pd.DataFrame, column: str, *, label: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: столбец {column!r} содержит пропуски или нечисловые значения")
    return values


def _build_peak_lookup(catalog: pd.DataFrame, theta_unit: str) -> dict[int, tuple[int, float]]:
    required = {"peak_id", "branch_id", "theta"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"branch_catalog не содержит поля {sorted(missing)}")
    if str(theta_unit).strip().lower() not in {"arcsec", "arcsecond", "arcseconds", "″"}:
        raise ValueError(
            f"Для HKL-идентификации требуется theta_unit='arcsec', получено {theta_unit!r}. "
            "Преобразование неизвестных единиц автоматически не выполняется."
        )
    peak_ids_raw = pd.to_numeric(catalog["peak_id"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    branch_ids = _finite_series(catalog, "branch_id", label="branch_catalog")
    theta = _finite_series(catalog, "theta", label="branch_catalog")
    if (not np.isfinite(peak_ids_raw).all() or not np.equal(peak_ids_raw, np.rint(peak_ids_raw)).all()
            or not np.equal(branch_ids, np.rint(branch_ids)).all()):
        raise ValueError("branch_catalog содержит некорректные peak_id/branch_id")
    ids = np.rint(peak_ids_raw).astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError("branch_catalog содержит повторяющиеся peak_id; однозначный join невозможен")
    branches = np.rint(branch_ids).astype(np.int64)
    return {int(pid): (int(bid), float(th)) for pid, bid, th in zip(ids, branches, theta)}


def _extract_pair_measurements(
    branch_catalog: pd.DataFrame, correlation_tables: Mapping[str, pd.DataFrame], *, theta_unit: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_tables = {"branch_correlation_pairs", "branch_correlation_assignments"}
    missing_tables = required_tables.difference(correlation_tables)
    if missing_tables:
        raise ValueError(f"В артефакте кросс-корреляции отсутствуют таблицы {sorted(missing_tables)}")
    pairs = correlation_tables["branch_correlation_pairs"]
    assignments = correlation_tables["branch_correlation_assignments"]
    if not {"source_branch_id", "partner_branch_id"}.issubset(pairs.columns):
        raise ValueError("branch_correlation_pairs не содержит идентификаторы пары ветвей")
    assignment_required = {
        "source_peak_id", "source_branch_id", "assignment_class",
        "assigned_partner_branch_id", "matched_partner_peak_id",
    }
    missing = assignment_required.difference(assignments.columns)
    if missing:
        raise ValueError(
            "Артефакт кросс-корреляции не содержит точные peak_id назначений "
            f"({sorted(missing)}). Нельзя восстанавливать пары по ближайшим координатам; "
            "повторите кросс-корреляцию актуальной версией."
        )
    lookup = _build_peak_lookup(branch_catalog, theta_unit)
    if pairs.empty:
        raise ValueError("Кросс-корреляция не сохранила ни одной пары ветвей")

    # A source event is assigned at most once by the directed matcher. Repeated
    # target IDs are retained and explicitly counted: the matcher is many-to-one.
    source_ids = pd.to_numeric(assignments["source_peak_id"], errors="coerce")
    if source_ids.dropna().duplicated().any():
        raise ValueError("В assignments повторяются source_peak_id; ожидается одна строка на source-пик")
    assignment_branch = pd.to_numeric(assignments["source_branch_id"], errors="coerce")
    assigned_partner = pd.to_numeric(assignments["assigned_partner_branch_id"], errors="coerce")
    matched_ids = pd.to_numeric(assignments["matched_partner_peak_id"], errors="coerce")
    partner_mask = assignments["assignment_class"].astype(str).eq("partner")
    measurements: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for pair in pairs.itertuples(index=False):
        source_branch = int(getattr(pair, "source_branch_id"))
        partner_branch = int(getattr(pair, "partner_branch_id"))
        mask = (partner_mask & assignment_branch.eq(source_branch)
                & assigned_partner.eq(partner_branch) & matched_ids.notna() & source_ids.notna())
        subset_indices = np.flatnonzero(mask.to_numpy())
        rows: list[dict[str, Any]] = []
        for index in subset_indices:
            source_peak_id = int(source_ids.iloc[index])
            partner_peak_id = int(matched_ids.iloc[index])
            source_record = lookup.get(source_peak_id)
            partner_record = lookup.get(partner_peak_id)
            if source_record is None or partner_record is None:
                missing_id = source_peak_id if source_record is None else partner_peak_id
                raise ValueError(
                    f"Назначение пары {source_branch}↔{partner_branch} ссылается на peak_id={missing_id}, "
                    "которого нет в branch_catalog. Пересчитайте кросс-корреляцию от того же каталога ветвей."
                )
            source_actual_branch, theta_source = source_record
            partner_actual_branch, theta_partner = partner_record
            if source_actual_branch != source_branch or partner_actual_branch != partner_branch:
                raise ValueError(
                    f"Несогласованные branch_id для соответствия peak_id={source_peak_id}↔{partner_peak_id}: "
                    f"ожидалось {source_branch}↔{partner_branch}, найдено "
                    f"{source_actual_branch}↔{partner_actual_branch}."
                )
            signed = float(theta_partner - theta_source)
            rows.append({
                "pair_id": f"{source_branch}↔{partner_branch}",
                "source_branch_id": source_branch, "partner_branch_id": partner_branch,
                "source_peak_id": source_peak_id, "matched_partner_peak_id": partner_peak_id,
                "theta_source_arcsec": theta_source, "theta_partner_arcsec": theta_partner,
                "delta_theta_signed_arcsec": signed, "delta_theta_abs_arcsec": abs(signed),
                "measurement_basis": "raw_theta_from_branch_catalog_by_peak_id",
            })
        measurements.extend(rows)
        values_signed = np.asarray([row["delta_theta_signed_arcsec"] for row in rows], dtype=np.float64)
        values = np.abs(values_signed)
        n = int(values.size)
        unique_partner = len({row["matched_partner_peak_id"] for row in rows})
        unique_source = len({row["source_peak_id"] for row in rows})
        median_signed = float(np.median(values_signed)) if n else float("nan")
        median_abs = float(np.median(values)) if n else float("nan")
        q10, q90 = (np.percentile(values, [10.0, 90.0]).tolist() if n else (float("nan"), float("nan")))
        mad = float(np.median(np.abs(values - median_abs))) if n else float("nan")
        final_shift = getattr(pair, "theta_shift_final_arcsec", float("nan"))
        try:
            final_shift = float(final_shift)
        except (TypeError, ValueError, OverflowError):
            final_shift = float("nan")
        consistency = median_signed + final_shift if np.isfinite(final_shift) and n else float("nan")
        summaries.append({
            "source_branch_id": source_branch, "partner_branch_id": partner_branch,
            "pair_id": f"{source_branch}↔{partner_branch}",
            "matched_assignment_count": n, "unique_source_peak_count": unique_source,
            "unique_partner_peak_count": unique_partner,
            "reused_partner_assignment_count": max(0, n - unique_partner),
            "delta_theta_signed_median_arcsec": median_signed,
            "delta_theta_abs_mean_arcsec": float(np.mean(values)) if n else float("nan"),
            "delta_theta_abs_median_arcsec": median_abs,
            "delta_theta_abs_q10_arcsec": float(q10),
            "delta_theta_abs_q90_arcsec": float(q90),
            "delta_theta_abs_mad_arcsec": mad,
            "theta_shift_final_arcsec": final_shift,
            "alignment_sign_consistency_arcsec": consistency,
            "identification_status": "ready" if n >= 2 else "insufficient_matches",
        })
    measurement_columns = [
        "pair_id", "source_branch_id", "partner_branch_id", "source_peak_id",
        "matched_partner_peak_id", "theta_source_arcsec", "theta_partner_arcsec",
        "delta_theta_signed_arcsec", "delta_theta_abs_arcsec", "measurement_basis",
    ]
    summary_columns = [
        "source_branch_id", "partner_branch_id", "pair_id", "matched_assignment_count",
        "unique_source_peak_count", "unique_partner_peak_count", "reused_partner_assignment_count",
        "delta_theta_signed_median_arcsec", "delta_theta_abs_mean_arcsec",
        "delta_theta_abs_median_arcsec", "delta_theta_abs_q10_arcsec",
        "delta_theta_abs_q90_arcsec", "delta_theta_abs_mad_arcsec", "theta_shift_final_arcsec",
        "alignment_sign_consistency_arcsec", "identification_status",
    ]
    return (
        pd.DataFrame(measurements, columns=measurement_columns),
        pd.DataFrame(summaries, columns=summary_columns),
    )


def identify_kalpha_hkl(
    branch_catalog: pd.DataFrame, correlation_tables: Mapping[str, pd.DataFrame], *,
    branch_metadata: Mapping[str, Any], config: KAlphaIdentificationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute theoretical doublet lines and rank them against each matched pair.

    The measured quantity is |theta_partner(raw)-theta_source(raw)| obtained by
    joining the exact stored peak IDs to the original branch_catalog. Aligned
    residuals from the matcher are deliberately not used: they should be near
    zero after registration and do not represent the Kα doublet separation.
    """
    if not np.isfinite(float(config.histogram_bin_width_arcsec)) or config.histogram_bin_width_arcsec <= 0:
        raise ValueError("Ширина гистограммы должна быть конечной и положительной")
    if not np.isfinite(float(config.crystal_thickness_m)) or config.crystal_thickness_m <= 0:
        raise ValueError("Толщина кристалла должна быть конечной и положительной")
    crystal = get_crystal_model(config.crystal_model_id)
    theory_config = BraggDoubletConfig(
        lattice_a_A=config.lattice_a_A, lambda_ka1_A=config.lambda_ka1_A,
        lambda_ka2_A=config.lambda_ka2_A, max_miller_index=config.max_miller_index,
    ).validate()
    reflections, rejected = calculate_doublet_reflections(crystal, theory_config)
    if config.include_crystalpy:
        reflections = add_crystalpy_properties(
            reflections, lattice_a_A=config.lattice_a_A,
            lambda_ka1_A=config.lambda_ka1_A, lambda_ka2_A=config.lambda_ka2_A,
            thickness_m=config.crystal_thickness_m,
            crystal_model_id=config.crystal_model_id,
        )
    measurements, pair_stats = _extract_pair_measurements(
        branch_catalog, correlation_tables, theta_unit=str(branch_metadata.get("theta_unit", "")),
    )
    ranking_rows: list[dict[str, Any]] = []
    best_by_pair: dict[str, str | None] = {}
    for stats in pair_stats.itertuples(index=False):
        pair_id = str(stats.pair_id)
        if stats.identification_status != "ready":
            best_by_pair[pair_id] = None
            continue
        median = float(stats.delta_theta_abs_median_arcsec)
        ranked = reflections.copy(deep=True)
        ranked["delta_theta_exp_median_arcsec"] = median
        ranked["delta_theta_exp_q10_arcsec"] = float(stats.delta_theta_abs_q10_arcsec)
        ranked["delta_theta_exp_q90_arcsec"] = float(stats.delta_theta_abs_q90_arcsec)
        ranked["difference_from_exp_median_arcsec"] = np.abs(
            ranked["delta_theta_arcsec"].to_numpy(dtype=np.float64) - median
        )
        # d-spacing is a lattice-aware deterministic tie-breaker. Raw h²+k²+l²
        # is not a physically meaningful norm for non-cubic reciprocal cells.
        ranked = ranked.sort_values(
            ["difference_from_exp_median_arcsec", "d_A", "h", "k", "l"],
            ascending=[True, False, True, True, True], kind="mergesort",
        ).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int32))
        ranked.insert(0, "pair_id", pair_id)
        ranked.insert(1, "source_branch_id", int(stats.source_branch_id))
        ranked.insert(2, "partner_branch_id", int(stats.partner_branch_id))
        ranking_rows.extend(ranked.to_dict(orient="records"))
        best_by_pair[pair_id] = str(ranked.iloc[0]["hkl"])
    ranking_columns = [
        "pair_id", "source_branch_id", "partner_branch_id", "rank",
        *reflections.columns.tolist(), "delta_theta_exp_median_arcsec",
        "delta_theta_exp_q10_arcsec", "delta_theta_exp_q90_arcsec",
        "difference_from_exp_median_arcsec",
    ]
    ranking = pd.DataFrame(ranking_rows, columns=list(dict.fromkeys(ranking_columns)))
    if not pair_stats.empty:
        pair_stats = pair_stats.copy()
        pair_stats["best_hkl"] = pair_stats["pair_id"].map(best_by_pair)
        best_delta = []
        best_error = []
        for row in pair_stats.itertuples(index=False):
            sub = ranking.loc[ranking["pair_id"] == row.pair_id] if not ranking.empty else ranking
            if sub.empty:
                best_delta.append(float("nan")); best_error.append(float("nan"))
            else:
                best_delta.append(float(sub.iloc[0]["delta_theta_arcsec"]))
                best_error.append(float(sub.iloc[0]["difference_from_exp_median_arcsec"]))
        pair_stats["best_theory_delta_theta_arcsec"] = best_delta
        pair_stats["best_abs_difference_arcsec"] = best_error
    if hasattr(crystal, "crystal_data"):
        cell_parameters = dict(crystal.unit_cell_parameters_A)
        cell_angles = dict(crystal.unit_cell_angles_deg)
        lattice_a_metadata = float(cell_parameters["a"])
        cell_description = (
            "a={a:.6g}, b={b:.6g}, c={c:.6g} Å; "
            "α={alpha:.5g}°, β={beta:.5g}°, γ={gamma:.5g}°"
        ).format(**cell_parameters, **cell_angles)
    else:
        lattice_a_metadata = float(config.lattice_a_A)
        cell_parameters = {"a": lattice_a_metadata, "b": lattice_a_metadata, "c": lattice_a_metadata}
        cell_angles = {"alpha": 90.0, "beta": 90.0, "gamma": 90.0}
        cell_description = f"кубическая ячейка a=b=c={lattice_a_metadata:.6g} Å"
    metadata = {
        "theory_model": config.crystal_model_id, "crystal_name": crystal.display_name,
        "space_group": crystal.space_group, "lattice_a_A": lattice_a_metadata,
        "unit_cell_parameters_A": cell_parameters,
        "unit_cell_angles_deg": cell_angles,
        "unit_cell_description": cell_description,
        "crystal_database": "XrayLib" if hasattr(crystal, "crystal_data") else "встроенная модель CaF2",
        "radiation": "Mo Kα1/Kα2", "lambda_ka1_A": float(config.lambda_ka1_A),
        "lambda_ka2_A": float(config.lambda_ka2_A),
        "max_miller_index": int(config.max_miller_index),
        "histogram_bin_width_arcsec": float(config.histogram_bin_width_arcsec),
        "crystalpy_thickness_m": float(config.crystal_thickness_m) if config.include_crystalpy else None,
        "structure_factor_backend": "CrystalPy/XrayLib" if config.include_crystalpy else (
            "XrayLib F_H at Kα1/Kα2" if hasattr(crystal, "crystal_data") else None
        ),
        "theory_selection_rule": str(getattr(crystal, "selection_rule", "модельное условие отбора")),
        "structure_factor_numerical_zero_threshold_relative": 1.0e-12 if hasattr(crystal, "crystal_data") else None,
        "structure_factor_zero_screen": "both Kα components below threshold" if hasattr(crystal, "crystal_data") else None,
        "hkl_index_enumeration": "full signed index cube; no point-group/Friedel reduction" if hasattr(crystal, "crystal_data") else "canonical cubic family representative",
        "experimental_delta_definition": "abs(theta_partner_raw_arcsec - theta_source_raw_arcsec)",
        "experimental_theta_source": "branch_catalog joined by exact peak_id assignments",
        "aligned_residual_used_for_identification": False,
        "ranking_definition": "absolute difference between theoretical separation and per-pair experimental median",
        "ranking_is_probability": False,
        "many_to_one_assignment_preserved": True,
        "minimum_matches_for_ranking": 2,
        "reflection_count": int(len(reflections)), "bragg_rejected_reflection_count": int(len(rejected)),
        "pair_count": int(len(pair_stats)), "measurement_count": int(len(measurements)),
    }
    return reflections, rejected, measurements, pair_stats, ranking, metadata
