"""K-alpha Bragg-angle separations for crystal reflection models."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from .crystals import CrystalReflectionModel

RAD_TO_ARCSEC = 206264.80624709636
HC_KEV_ANGSTROM = 12.398419843320026


def _format_hkl(hkl: tuple[int, int, int]) -> str:
    """Format signed and multi-digit Miller indices without ambiguity."""
    values = tuple(int(value) for value in hkl)
    if any(value < 0 for value in values):
        return "(" + ", ".join(str(value) for value in values) + ")"
    if any(value >= 10 for value in values):
        return "(" + " ".join(str(value) for value in values) + ")"
    return "(" + "".join(str(value) for value in values) + ")"


@dataclass(frozen=True, slots=True)
class BraggDoubletConfig:
    lattice_a_A: float = 5.4635
    lambda_ka1_A: float = 0.7093000
    lambda_ka2_A: float = 0.7135900
    max_miller_index: int = 10

    def validate(self) -> "BraggDoubletConfig":
        for name, value in (("lattice_a_A", self.lattice_a_A),
                            ("lambda_ka1_A", self.lambda_ka1_A),
                            ("lambda_ka2_A", self.lambda_ka2_A)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} должен быть конечным положительным числом")
        if not float(self.lambda_ka2_A) > float(self.lambda_ka1_A):
            raise ValueError("Для Kα-дублета требуется lambda_Kα2 > lambda_Kα1")
        if isinstance(self.max_miller_index, bool) or int(self.max_miller_index) != self.max_miller_index:
            raise ValueError("max_miller_index должен быть целым числом")
        if not 1 <= int(self.max_miller_index) <= 30:
            raise ValueError("max_miller_index должен находиться в диапазоне 1–30")
        return self


def calculate_doublet_reflections(
    crystal: CrystalReflectionModel, config: BraggDoubletConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Kα1/Kα2 Bragg separations and rejected reflections.

    For the legacy CaF2 model this applies F-centering, as before. Database-backed
    models enumerate signed HKLs and use XrayLib structure factors to remove
    numerical systematic absences. Structure-factor values remain attached to
    surviving rows and are not interpreted as measured intensity or probability.
    """
    config.validate()
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    lam1 = float(config.lambda_ka1_A)
    lam2 = float(config.lambda_ka2_A)
    energy_ka1_keV = HC_KEV_ANGSTROM / lam1
    energy_ka2_keV = HC_KEV_ANGSTROM / lam2
    structure_factor_fn = getattr(crystal, "structure_factor", None)
    f0_abs_ka1 = f0_abs_ka2 = None
    extinction_threshold_ka1 = extinction_threshold_ka2 = None
    if callable(structure_factor_fn):
        f0_abs_ka1 = abs(complex(structure_factor_fn((0, 0, 0), energy_ka1_keV)))
        f0_abs_ka2 = abs(complex(structure_factor_fn((0, 0, 0), energy_ka2_keV)))
        extinction_threshold_ka1 = max(1.0e-12, 1.0e-12 * f0_abs_ka1)
        extinction_threshold_ka2 = max(1.0e-12, 1.0e-12 * f0_abs_ka2)

    for hkl in crystal.allowed_reflections(int(config.max_miller_index)):
        d = float(crystal.d_spacing_A(hkl, config.lattice_a_A))
        if not math.isfinite(d) or d <= 0.0:
            raise ValueError(f"Модель {crystal.model_id!r} вернула некорректное d-spacing для {hkl}")
        arg1 = lam1 / (2.0 * d)
        arg2 = lam2 / (2.0 * d)
        # Kα2 has the longer wavelength and is the limiting Bragg condition.
        # Reject geometrically impossible high-index reflections before asking
        # XrayLib for F_H; this keeps the database-backed search inexpensive.
        if arg2 > 1.0 + 1e-12:
            rejected.append({
                "h": hkl[0], "k": hkl[1], "l": hkl[2],
                "hkl": _format_hkl(hkl),
                "d_A": d, "reason": "Bragg condition not satisfied for Kα2",
            })
            continue

        structure_factor_ka1 = structure_factor_ka2 = None
        if callable(structure_factor_fn):
            structure_factor_ka1 = complex(structure_factor_fn(hkl, energy_ka1_keV))
            structure_factor_ka2 = complex(structure_factor_fn(hkl, energy_ka2_keV))
            if not all(math.isfinite(value) for factor in (structure_factor_ka1, structure_factor_ka2)
                       for value in (factor.real, factor.imag)):
                raise ValueError(f"XrayLib вернул не конечный F_H для {hkl}")
            extinct_ka1 = abs(structure_factor_ka1) <= float(extinction_threshold_ka1)
            extinct_ka2 = abs(structure_factor_ka2) <= float(extinction_threshold_ka2)
            # Only remove a reflection if the factor is numerically zero for
            # both doublet components. A one-energy cancellation must not
            # suppress the other component's physically possible reflection.
            if extinct_ka1 and extinct_ka2:
                rejected.append({
                    "h": hkl[0], "k": hkl[1], "l": hkl[2],
                    "hkl": _format_hkl(hkl), "d_A": d,
                    "reason": "XrayLib F_H is below numerical-zero tolerance for both Kα components",
                })
                continue

        theta1 = math.asin(min(1.0, max(0.0, arg1)))
        theta2 = math.asin(min(1.0, max(0.0, arg2)))
        delta = (theta2 - theta1) * RAD_TO_ARCSEC
        row = {
            "h": hkl[0], "k": hkl[1], "l": hkl[2],
            "hkl": _format_hkl(hkl),
            "hkl_norm_sq": hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2,
            "d_A": d,
            "theta_ka1_deg": math.degrees(theta1),
            "theta_ka2_deg": math.degrees(theta2),
            "delta_theta_arcsec": abs(delta),
            "lambda_ka1_A": lam1, "lambda_ka2_A": lam2,
            "reflection_condition": str(getattr(
                crystal, "selection_rule", "Model-defined geometrical reflection condition"
            )),
        }
        if structure_factor_ka1 is not None:
            row.update({
                "F_H_real_at_ka1": float(structure_factor_ka1.real),
                "F_H_imag_at_ka1": float(structure_factor_ka1.imag),
                "F_H_abs_at_ka1": float(abs(structure_factor_ka1)),
                "F_H_sq_at_ka1": float(abs(structure_factor_ka1) ** 2),
                "F_H_real_at_ka2": float(structure_factor_ka2.real),
                "F_H_imag_at_ka2": float(structure_factor_ka2.imag),
                "F_H_abs_at_ka2": float(abs(structure_factor_ka2)),
                "F_H_sq_at_ka2": float(abs(structure_factor_ka2) ** 2),
            })
        rows.append(row)

    reflection_columns = [
        "h", "k", "l", "hkl", "hkl_norm_sq", "d_A", "theta_ka1_deg",
        "theta_ka2_deg", "delta_theta_arcsec", "lambda_ka1_A", "lambda_ka2_A",
        "reflection_condition",
    ]
    if callable(structure_factor_fn):
        reflection_columns.extend([
            "F_H_real_at_ka1", "F_H_imag_at_ka1", "F_H_abs_at_ka1", "F_H_sq_at_ka1",
            "F_H_real_at_ka2", "F_H_imag_at_ka2", "F_H_abs_at_ka2", "F_H_sq_at_ka2",
        ])
    rejected_columns = ["h", "k", "l", "hkl", "d_A", "reason"]
    reflections = pd.DataFrame(rows, columns=reflection_columns)
    rejected_frame = pd.DataFrame(rejected, columns=rejected_columns)
    if reflections.empty:
        raise ValueError(
            "Для заданных параметров не найдено отражений, одновременно удовлетворяющих "
            "условию Брэгга для обеих компонент Kα и правилам отбора модели. "
            "Увеличьте max_miller_index или проверьте материал/длины волн."
        )
    reflections = reflections.sort_values(
        ["delta_theta_arcsec", "d_A", "h", "k", "l"],
        ascending=[True, False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    return reflections, rejected_frame
