"""Optional CrystalPy/XrayLib enrichment for built-in and database crystals.

The historical module name is preserved for import compatibility.  The
CaF2-specific helper remains the source of the original notebook's explicit
fluorite basis; XrayLib database materials use their native embedded records.
"""
from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np
import pandas as pd

from .bragg import HC_KEV_ANGSTROM, RAD_TO_ARCSEC
from .crystals import CA_F2_FLUORITE, XraylibCrystalModel, get_crystal_model


def xraylib_available() -> bool:
    try:
        return importlib.util.find_spec("xraylib") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # find_spec may raise when a test harness injected a module directly.
        try:
            import xraylib  # noqa: F401
            return True
        except ImportError:
            return False


def crystalpy_available() -> bool:
    try:
        return importlib.util.find_spec("crystalpy") is not None and xraylib_available()
    except (ImportError, ModuleNotFoundError, ValueError):
        try:
            import crystalpy  # noqa: F401
            return xraylib_available()
        except ImportError:
            return False


def caf2_crystal_data(lattice_a_A: float) -> dict[str, Any]:
    """CaF2 fluorite basis from the supplied notebook (12 atoms/cell)."""
    a = float(lattice_a_A)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("Параметр решётки CaF2 должен быть конечным и положительным")
    atoms = [
        {"Zatom": 20, "fraction": 1.0, "x": 0.0, "y": 0.0, "z": 0.0, "charge": 0.0},
        {"Zatom": 20, "fraction": 1.0, "x": 0.0, "y": 0.5, "z": 0.5, "charge": 0.0},
        {"Zatom": 20, "fraction": 1.0, "x": 0.5, "y": 0.0, "z": 0.5, "charge": 0.0},
        {"Zatom": 20, "fraction": 1.0, "x": 0.5, "y": 0.5, "z": 0.0, "charge": 0.0},
    ]
    for x in (0.25, 0.75):
        for y in (0.25, 0.75):
            for z in (0.25, 0.75):
                atoms.append({"Zatom": 9, "fraction": 1.0, "x": x, "y": y, "z": z, "charge": 0.0})
    return {
        "name": "CaF2", "a": a, "b": a, "c": a,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        "volume": a ** 3, "n_atom": 12, "atom": atoms, "cpointer": None,
        "Aniso": [{"Name": "       ", "start": 0, "end": 0,
                   "beta11": 0.0, "beta22": 0.0, "beta33": 0.0,
                   "beta12": 0.0, "beta13": 0.0, "beta23": 0.0}],
        "n_aniso": 1,
    }


def add_crystalpy_properties(
    reflections: pd.DataFrame, *, lattice_a_A: float, lambda_ka1_A: float,
    lambda_ka2_A: float, thickness_m: float = 1e-5,
    crystal_model_id: str = "caf2_fluorite",
) -> pd.DataFrame:
    """Add CrystalPy amplitudes, susceptibilities and Darwin full widths.

    For CaF2 the explicitly declared fluorite basis from the old notebook is
    retained. For XrayLib-backed crystals, CrystalPy receives the database
    material name and loads its native unit-cell/atomic-basis record. This
    computes dynamical-theory quantities but does not synthesize rocking curves.
    """
    if not crystalpy_available():
        raise RuntimeError(
            "Расчёт F0/FH/FH_bar, psiH/psiH_bar, |FH|² и Darwin widths "
            "требует установленных crystalpy и xraylib. Отключите этот "
            "дополнительный расчёт либо установите обе зависимости."
        )
    thickness = float(thickness_m)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("Толщина кристалла должна быть конечным положительным числом")
    model = get_crystal_model(crystal_model_id)
    from crystalpy.diffraction.GeometryType import BraggDiffraction
    from crystalpy.diffraction.DiffractionSetupXraylib import DiffractionSetupXraylib

    e1 = HC_KEV_ANGSTROM / float(lambda_ka1_A) * 1000.0  # CrystalPy energy is eV.
    e2 = HC_KEV_ANGSTROM / float(lambda_ka2_A) * 1000.0
    if isinstance(model, XraylibCrystalModel):
        crystal_name = model.crystal_name
        crystal_data = None
    elif model.model_id == CA_F2_FLUORITE.model_id:
        crystal_name = "CaF2"
        crystal_data = caf2_crystal_data(lattice_a_A)
    else:
        raise ValueError(
            f"Для модели {model.display_name!r} пока нет адаптера CrystalPy. "
            "Используйте материал из базы XrayLib или отключите дополнительный расчёт CrystalPy."
        )

    enriched = reflections.copy(deep=True)
    columns: dict[str, list[float]] = {
        "F0_real": [], "F0_imag": [],
        "FH_real": [], "FH_imag": [],
        "FH_bar_real": [], "FH_bar_imag": [],
        "psiH_real": [], "psiH_imag": [],
        "psiH_bar_real": [], "psiH_bar_imag": [],
        "F0_sq": [], "FH_sq": [],
        "darwin_sigma_arcsec": [], "darwin_pi_arcsec": [],
        "crystalpy_theta_ka1_deg": [], "crystalpy_theta_ka2_deg": [],
        "crystalpy_thickness_m": [],
    }
    for row in enriched.itertuples(index=False):
        h, k, l = int(row.h), int(row.k), int(row.l)
        try:
            kwargs = {
                "geometry_type": BraggDiffraction(), "crystal_name": crystal_name,
                "thickness": thickness, "miller_h": h, "miller_k": k, "miller_l": l,
                "asymmetry_angle": 0.0, "azimuthal_angle": 0.0,
            }
            if crystal_data is not None:
                kwargs["crystal_data"] = crystal_data
            setup = DiffractionSetupXraylib(**kwargs)
            f0 = complex(setup.F0(e1))
            fh = complex(setup.FH(e1))
            fh_bar = complex(setup.FH_bar(e1))
            psi_h = complex(setup.psiH(e1))
            psi_h_bar = complex(setup.psiH_bar(e1))
            theta1 = float(setup.angleBragg(e1))
            theta2 = float(setup.angleBragg(e2))
            sigma = 2.0 * float(setup.darwinHalfwidthS(e1)) * RAD_TO_ARCSEC
            pi = 2.0 * float(setup.darwinHalfwidthP(e1)) * RAD_TO_ARCSEC
            values = {
                "F0_real": f0.real, "F0_imag": f0.imag,
                "FH_real": fh.real, "FH_imag": fh.imag,
                "FH_bar_real": fh_bar.real, "FH_bar_imag": fh_bar.imag,
                "psiH_real": psi_h.real, "psiH_imag": psi_h.imag,
                "psiH_bar_real": psi_h_bar.real, "psiH_bar_imag": psi_h_bar.imag,
                "F0_sq": abs(f0) ** 2, "FH_sq": abs(fh) ** 2,
                "darwin_sigma_arcsec": sigma, "darwin_pi_arcsec": pi,
                "crystalpy_theta_ka1_deg": math.degrees(theta1),
                "crystalpy_theta_ka2_deg": math.degrees(theta2),
                "crystalpy_thickness_m": thickness,
            }
            if not all(math.isfinite(float(value)) for value in values.values()):
                raise ValueError("CrystalPy вернул не конечные значения")
            for key, value in values.items():
                columns[key].append(float(value))
        except Exception as exc:
            raise RuntimeError(
                f"CrystalPy не смог рассчитать рефлекс ({h}, {k}, {l}) "
                f"для {model.display_name}: {exc}"
            ) from exc
    for key, values in columns.items():
        enriched[key] = np.asarray(values, dtype=np.float64)
    enriched.attrs["crystalpy_enrichment"] = True
    enriched.attrs["crystalpy_thickness_m"] = thickness
    enriched.attrs["crystal_model_id"] = crystal_model_id
    return enriched
