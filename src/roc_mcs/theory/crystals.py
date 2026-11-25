"""Crystal reflection models: explicit lattices plus optional XrayLib database.

The geometrical Bragg layer depends only on this module's small model contract.
XrayLib is imported lazily, so the established CaF2-only path remains usable
without optional diffraction packages.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable


@runtime_checkable
class CrystalReflectionModel(Protocol):
    """Contract for a crystal model used by the Bragg-doublet calculator."""

    model_id: str
    display_name: str
    space_group: str

    def allowed_reflections(self, max_index: int) -> Iterator[tuple[int, int, int]]: ...

    def d_spacing_A(self, hkl: tuple[int, int, int], lattice_a_A: float | None = None) -> float: ...


@dataclass(frozen=True, slots=True)
class CubicFccCrystal:
    """Cubic F-centred lattice model used for the established CaF2 workflow.

    This models the F-centering condition only. It does not claim that every
    permitted reflection has appreciable intensity for the CaF2 basis.
    """

    model_id: str
    display_name: str
    lattice_a_default_A: float
    space_group: str = "Fm-3m"
    centering: str = "F"
    selection_rule: str = "F-centering: h,k,l all even or all odd; canonical 0≤h≤k≤l"

    def allowed_reflections(self, max_index: int) -> Iterator[tuple[int, int, int]]:
        maximum = _validate_max_index(max_index)
        for h in range(maximum + 1):
            for k in range(h, maximum + 1):
                for l in range(k, maximum + 1):
                    if h == k == l == 0:
                        continue
                    if h % 2 == k % 2 == l % 2:
                        yield (h, k, l)

    def d_spacing_A(self, hkl: tuple[int, int, int], lattice_a_A: float | None = None) -> float:
        a = float(self.lattice_a_default_A if lattice_a_A is None else lattice_a_A)
        if not math.isfinite(a) or a <= 0.0:
            raise ValueError("Параметр решётки a должен быть конечным положительным числом")
        h, k, l = (int(value) for value in hkl)
        norm_sq = h * h + k * k + l * l
        if norm_sq <= 0:
            raise ValueError("Рефлекс (000) не определён")
        return a / math.sqrt(norm_sq)

    @property
    def unit_cell_parameters_A(self) -> dict[str, float]:
        a = float(self.lattice_a_default_A)
        return {"a": a, "b": a, "c": a}

    @property
    def unit_cell_angles_deg(self) -> dict[str, float]:
        return {"alpha": 90.0, "beta": 90.0, "gamma": 90.0}


@dataclass(frozen=True, slots=True)
class XraylibCrystalModel:
    """A material from XrayLib's embedded crystal database.

    HKLs are enumerated throughout the signed reciprocal-index cube. Neither
    point-group nor Friedel equivalents are silently collapsed: non-centrosymmetric
    materials can show Bijvoet differences when anomalous scattering matters,
    and XrayLib's crystal record does not provide the full symmetry-operation set.
    """

    model_id: str
    crystal_name: str
    crystal_data: Mapping[str, Any]
    display_name: str
    space_group: str = "не указан в записи XrayLib"
    selection_rule: str = (
        "XrayLib unit-cell metric; full signed HKL enumeration; "
        "systematic absences screened by XrayLib structure factor"
    )

    @property
    def unit_cell_parameters_A(self) -> dict[str, float]:
        data = self.crystal_data
        return {axis: float(data[axis]) for axis in ("a", "b", "c")}

    @property
    def unit_cell_angles_deg(self) -> dict[str, float]:
        data = self.crystal_data
        return {axis: float(data[axis]) for axis in ("alpha", "beta", "gamma")}

    def allowed_reflections(self, max_index: int) -> Iterator[tuple[int, int, int]]:
        maximum = _validate_max_index(max_index)
        for h in range(-maximum, maximum + 1):
            for k in range(-maximum, maximum + 1):
                for l in range(-maximum, maximum + 1):
                    if h == k == l == 0:
                        continue
                    yield (h, k, l)

    def d_spacing_A(self, hkl: tuple[int, int, int], lattice_a_A: float | None = None) -> float:
        del lattice_a_A  # Database cell constants are authoritative for this model.
        h, k, l = (int(value) for value in hkl)
        if h == k == l == 0:
            raise ValueError("Рефлекс (000) не определён")
        cell = self.crystal_data
        a, b, c = (float(cell[key]) for key in ("a", "b", "c"))
        angles_deg = tuple(float(cell[key]) for key in ("alpha", "beta", "gamma"))
        if any(not math.isfinite(value) or value <= 0.0 for value in (a, b, c)):
            raise ValueError(f"Некорректные параметры ячейки XrayLib для {self.crystal_name}")
        if any(not math.isfinite(value) or not 0.0 < value < 180.0 for value in angles_deg):
            raise ValueError(f"Некорректные углы ячейки XrayLib для {self.crystal_name}: {angles_deg}")
        alpha, beta, gamma = (math.radians(value) for value in angles_deg)
        sin_a, sin_b, sin_g = math.sin(alpha), math.sin(beta), math.sin(gamma)
        volume_factor_sq = (
            1.0 + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
            - math.cos(alpha) ** 2 - math.cos(beta) ** 2 - math.cos(gamma) ** 2
        )
        if min(abs(sin_a), abs(sin_b), abs(sin_g)) < 1e-12 or volume_factor_sq <= 1e-14:
            raise ValueError(f"Вырожденная кристаллическая ячейка XrayLib для {self.crystal_name}")
        volume = a * b * c * math.sqrt(volume_factor_sq)
        astar = b * c * sin_a / volume
        bstar = a * c * sin_b / volume
        cstar = a * b * sin_g / volume
        cos_alpha_star = (math.cos(beta) * math.cos(gamma) - math.cos(alpha)) / (sin_b * sin_g)
        cos_beta_star = (math.cos(alpha) * math.cos(gamma) - math.cos(beta)) / (sin_a * sin_g)
        cos_gamma_star = (math.cos(alpha) * math.cos(beta) - math.cos(gamma)) / (sin_a * sin_b)
        inv_d_sq = (
            h * h * astar * astar + k * k * bstar * bstar + l * l * cstar * cstar
            + 2.0 * h * k * astar * bstar * cos_gamma_star
            + 2.0 * h * l * astar * cstar * cos_beta_star
            + 2.0 * k * l * bstar * cstar * cos_alpha_star
        )
        if not math.isfinite(inv_d_sq) or inv_d_sq <= 0.0:
            raise ValueError(f"Не удалось вычислить d-spacing для {self.crystal_name} {hkl}")
        return 1.0 / math.sqrt(inv_d_sq)

    def structure_factor(self, hkl: tuple[int, int, int], energy_keV: float) -> complex:
        """Return XrayLib's complex unit-cell F_H at the supplied energy (keV)."""
        xraylib = _load_xraylib()
        h, k, l = (int(value) for value in hkl)
        return complex(xraylib.Crystal_F_H_StructureFactor(
            self.crystal_data, float(energy_keV), h, k, l, 1.0, 1.0
        ))


def _validate_max_index(max_index: int) -> int:
    if isinstance(max_index, bool) or int(max_index) != max_index or int(max_index) < 1:
        raise ValueError("max_index должен быть целым числом не меньше 1")
    if int(max_index) > 30:
        raise ValueError("max_index не должен превышать 30")
    return int(max_index)


@lru_cache(maxsize=1)
def _load_xraylib():
    try:
        return importlib.import_module("xraylib")
    except ImportError as exc:
        raise RuntimeError(
            "Для материалов из базы кристаллов требуется установленный пакет xraylib. "
            "Модель CaF2 доступна без него."
        ) from exc


@lru_cache(maxsize=128)
def _load_xraylib_crystal(crystal_name: str) -> XraylibCrystalModel:
    xraylib = _load_xraylib()
    data = xraylib.Crystal_GetCrystal(str(crystal_name))
    if not data:
        raise ValueError(f"Кристалл {crystal_name!r} отсутствует в базе XrayLib")
    required = {"a", "b", "c", "alpha", "beta", "gamma"}
    missing = required.difference(data)
    if missing:
        raise ValueError(
            f"Запись XrayLib для {crystal_name!r} не содержит параметры ячейки {sorted(missing)}"
        )
    name = str(data.get("name") or crystal_name)
    return XraylibCrystalModel(
        model_id=f"xraylib:{crystal_name}", crystal_name=str(crystal_name),
        crystal_data=data, display_name=f"{name} · XrayLib",
    )


CA_F2_FLUORITE = CubicFccCrystal(
    model_id="caf2_fluorite",
    display_name="CaF₂ · флюорит",
    lattice_a_default_A=5.4635,
)

_CRYSTAL_MODELS: dict[str, CrystalReflectionModel] = {CA_F2_FLUORITE.model_id: CA_F2_FLUORITE}


def xraylib_available() -> bool:
    try:
        _load_xraylib()
        return True
    except (ImportError, RuntimeError):
        return False


def available_crystal_models() -> tuple[tuple[str, str], ...]:
    """Return UI-ready (stable ID, display label) entries.

    The built-in CaF2 model is always available. Additional entries are read
    directly from XrayLib; no list of materials is copied into this project.
    """
    entries = [(CA_F2_FLUORITE.model_id, CA_F2_FLUORITE.display_name)]
    if xraylib_available():
        xraylib = _load_xraylib()
        for crystal_name in sorted(str(name) for name in xraylib.Crystal_GetCrystalsList()):
            try:
                model = _load_xraylib_crystal(crystal_name)
            except (KeyError, TypeError, ValueError):
                continue
            entries.append((model.model_id, model.display_name))
    return tuple(entries)


def register_crystal_model(model: CrystalReflectionModel, *, replace: bool = False) -> None:
    """Register an explicit application-specific crystal plug-in."""
    model_id = str(getattr(model, "model_id", "")).strip()
    if not model_id:
        raise ValueError("Модель кристалла должна иметь непустой model_id")
    if not isinstance(model, CrystalReflectionModel):
        raise TypeError("Модель должна реализовывать allowed_reflections() и d_spacing_A()")
    if model_id in _CRYSTAL_MODELS and not replace:
        raise ValueError(f"Модель кристалла {model_id!r} уже зарегистрирована")
    _CRYSTAL_MODELS[model_id] = model


def get_crystal_model(model_id: str) -> CrystalReflectionModel:
    key = str(model_id).strip()
    if key in _CRYSTAL_MODELS:
        return _CRYSTAL_MODELS[key]
    if key.startswith("xraylib:"):
        crystal_name = key.partition(":")[2].strip()
        if crystal_name:
            return _load_xraylib_crystal(crystal_name)
    raise ValueError(f"Неизвестная теоретическая модель кристалла: {model_id!r}")
