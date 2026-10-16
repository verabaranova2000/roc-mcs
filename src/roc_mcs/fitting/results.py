# results.py
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from roc_mcs.fitting.params import FitParameter




@dataclass(slots=True)
class FitResult:
    """
    Результат подгонки одной ROC-кривой: параметры, метрики, covariance, y_fit.
    """
    model: str
    parameters: dict[str, FitParameter]
    metrics: dict[str, float]
    y_fit_global: np.ndarray | None = None
    y_fit_local: np.ndarray | None = None
    covariance: np.ndarray | None = None
    success: bool = True
    message: str = ""

    @property
    def y_fit(self):
        if self.y_fit_local is not None:
            return self.y_fit_local
        if self.y_fit_global is not None:
            return self.y_fit_global
        return None    
    



@dataclass(slots=True)
class FitScanResult:
    """
    Результат подгонки пространственного скана: 
    каталоги пиков, QC по пикселям, карты параметров и метаданные запуска.
    """
    peak_catalog: pd.DataFrame
    fitted_peak_catalog: pd.DataFrame
    pixel_fit_catalog: pd.DataFrame
    theta: np.ndarray
    theta_valid: np.ndarray
    image_shape: tuple[int, int]
    model_id: str
    backend: str
    component_param_names: tuple[str, ...]
    pixel_param_names: tuple[str, ...] = ()
    fit_cube: np.ndarray | None = None
    residual_cube: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    run_id: str | None = None
    fit_config: dict[str, Any] = field(default_factory=dict)
    model_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.theta = np.asarray(self.theta, dtype=float)
        self.theta_valid = np.asarray(self.theta_valid, dtype=bool)
        self.image_shape = tuple(map(int, self.image_shape))
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("image_shape должен быть парой положительных размеров (Y, X)")
        self.component_param_names = tuple(self.component_param_names)
        self.pixel_param_names = tuple(self.pixel_param_names)
        if self.theta.ndim != 1 or self.theta_valid.shape != self.theta.shape:
            raise ValueError("theta и theta_valid должны быть одномерными массивами одинаковой длины")
        if self.fit_cube is not None and np.shape(self.fit_cube) != (*self.image_shape, len(self.theta)):
            raise ValueError("fit_cube имеет несовместимую форму")
        if self.residual_cube is not None and np.shape(self.residual_cube) != (*self.image_shape, len(self.theta)):
            raise ValueError("residual_cube имеет несовместимую форму")
        if self.valid_mask is not None:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
            allowed_masks = {self.image_shape, (len(self.theta),), (*self.image_shape, len(self.theta))}
            if self.valid_mask.shape not in allowed_masks:
                raise ValueError(f"valid_mask имеет несовместимую форму {self.valid_mask.shape}")
        if not isinstance(self.peak_catalog, pd.DataFrame):
            raise TypeError("peak_catalog должен быть pandas.DataFrame")
        if not isinstance(self.fitted_peak_catalog, pd.DataFrame) or not isinstance(self.pixel_fit_catalog, pd.DataFrame):
            raise TypeError("Каталоги результатов должны быть pandas.DataFrame")

    @property
    def shape(self) -> tuple[int, int]:
        return self.image_shape

    def parameter_map(self, name: str, *, scope: str = "component", component: int | None = None) -> np.ndarray:
        """Карта параметра: component → (Y,X,K), pixel → (Y,X); component выбирает слот пика."""
        ny, nx = self.image_shape
        if scope == "pixel":
            if component is not None:
                raise ValueError("component нельзя задавать для pixel-параметра")
            if name not in self.pixel_fit_catalog.columns:
                raise KeyError(f"Неизвестный pixel-параметр {name!r}")
            out = np.full((ny, nx), np.nan, dtype=float)
            for row in self.pixel_fit_catalog[["x", "y", name]].itertuples(index=False, name=None):
                x, y, value = row
                out[int(y), int(x)] = _as_float(value)
            return out
        if scope != "component":
            raise ValueError("scope должен быть 'component' или 'pixel'")
        if name not in self.fitted_peak_catalog.columns:
            raise KeyError(f"Неизвестный параметр компоненты {name!r}")
        if self.peak_catalog.empty:
            kmax = 0
        elif "n_components" in self.peak_catalog:
            counts = pd.to_numeric(self.peak_catalog["n_components"], errors="coerce").to_numpy()
            kmax = int(np.nanmax(counts)) if np.isfinite(counts).any() else 0
        elif "component_id" in self.peak_catalog:
            ids = pd.to_numeric(self.peak_catalog["component_id"], errors="coerce").to_numpy()
            kmax = int(np.nanmax(ids)) + 1 if np.isfinite(ids).any() else 0
        else:
            kmax = int(self.fitted_peak_catalog.get("component_id", pd.Series(dtype=float)).max()) + 1 if not self.fitted_peak_catalog.empty else 0
        if component is not None:
            component = int(component)
            if component < 0 or component >= kmax:
                raise IndexError(f"component={component} вне диапазона 0..{kmax - 1}")
            out = np.full((ny, nx), np.nan, dtype=float)
        else:
            out = np.full((ny, nx, kmax), np.nan, dtype=float)
        for row in self.fitted_peak_catalog[["x", "y", "component_id", name]].itertuples(index=False, name=None):
            x, y, ci, value = row
            ci = int(ci)
            if component is not None:
                if ci == component:
                    out[int(y), int(x)] = _as_float(value)
            elif 0 <= ci < kmax:
                out[int(y), int(x), ci] = _as_float(value)
        return out


def _as_float(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan



@dataclass(slots=True)
class ExperimentArtifact:
    """Контейнер результата эксперимента — пока оставляем как есть."""
    roc_map: dict
    fit_tables: dict
    fit_results: dict
    trajectory_results: dict    # сырой результат анализа траекторий
    trajectory_store: dict        # готовые ScalarTrajectory для графиков
    metrics: dict
    metadata: dict
    model_config: dict
    profile_moments: pd.DataFrame | None = None
    control_log: dict | None = None    
    output_files: list[str] | None = None    




