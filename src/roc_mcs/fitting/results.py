# results.py
from dataclasses import dataclass
import numpy as np
import pandas as pd

from roc_mcs.fitting.params import FitParameter




@dataclass(slots=True)
class FitResult:
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
class ExperimentArtifact:
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
  