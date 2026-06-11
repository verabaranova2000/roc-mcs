# specs.py
from dataclasses import dataclass
from typing import Callable

from roc_mcs.fitting.models import US_MODELS, NO_US_MODELS



@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    func: Callable
    param_names: tuple[str, ...]
    guess_fn: Callable
    bounds_fn: Callable

MODEL_SPECS = {
    "gauss_us": ModelSpec(
        name="gauss_us",
        func=US_MODELS["gauss"],
        param_names=("S", "theta0", "sigma", "Delta"),
        guess_fn=estimate_gauss_us_p0,
        bounds_fn=gauss_us_bounds_from_guess_us,
    ),
    "gauss": ModelSpec(
        name="gauss",
        func=NO_US_MODELS["gauss"],
        param_names=("S", "theta0", "sigma"),
        guess_fn=estimate_gauss_core_p0,
        bounds_fn=gauss_us_bounds_from_guess,
    ),
}    