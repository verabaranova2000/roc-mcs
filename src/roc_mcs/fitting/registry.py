# registry.py
from dataclasses import dataclass
from typing import Callable
from roc_mcs.fitting.params import (
    lorentz_guess,
    gauss_guess,
    gauss_us_guess,
    voigt_guess,
    pvoigt_guess,
    emg_guess,
    split_voigt_guess,
)
from roc_mcs.fitting.params import (
    lorentz_bounds,
    gauss_bounds,
    gauss_us_bounds,
    voigt_bounds,
    pvoigt_bounds,
    emg_bounds,
    split_voigt_bounds,
)
from roc_mcs.fitting.models import (
    model_lorentz,
    model_gauss,
    model_gauss_us,
    model_voigt,
    model_pvoigt,
    model_emg,
    model_split_voigt
)

from roc_mcs.fitting.models import (
    batch_model_lorentz,
    batch_model_gauss,
    batch_model_voigt,
    batch_model_pvoigt,
    batch_model_emg,
    batch_model_split_voigt
)

from roc_mcs.fitting.models import (
    batch_jac_pvoigt
)

from roc_mcs.fitting.models import (
    # model_lorentz_jax,
    # model_gauss_jax,
    # model_gauss_us_jax,
    # model_voigt_jax,
    jax_model_pvoigt,
    # model_emg_jax,
    # model_split_voigt_jax
)

from roc_mcs.fitting.derived import (
    estimate_fwhm_from_curve,
    fwhm_lorentz,
    fwhm_gauss,
    fwhm_voigt,
    fwhm_pvoigt,
    #fwhm_emg,
    #fwhm_split_voigt
)

from roc_mcs.fitting.derived import (
    grad_fwhm_lorentz,
    grad_fwhm_gauss,
    grad_fwhm_voigt,
    grad_fwhm_pvoigt,
    #fwhm_emg,
    #fwhm_split_voigt
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    func: Callable
    batch_func: Callable
    batch_jac_func: Callable | None
    jax_func: Callable | None
    param_names: tuple[str, ...]
    guess_fn: Callable
    bounds_fn: Callable

    # Поведение самого ModelSpec: Возьми вот этот ModelSpec и выполни его контракт guess_fn / bounds_fn
    def make_initial_guess(self, theta, intensity):
        """
        Возвращает начальные значения параметров — dict:
            param_name -> initial value
        В формате:
            {"S": ..., "theta0": ..., "sigma": ...}
        """
        values = self.guess_fn(theta, intensity)
        if len(values) != len(self.param_names):
            raise ValueError(f"{self.name}.guess_fn returned {len(values)} values, but model has {len(self.param_names)} parameters")
        return dict(zip(self.param_names, values))


    def make_bounds(self, guess_values, dtheta=None):
        """
        Возвращает границы параметров для данного ModelSpec — словарь: 
            param_name -> (lower, upper)
        dtheta передается в bounds_fn() для динамического ограничения минимальной ширины пика.
        """
        values = tuple(guess_values[name] for name in self.param_names)
        bounds = self.bounds_fn(values, dtheta=dtheta)

        missing = set(self.param_names) - set(bounds)
        if missing:
            raise ValueError(f"{self.name}.bounds_fn is missing parameters: {sorted(missing)}")
        return bounds



MODEL_SPECS = {
    "lorentz": ModelSpec(
        name="lorentz",
        func=model_lorentz,
        batch_func=batch_model_lorentz,
        batch_jac_func=None,
        jax_func=None,
        param_names=("S", "theta0", "gamma"),
        guess_fn=lorentz_guess,
        bounds_fn=lorentz_bounds,
        # activation_param="S",
    ),    
    "gauss": ModelSpec(
        name="gauss",
        func=model_gauss,
        batch_func=batch_model_gauss,
        batch_jac_func=None,
        jax_func=None,
        param_names=("S", "theta0", "sigma"),
        guess_fn=gauss_guess,
        bounds_fn=gauss_bounds,
        # activation_param="S",
    ),    
    "gauss_us": ModelSpec(
        name="gauss_us",
        func=model_gauss_us,
        batch_func=model_gauss_us,   # ⚠️
        batch_jac_func=None,
        jax_func=None,
        param_names=("S", "theta0", "sigma", "Delta"),
        guess_fn=gauss_us_guess,
        bounds_fn=gauss_us_bounds,
        # activation_param="S",
    ),
    "voigt": ModelSpec(
        name="voigt",
        func=model_voigt,
        batch_func=batch_model_voigt,
        batch_jac_func=None,
        jax_func=None,
        param_names=("S", "theta0", "sigma", "gamma"),
        guess_fn=voigt_guess,
        bounds_fn=voigt_bounds,
        # activation_param="S",
    ),    
    "pvoigt": ModelSpec(
        name="pvoigt",
        func=model_pvoigt,
        batch_func=batch_model_pvoigt,
        batch_jac_func=batch_jac_pvoigt,
        jax_func=jax_model_pvoigt,
        param_names=("S","theta0","H","eta"),
        guess_fn=pvoigt_guess,
        bounds_fn=pvoigt_bounds,
        # activation_param="S",
    ),
    "emg": ModelSpec(
        name="emg",
        func=model_emg,
        batch_func=batch_model_emg,
        batch_jac_func=None,
        jax_func=None,
        param_names=("S","theta0","sigma","gamma","lam"),
        guess_fn=emg_guess,
        bounds_fn=emg_bounds,
        # activation_param="S",
    ),    
    "split_voigt": ModelSpec(
        name="split_voigt",
        func=model_split_voigt,
        batch_func=batch_model_split_voigt,
        batch_jac_func=None,
        jax_func=None,
        param_names=("S", "theta0", "beta_Gl", "beta_Cl", "beta_Gr", "beta_Cr"),
        guess_fn=split_voigt_guess,
        bounds_fn=split_voigt_bounds,
        # activation_param="S",
    )
}   



# ======================================
# Валидация моделей
# ======================================
def validate_model(model_name):
    if model_name not in MODEL_SPECS:
        available = ", ".join(sorted(MODEL_SPECS))

        raise ValueError(
            f"Неизвестная модель '{model_name}'. "
            f"Доступные модели: {available}."
        )
    



@dataclass(frozen=True)
class DerivedSpec:
    value_func: Callable
    grad_func: Callable | None = None
    needs_curve: bool = False


DERIVED_SPECS = {
    "lorentz": {
        "FWHM": DerivedSpec(
            value_func=fwhm_lorentz,
            grad_func=grad_fwhm_lorentz,
            needs_curve=False,
        )
    },
    "gauss": {
        "FWHM": DerivedSpec(
            value_func=fwhm_gauss,
            grad_func=grad_fwhm_gauss,
            needs_curve=False,
        )
    },   
    "voigt": {
        "FWHM": DerivedSpec(
            value_func=fwhm_voigt,
            grad_func=grad_fwhm_voigt,
            needs_curve=False,
        )
    },     
    "pvoigt": {
        "FWHM": DerivedSpec(
            value_func=fwhm_pvoigt,
            grad_func=grad_fwhm_pvoigt,
            needs_curve=False,
        )
    },
    "split_voigt": {
        "FWHM": DerivedSpec(
            value_func=estimate_fwhm_from_curve,
            grad_func=None,
            needs_curve=True,
        )
    },
}