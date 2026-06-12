# analysis.py
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class AnalysisSpec:
    derived_fn: Callable
    primary_plots: list[str]


MODEL_ANALYSIS = {
    "lorentz": {
        "fwhm": lambda r, theta: 2 * r.parameters["gamma"].value,
    
        "derived": {},
        "plot_groups": [
            [("theta0", "θ₀"), ("gamma", "γ"), ("S", "S")],
        ],
    },    
    "gauss": {
        "fwhm": lambda r, theta: 2 * np.sqrt(2 * np.log(2)) * r.parameters["sigma"].value,
        "derived": {},
        "plot_groups": [
            [("theta0", "θ₀"), ("sigma", "σ"), ("S", "S")],
        ],
    },

    "gauss_us": {
        "fwhm": lambda r, theta: 2 * np.sqrt(2 * np.log(2)) * r.parameters["sigma"].value,
        "derived": {},
        "plot_groups": [
            [("theta0", "θ₀"), ("sigma", "σ"), ("Delta", "Δ"), ("S", "S")],
        ],
    },

     "voigt": {
        "fwhm": lambda r, theta: (
            0.5346 * 2 * r.parameters["gamma"].value +
            np.sqrt(0.2166 * (2 * r.parameters["gamma"].value)**2 +
                    (2.3548 * r.parameters["sigma"].value)**2)
        ),  # Формула Olivero–Longbothum (1977)
        "derived": {},
        "plot_groups": [
            [("theta0", "θ₀"), ("sigma", "σ"), ("gamma", "γ"), ("S", "S")],
        ],
    },   
    "pvoigt": {
        "fwhm": lambda r, theta: r.parameters["H"].value,
        "derived": {},
        "plot_groups": [
            [("theta0", "θ₀"), ("H", "H"), ("S", "S"), ("eta", "η")],
        ],
    },

    "split_voigt": {
        "fwhm": lambda r, theta: estimate_fwhm(theta, r.y_fit),
        "derived": {
            "beta_G_mean": lambda p: 0.5 * (p["beta_Gl"] + p["beta_Gr"]),
            "beta_C_mean": lambda p: 0.5 * (p["beta_Cl"] + p["beta_Cr"]),
            "beta_G_asym": lambda p: p["beta_Gr"] - p["beta_Gl"],
            "beta_C_asym": lambda p: p["beta_Cr"] - p["beta_Cl"],
        },
        "plot_groups": [
            [("theta0", "θ₀"), ("S", "S")],
            [("beta_G_mean", "β_G mean"), ("beta_C_mean", "β_C mean")],
            [("beta_G_asym", "β_G asym"), ("beta_C_asym", "β_C asym")],
        ],
    },
}    


# ==================================================
# Вычисляемые величины
# ==================================================
def estimate_fwhm(theta, intensity):
    theta = np.asarray(theta)
    I = np.asarray(intensity)
    half = 0.5 * np.max(I)
    idx = np.where(I >= half)[0]
    if len(idx) < 2:
        return np.nan
    return theta[idx[-1]] - theta[idx[0]]