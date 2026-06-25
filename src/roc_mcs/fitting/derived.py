import numpy as np
from typing import Sequence

# ==================================================
# FWHM: explicit formulas
# ==================================================
def fwhm_lorentz(params: dict[str, float]) -> float:
    return 2.0 * params["gamma"]

def fwhm_gauss(params: dict[str, float]) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * params["sigma"]

def fwhm_voigt(params: dict[str, float]) -> float:
    sigma = params["sigma"]
    gamma = params["gamma"]
    return (0.5346 * (2.0 * gamma) + np.sqrt(0.2166 * (2.0 * gamma) ** 2 + (2.3548 * sigma) ** 2))

def fwhm_pvoigt(params: dict[str, float]) -> float:
    return params["H"]

# ==================================================
# grad FWHM: explicit formulas
# ==================================================
def grad_fwhm_lorentz(params: dict[str, float], param_order: Sequence[str]) -> np.ndarray:
    g = np.zeros(len(param_order), dtype=float)
    g[param_order.index("gamma")] = 2.0
    return g

def grad_fwhm_gauss(params: dict[str, float], param_order: Sequence[str]) -> np.ndarray:
    g = np.zeros(len(param_order), dtype=float)
    g[param_order.index("sigma")] = 2.0 * np.sqrt(2.0 * np.log(2.0))
    return g

    
def grad_fwhm_voigt(params: dict[str, float], param_order: Sequence[str]) -> np.ndarray:
    g = np.zeros(len(param_order), dtype=float)

    sigma = params["sigma"]
    gamma = params["gamma"]

    b = 2.0 * gamma
    c = 0.2166
    d = 2.3548**2
    rad = c * b * b + d * sigma * sigma
    root = np.sqrt(max(rad, 0.0))

    if root > 0:
        df_dgamma = 2.0 * (0.5346 + c * b / root)
        df_dsigma = (d * sigma) / root
    else:
        df_dgamma = 2.0 * 0.5346
        df_dsigma = 0.0

    g[param_order.index("gamma")] = df_dgamma
    g[param_order.index("sigma")] = df_dsigma
    return g

def grad_fwhm_pvoigt(params: dict[str, float], param_order: Sequence[str]) -> np.ndarray:
    g = np.zeros(len(param_order), dtype=float)
    g[param_order.index("H")] = 1.0
    return g



# ==================================================
# Numerical FWHM from curve
# ==================================================
def estimate_fwhm_from_curve(theta, intensity):
    """
    Оценка полной ширины на полувысоте (FWHM) по дискретному профилю
    с линейной интерполяцией точек пересечения уровня 0.5·I_max.
    """
    theta = np.asarray(theta, dtype=float)
    I = np.asarray(intensity, dtype=float)

    if theta.ndim != 1 or I.ndim != 1 or len(theta) != len(I):
        raise ValueError("theta and intensity must be 1D arrays of the same length")

    imax = int(np.argmax(I))
    half = 0.5 * np.max(I)

    # левая граница: ищем последний индекс слева, где I < half
    left_candidates = np.where(I[:imax] < half)[0]
    if len(left_candidates) == 0:
        return np.nan
    i1 = left_candidates[-1]
    i2 = i1 + 1

    # правая граница: ищем первый индекс справа, где I < half
    right_candidates = np.where(I[imax:] < half)[0]
    if len(right_candidates) == 0:
        return np.nan
    j2 = imax + right_candidates[0]
    j1 = j2 - 1

    # линейная интерполяция пересечения half
    x_left = theta[i1] + (half - I[i1]) * (theta[i2] - theta[i1]) / (I[i2] - I[i1])
    x_right = theta[j1] + (half - I[j1]) * (theta[j2] - theta[j1]) / (I[j2] - I[j1])
    return float(x_right - x_left)