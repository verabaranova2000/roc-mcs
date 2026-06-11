# models.py
import numpy as np
from scipy.special import voigt_profile
from dataclasses import dataclass
from typing import Callable
from roc_mcs.fitting.kernels import ultrasound_RC


# ==================================================
# Base profiles
# ==================================================
"""
Нормированные профили:   ∫f(θ)dθ = 1

np.trapezoid(...)
    Численная перенормировка, потому что интегрируем не по всей прямой, 
    а по конечному диапазону theta, т.е. хвосты обрезаются -> площадь не дооценивается
"""

def profile_lorentz(th, gamma):
    L = (1 / np.pi) * gamma / (th**2 + gamma**2)
    L /= np.trapezoid(L, th)
    return L

def profile_gauss(th, sigma):
    G = np.exp(-(th**2) / (2 * sigma**2))
    G /= np.trapezoid(G, th)
    return G

def profile_voigt(th, sigma, gamma):
    V = voigt_profile(th, sigma, gamma)
    V /= np.trapezoid(V, th)
    return V


# ==================================================
# No-US models
# ==================================================
def model_no_us_lorentz(theta, S, theta0, gamma):
    return S * profile_lorentz(theta - theta0, gamma)

def model_no_us_gauss(theta, S, theta0, sigma):
    return S * profile_gauss(theta - theta0, sigma)

def model_no_us_voigt(theta, S, theta0, sigma, gamma):
    return S * profile_voigt(theta - theta0, sigma, gamma)

# ==================================================
# US models
# ==================================================
def model_us_lorentz(theta, S, theta0, gamma, Delta):
    I0 = model_no_us_lorentz(theta, S, theta0, gamma)
    return ultrasound_RC(I0, theta, Delta)

def model_us_gauss(theta, S, theta0, sigma, Delta):
    I0 = model_no_us_gauss(theta, S, theta0, sigma)
    return ultrasound_RC(I0, theta, Delta)

def model_us_voigt(theta, S, theta0, sigma, gamma, Delta):
    I0 = model_no_us_voigt(theta, S, theta0, sigma, gamma)
    return ultrasound_RC(I0, theta, Delta)

# ==================================================
# Registry
# ==================================================
NO_US_MODELS = {
    "lorentz": model_no_us_lorentz,
    "gauss":   model_no_us_gauss,
    "voigt":   model_no_us_voigt,
}

US_MODELS = {
    "lorentz": model_us_lorentz,
    "gauss":   model_us_gauss,
    "voigt":   model_us_voigt,
}




