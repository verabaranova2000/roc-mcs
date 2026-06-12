# params.py
import numpy as np
from dataclasses import dataclass
from roc_mcs.fitting.analysis import estimate_fwhm

# ======================================
# Метаданные
# ======================================
PARAMETER_INFO = {
    "S": {"symbol": "S", "unit": None},
    "theta0": {"symbol": "θ₀", "unit": "arcsec"},
    "sigma": {"symbol": "σ", "unit": "arcsec"},

    "gamma": {"symbol": "γ", "unit": "arcsec"},
    "eta": {"symbol": "η", "unit": None},    
    "H": {"symbol": "H", "unit": "arcsec"},
    
    "lam": {"symbol": "λ", "unit": "arcsec⁻¹"},

    "beta_Gl": {"symbol": "β_Gl", "unit": "arcsec"},
    "beta_Cl": {"symbol": "β_Cl", "unit": "arcsec"},
    "beta_Gr": {"symbol": "β_Gr", "unit": "arcsec"},
    "beta_Cr": {"symbol": "β_Cr", "unit": "arcsec"},    
    
    "Delta": {"symbol": "Δ", "unit": "arcsec"},
}


# ==================================================
# Контейнер
# ==================================================
@dataclass(slots=True)
class FitParameter:
    key: str
    symbol: str      
    value: float
    error: float | None = None  
    unit: str | None = None

# Пример
# Parameter(
#     key="theta0",
#     symbol="θ₀",
#     value=1.37,
#     error=0.02,
# )  

# ==================================================
# Конструктор
# ==================================================
def build_parameters(param_names, values, errors=None):
    params = {}
    for i, name in enumerate(param_names):
        info = PARAMETER_INFO[name]
        err = None if errors is None else errors[i]
        params[name] = FitParameter(
            key=name,
            symbol=info["symbol"],
            value=float(values[i]),
            error=None if err is None else float(err),
            unit=info.get("unit"),
        )
    return params
    
# ==================================================
# Оценки стартовых параметров
# ==================================================
# --- Lorentzian ---
def lorentz_guess(theta, intensity):
    S0 = np.trapezoid(intensity, theta)
    theta0 = theta[np.argmax(intensity)]
    gamma0 = estimate_fwhm(theta, intensity) / 2
    if not np.isfinite(gamma0) or gamma0 <= 0:
        gamma0 = (theta.max() - theta.min()) / 20
    return S0, theta0, gamma0
    
# --- Gaussian ---
def gauss_guess(theta, intensity):
    S0 = np.trapezoid(intensity, theta)
    theta0 = np.trapezoid(theta * intensity, theta) / S0
    sigma0 = np.sqrt(np.trapezoid((theta - theta0) ** 2 * intensity, theta) / S0)
    return S0, theta0, sigma0

def gauss_us_guess(theta, intensity):
    S0, theta0, sigma0 = gauss_guess(theta, intensity)
    Delta0 = 0.0
    return S0, theta0, sigma0, Delta0

# --- Voigt ---
def voigt_guess(theta, intensity):
    S0 = np.trapezoid(intensity, theta)
    theta0 = np.trapezoid(theta * intensity, theta) / S0
    sigma0 = estimate_fwhm(theta, intensity) / 2.355
    gamma0 = sigma0
    return (S0, theta0, sigma0, gamma0)
    
# --- pseudo-Voigt ---
def pvoigt_guess(theta, intensity):
    S0 = np.trapezoid(intensity, theta)
    theta0 = np.trapezoid(theta * intensity, theta) / S0
    H0 = estimate_fwhm(theta, intensity)
    if not np.isfinite(H0):     # запасной вариант
        sigma0 = np.sqrt(np.trapezoid((theta - theta0)**2 * intensity, theta) / S0)
        H0 = 2*np.sqrt(2*np.log(2))*sigma0
    eta0 = 0.5
    return S0, theta0, H0, eta0    

# --- EMG (экспоненциальный хвост) ---
def emg_guess(theta, intensity):
    S0 = np.trapezoid(intensity, theta)
    theta0 = np.trapezoid(theta * intensity, theta) / S0
    sigma0 = np.std(theta)
    gamma0 = sigma0
    lam0 = 0.1  # слабая асимметрия стартово
    return S0, theta0, sigma0, gamma0, lam0

# --- split-Voigt ---
def split_voigt_guess(theta, intensity):
    theta = np.asarray(theta)
    intensity = np.asarray(intensity)

    S0 = np.trapezoid(intensity, theta)          # площадь пика
    theta0 = theta[np.argmax(intensity)]         # центр по максимуму

    H0 = estimate_fwhm(theta, intensity)
    if not np.isfinite(H0):
        H0 = (theta.max() - theta.min()) / 10    # fallback: безопасная стартовая оценка
    beta_Gl0 = H0 / 2   # для старта задаём симметричную форму слева/справа
    beta_Cl0 = H0 / 2
    beta_Gr0 = H0 / 2
    beta_Cr0 = H0 / 2
    return (S0, theta0, beta_Gl0, beta_Cl0, beta_Gr0, beta_Cr0)   

     
# ==================================================
# Границы
# ==================================================
def lorentz_bounds(p0):
    S0, theta0, gamma0 = p0
    return {
        "S": (0.5 * S0, 2.0 * S0),
        "theta0": (theta0 - 5, theta0 + 5),
        "gamma": (1e-6, 10 * gamma0),
    }
     
def gauss_bounds(p0):
    S0, theta0, sigma0 = p0
    return {
        "S": (0.5 * S0, 1.5 * S0),
        "theta0": (theta0 - 5, theta0 + 5),
        "sigma": (0.2, sigma0),
    }
    
def gauss_us_bounds(p0):
    S0, theta0, sigma0, _ = p0
    return {
        "S": (0.5 * S0, 1.5 * S0),
        "theta0": (theta0 - 5, theta0 + 5),
        "sigma": (0.2, sigma0),
        "Delta": (1, 20.0),   # 🚨 Не меньше шага theta!
    }

def voigt_bounds(p0):
    S0, theta0, sigma0, gamma0 = p0
    return {
        "S": (0.5*S0, 2*S0),
        "theta0": (theta0-5, theta0+5),
        "sigma": (0.1*sigma0, 5*sigma0),
        "gamma": (0.1*gamma0, 5*gamma0),
    }    

def pvoigt_bounds(p0):
    S0, theta0, H0, eta0 = p0
    return {
        "S": (0.5*S0, 1.5*S0),
        "theta0": (theta0-5, theta0+5),
        "H": (0.2, 5*H0),
        "eta": (0.0, 1.0),
    }
    
def emg_bounds(p0):
    S0, theta0, sigma0, gamma0, lam0 = p0
    return {
        "S": (0.5*S0, 1.5*S0),
        "theta0": (theta0-5, theta0+5),
        "sigma": (0.2, 5*sigma0),
        "gamma": (0.2, 5*gamma0),
        "lam": (0.0, 5.0),
    }    

def split_voigt_bounds(p0):
    S0, theta0, bGl, bCl, bGr, bCr = p0
    eps = 1      # шаг theta-сетки
    return {
        "S": (0.2 * S0, 5.0 * S0),
        "theta0": (theta0 - 5, theta0 + 5),

        "beta_Gl": (eps, 10.0 * bGl),
        "beta_Cl": (eps, 10.0 * bCl),
        "beta_Gr": (eps, 10.0 * bGr),
        "beta_Cr": (eps, 10.0 * bCr),
    }

    
def pack_bounds(param_names, bounds_dict):
    lower = [bounds_dict[name][0] for name in param_names]
    upper = [bounds_dict[name][1] for name in param_names]
    return (np.array(lower, dtype=float), np.array(upper, dtype=float))