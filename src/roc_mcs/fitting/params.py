# params.py
import numpy as np
from dataclasses import dataclass

# ======================================
# Метаданные
# ======================================
PARAMETER_INFO = {
    "S": {"symbol": "S", "unit": None},
    "theta0": {"symbol": "θ₀", "unit": "arcsec"},
    "sigma": {"symbol": "σ", "unit": "arcsec"},
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
# --- Gaussian ---
def estimate_gauss_p0(theta, intensity):
    S0 = np.trapezoid(intensity, theta)
    theta0 = np.trapezoid(theta * intensity, theta) / S0
    sigma0 = np.sqrt(np.trapezoid((theta - theta0) ** 2 * intensity, theta) / S0)
    return S0, theta0, sigma0

def estimate_gauss_us_p0(theta, intensity):
    S0, theta0, sigma0 = estimate_gauss_p0(theta, intensity)
    Delta0 = 0.0
    return S0, theta0, sigma0, Delta0



# ==================================================
# Границы
# ==================================================
def bounds_from_guess_gauss_us(p0):
    S0, theta0, sigma0, _ = p0
    return {
        "S": (0.5 * S0, 1.5 * S0),
        "theta0": (theta0 - 5, theta0 + 5),
        "sigma": (0.2, sigma0),
        "Delta": (0.0, 20.0),
    }

def pack_bounds(param_names, bounds_dict):
    lower = [bounds_dict[name][0] for name in param_names]
    upper = [bounds_dict[name][1] for name in param_names]
    return (np.array(lower, dtype=float), np.array(upper, dtype=float))

