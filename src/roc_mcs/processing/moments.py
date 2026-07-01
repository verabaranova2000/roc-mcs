from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(slots=True)
class ProfileMoments:
    area: float

    centroid: float

    variance: float
    sigma: float

    skewness: float
    kurtosis: float

    fw50_int: float
    fw80_int: float
    fw90_int: float



# --- 1. Вспомогательная функция для CDF ---
# Она понадобится для интегральных ширин.

def cumulative_trapezoid_manual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Накопленный интеграл методом трапеций.

    Возвращает массив той же длины:
        cdf[i] = ∫_{x0}^{x[i]} y(x) dx
    """
    dx = np.diff(x)
    increments = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(increments)])

    
# --- 2. Основная функция ---
def compute_profile_moments(
    theta: np.ndarray,
    intensity: np.ndarray,
) -> ProfileMoments:
    
    try:
        trapz = np.trapezoid
    except AttributeError:
        trapz = np.trapz
    
    # --- Шаг 1. Проверка входных данных ---
    theta = np.asarray(theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta должен быть одномерным")
    if intensity.ndim != 1:
        raise ValueError("intensity должен быть одномерным")
    if len(theta) != len(intensity):
        raise ValueError("Размерности theta и intensity не совпадают")   
        
    # --- Шаг 2. NaN ---
    mask = np.isfinite(theta) & np.isfinite(intensity)

    theta = theta[mask]
    intensity = intensity[mask]

    # --- Шаг 3. Отрицательные интенсивности ---
    # Очень важно. Для моментов интенсивность должна быть неотрицательной.
    intensity = np.clip(intensity, 0.0, None)

    # --- Шаг 4. Интегральная интенсивность ---
    area = trapz(intensity, theta)
    # Проверка вырожденного случая.
    if area <= 0:
        return ProfileMoments(
            area=0.0,
            centroid=np.nan,
            variance=np.nan,
            sigma=np.nan,
            skewness=np.nan,
            kurtosis=np.nan,
            fw50_int=np.nan,
            fw80_int=np.nan,
            fw90_int=np.nan,
        )

    # --- Шаг 5. Центроид ---
    centroid = trapz(theta * intensity, theta) / area

    # --- Шаг 6. Центральные моменты --- 
    delta = theta - centroid
    variance = trapz(delta**2 * intensity, theta) / area
    sigma = np.sqrt(variance)    

    # --- Шаг 7. Защита от sigma≈0 ---
    if sigma <= 0:
        skewness = np.nan
        kurtosis = np.nan
    else:    
        # --- skewness γ₁ = m₃/σ³      
        m3 = trapz(delta**3 * intensity, theta) / area
        skewness = m3 / sigma**3
        # --- excess kurtosis γ² = m₄/σ⁴ − 3
        m4 = trapz(delta**4 * intensity, theta) / area 
        kurtosis = m4 / sigma**4 - 3.0
    
    # --- Шаг 8. Интегральные ширины. Строим CDF. ---       
    cdf = cumulative_trapezoid_manual(intensity, theta)
    if cdf[-1] <= 0:
        raise ValueError("Невозможно нормировать CDF: интегральная интенсивность равна нулю.")    
    cdf /= cdf[-1]  
    # --- Шаг 9. Квантили. Очень удобно через интерполяцию. ---
    q05 = np.interp(0.05, cdf, theta)
    q10 = np.interp(0.10, cdf, theta)
    q25 = np.interp(0.25, cdf, theta)
    q75 = np.interp(0.75, cdf, theta)
    q90 = np.interp(0.90, cdf, theta)
    q95 = np.interp(0.95, cdf, theta)    
    # --- Шаг 10. Ширины ---
    fw50_int = q75 - q25
    fw80_int = q90 - q10
    fw90_int = q95 - q05
    
    return ProfileMoments(
        area=float(area),

        centroid=float(centroid),

        variance=float(variance),
        sigma=float(sigma),

        skewness=float(skewness),
        kurtosis=float(kurtosis),

        fw50_int=float(fw50_int),
        fw80_int=float(fw80_int),
        fw90_int=float(fw90_int),
    )       


def compute_roc_map_moments(roc_map) -> pd.DataFrame:
    """
    Вычисляет моментные характеристики для всех рок-кривых эксперимента.
    """
    theta = np.asarray(roc_map["theta_axis"], dtype=float)
    intensity_map = np.asarray(roc_map["intensity"], dtype=float)
    time_s = np.asarray(roc_map["time_s"], dtype=float)

    rows = []
    for scan_id, (t, intensity) in enumerate(zip(time_s, intensity_map), start=1):
        moments = compute_profile_moments(theta=theta, intensity=intensity)

        # положение и высота глобального максимума
        idx_max = np.argmax(intensity)
        peak_theta = theta[idx_max]
        peak_intensity = intensity[idx_max]

        rows.append({
            "scan_id": scan_id,
            "time_s": t,

            "peak_theta": float(peak_theta),
            "peak_intensity": float(peak_intensity),

            "area": moments.area,

            "centroid": moments.centroid,
            "centroid_shift": moments.centroid - peak_theta,

            "variance": moments.variance,
            "sigma": moments.sigma,

            "skewness": moments.skewness,
            "kurtosis": moments.kurtosis,

            "fw50_int": moments.fw50_int,
            "fw80_int": moments.fw80_int,
            "fw90_int": moments.fw90_int,
        })
    df = pd.DataFrame(rows)

    column_order = [
        "scan_id",
        "time_s",

        "peak_theta",
        "peak_intensity",

        "area",

        "centroid",
        "centroid_shift",

        "variance",
        "sigma",

        "skewness",
        "kurtosis",

        "fw50_int",
        "fw80_int",
        "fw90_int",
    ]
    return df[column_order]



def enrich_profile_moments_with_control_log(
    profile_moments: pd.DataFrame,
    control_log: dict,
) -> pd.DataFrame:
    """
    Добавляет к таблице profile moments силу и давление,
    соответствующие моменту старта каждого скана.
    """
    if control_log is None:
        return profile_moments

    if "scan_points" not in control_log:
        raise ValueError("control_log не содержит scan_points")

    sp = control_log["scan_points"].copy()

    required_cols = {"scan_id", "force", "pressure_mpa"}
    missing = required_cols - set(sp.columns)
    if missing:
        raise ValueError(f"В scan_points не хватает столбцов: {sorted(missing)}")

    df = profile_moments.copy()
    df = df.merge(
        sp[["scan_id", "force", "pressure_mpa"]],
        on="scan_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(profile_moments):
        raise ValueError("Не все scan_id удалось сопоставить между profile_moments и scan_points")

    column_order = [
        "scan_id",
        "time_s",
        "force_kg",
        "pressure_MPa",
        "peak_theta",
        "peak_intensity",
        "area",
        "centroid",
        "centroid_shift",
        "variance",
        "sigma",
        "skewness",
        "kurtosis",
        "fw50_int",
        "fw80_int",
        "fw90_int",
    ]

    df = df.rename(columns={
        "force": "force_kg",
        "pressure_mpa": "pressure_MPa",
    })

    return df[column_order]