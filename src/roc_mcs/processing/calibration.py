def calibrate_roc_map(roc_map, amplitude, reference_amplitude, reference_angle):
    """
    Калибровка угловой оси ROC-карты.

    Parameters
    ----------
    reference_angle : float
        Калибровочный коэффициент (arcsec), полученный при reference_amplitude.

    reference_amplitude : float
        Амплитуда пьезоактуатора (mVpp), для которой была выполнена калибровка.

    amplitude : float
        Амплитуда текущего эксперимента (mVpp).
    """
    scale = reference_angle * amplitude / reference_amplitude

    roc_map = roc_map.copy()
    roc_map["theta_axis"] = roc_map["s_axis"] * scale
    roc_map["calibration"] = {
        "reference_angle": reference_angle,
        "reference_amplitude": reference_amplitude,
        "amplitude": amplitude,
        "scale": scale,
    }
    return roc_map    
