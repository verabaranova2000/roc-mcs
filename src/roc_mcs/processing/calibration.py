def calibrate_phase_scan(phase_scan, angle_scale):
    """
    Добавляет калиброванную угловую ось.
    """
    phase_scan = phase_scan.copy()
    phase_scan["theta_axis"] = (phase_scan["s_axis"] * angle_scale)
    phase_scan["angle_scale"] = angle_scale
    return phase_scan
