def calibrate_roc_map(roc_map, angle_scale):
    """
    Добавляет калиброванную угловую ось.
    """
    roc_map = roc_map.copy()
    roc_map["theta_axis"] = (roc_map["s_axis"] * angle_scale)
    roc_map["angle_scale"] = angle_scale
    return roc_map
