"""Проверяемая передача калибровки live-серии в pipeline/build.py."""
def build_pipeline_calibration_args(calibration: dict, series_amplitude_mVpp: float) -> dict:
    """Сопоставляет Aк/Uк с legacy-аргументами pipeline; B и φк не участвуют."""
    if not isinstance(calibration, dict): raise ValueError("Калибровка не задана.")
    try:
        a = float(calibration["A"]); u_cal = float(calibration["reference_amplitude_mVpp"]); u_series = float(series_amplitude_mVpp)
    except (KeyError, TypeError, ValueError) as exc: raise ValueError("Для экспорта нужны Aк, Uк и Uс.") from exc
    if not all(map(lambda x: x == x and abs(x) != float("inf"), (a, u_cal, u_series))) or a == 0 or u_cal <= 0 or u_series <= 0:
        raise ValueError("Aк должно быть конечным ненулевым, Uк и Uс — положительными конечными значениями.")
    if calibration.get("branch", "up") not in {"up", "down"}: raise ValueError("Неизвестная ветвь MCS.")
    return {"amplitude": u_series, "reference_amplitude": u_cal, "reference_angle": a, "branch": calibration.get("branch", "up")}
