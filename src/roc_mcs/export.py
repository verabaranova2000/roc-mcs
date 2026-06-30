import pandas as pd
from roc_mcs.utils import resolve_output_folder
from roc_mcs.fitting.postprocessing import prepare_fit_table_for_export

def export_roc_map_excel(
    roc_map,
    output_folder=None,
    filename="rocking_curve_dynamics.xlsx",
    fit_tables=None,
    trajectory_tables=None,
    control_log=None,
):
    """
    Экспорт ROC-карт и метаданных эксперимента в Excel.

    Структура:
        [локальные параметры]
        [локальные метрики]
        [Kalman параметры]

    Parameters
    ----------
    roc_map : dict
        Результат build_roc_map().
    filename : str
        Имя выходного Excel-файла.
    output_folder : str | Path | None
        Каталог сохранения.
        Если None — используется текущая директория.
    """
    output_folder = resolve_output_folder(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / filename

    if fit_tables is None:
        fit_tables = {}

    if trajectory_tables is None:
        trajectory_tables = {}        

    # ---------- ось Y ----------
    if "theta_axis" in roc_map:
        axis = roc_map["theta_axis"]
        axis_name = "theta_arcsec"
    else:
        axis = roc_map["s_axis"]
        axis_name = "sin_axis"
    # ---------- ось X ----------
    time_s = roc_map["time_s"]

    # ---------- ROC ----------
    data = {axis_name: axis}
    for t, intensity in zip(time_s, roc_map["intensity"]):
        data[f"t_{t:.1f}s"] = intensity
    roc_df = pd.DataFrame(data)
    # ---------- Metadata ----------
    meta_df = pd.DataFrame({
        "file_name": roc_map["file_name"],
        "datetime": roc_map["time"],        
        "time_s": time_s,                             # точное время стартов сканов; рабочая ось для анализа
        "time_elapsed_s": roc_map["time_elapsed_s"]   # исходный elapsed из контроллера
        #"phi": roc_map["phi"],
    })

    # ---------- Calibration ----------
    calibration_df = pd.DataFrame({
        "phi": [roc_map["phi"]],
        "reference_amplitude": [roc_map.get("calibration", {}).get("reference_amplitude")],
        "reference_angle": [roc_map.get("calibration", {}).get("reference_angle")],
        "amplitude": [roc_map.get("calibration", {}).get("amplitude")],
        "scale": [roc_map["calibration"]["scale"]],
    })

    # ---------- Force and Pressure ----------
    control_df = None
    if control_log is not None:
        t0 = roc_map["time_elapsed_s"][0]
        mask = control_log["time_s"] >= t0
        control_df = pd.DataFrame({
            "time_s": control_log["time_s"][mask] - t0,
            "force": control_log["force"][mask],
            "pressure_mpa": control_log["pressure_mpa"][mask],
            "state": control_log["state"][mask],
        })

    with pd.ExcelWriter(output_file) as writer:
        roc_df.to_excel(writer, sheet_name="ROC", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)  
        calibration_df.to_excel(writer, sheet_name="Calibration", index=False)
        if control_df is not None:
            control_df.to_excel(writer, sheet_name="ForcePressure", index=False)        
        for model_name, df_fit in fit_tables.items():
            sheet_name = f"Fit_{model_name}"[:31]   # ограничение Excel
            export_df = prepare_fit_table_for_export(df_fit)
            if model_name in trajectory_tables:
                df_kf = trajectory_tables[model_name]
                kf_cols = [c for c in df_kf.columns if c.endswith("_kf")]
                export_df = export_df.join(df_kf[kf_cols])               
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)        
    return output_file  