import pandas as pd
from roc_mcs.utils import resolve_output_folder
from roc_mcs.fitting.postprocessing import prepare_fit_table_for_export

def export_roc_map_excel(roc_map, output_folder=None, filename="rocking_curve_dynamics.xlsx", fit_tables=None):
    """
    Экспорт ROC-карт и метаданных эксперимента в Excel.

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
        "time_s": time_s,
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

    with pd.ExcelWriter(output_file) as writer:
        roc_df.to_excel(writer, sheet_name="ROC", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)  
        calibration_df.to_excel(writer, sheet_name="Calibration", index=False)
        
        for model_name, df_fit in fit_tables.items():
            sheet_name = f"Fit_{model_name}"[:31]   # ограничение Excel
            export_df = prepare_fit_table_for_export(df_fit)
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output_file  