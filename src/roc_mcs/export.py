import pandas as pd
from pathlib import Path

def export_phase_scan_excel(phase_scan, filename="rocking_curve_dynamics.xlsx", folder=None):
    """
    Экспорт ROC-карт и метаданных эксперимента в Excel.

    Parameters
    ----------
    phase_scan : dict
        Результат build_phase_scan().
    filename : str
        Имя выходного Excel-файла.
    folder : str | Path | None
        Каталог сохранения.
        Если None — используется текущая директория.
    """

    if folder is None:
        folder = Path.cwd()
    else:
        folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    output_file = folder / filename

    # ---------- ось Y ----------
    if "theta_axis" in phase_scan:
        axis = phase_scan["theta_axis"]
        axis_name = "theta_arcsec"
    else:
        axis = phase_scan["s_axis"]
        axis_name = "sin_axis"
    # ---------- ось X ----------
    time_s = phase_scan["time_s"]

    # ---------- ROC ----------
    data = {axis_name: axis}
    for t, intensity in zip(time_s, phase_scan["intensity"]):
        data[f"t_{t:.1f}s"] = intensity
    roc_df = pd.DataFrame(data)
    # ---------- Metadata ----------
    meta_df = pd.DataFrame({
        "file_name": phase_scan["file_name"],
        "datetime": phase_scan["time"],        
        "time_s": time_s,
        "phi": phase_scan["phi"],
    })

    with pd.ExcelWriter(output_file) as writer:
        roc_df.to_excel(writer, sheet_name="ROC", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)  
        
    print(f"Excel-файл сохранён: {output_file.resolve()}")
    return output_file  