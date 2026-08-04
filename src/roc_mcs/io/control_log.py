from pathlib import Path
import pandas as pd
import re



def find_control_log_file(folder,
                      pattern="*_control_log_*.csv"):
    """
    Поиск лог-файла управления экспериментом.
    """
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Не найден лог-файл: {pattern}")
    if len(files) > 1:
        raise ValueError(f"Обнаружено несколько лог-файлов: {files}")
    return files[0]


def load_control_log(file):
    """
    Чтение полного control_log-файла.
    """
    return pd.read_csv(file, sep=";")  


def extract_scan_points(df_log):
    """
    Извлечение моментов запуска MCS-сканов.
    Универсально для PERIODIC_SCAN_START и CYCLE_SCAN_START.
    """
    scans = []
    # mask = df_log["state"].str.contains("CYCLE_SCAN_START", na=False)
    # mask = df_log["state"].str.contains(r"_SCAN_START_", regex=True, na=False)
    mask = df_log["state"].str.contains(r"^[A-Z_]+_SCAN_START_\d+$",
                                        regex=True, na=False,)
    for _, row in df_log[mask].iterrows():
        m = re.search( r"SCAN_START_(\d+)", row["state"])
        if m is None:
            continue
        scans.append({
            "scan_id": int(m.group(1)),
            "elapsed": row["elapsed"],
            "force": row["force"],
            "pressure_mpa": row["pressure_mpa"],
            "time": row["time"],
        })
    return pd.DataFrame(scans).sort_values("scan_id").reset_index(drop=True)



def build_control_log(folder):
    """
    Подготовка данных из *_control_log-файла.

    Возвращает:
        control_log["raw"]
            полный лог

        control_log["scan_points"]
            точки запуска MCS-сканов

        control_log["time_s"]
            плотная временная ось

        control_log["force"]
            сила во времени

        control_log["pressure_mpa"]
            давление во времени
    """

    file = find_control_log_file(folder)
    df_log = load_control_log(file)
    scan_points = extract_scan_points(df_log)
    control_log = {
        "file_name": file.name,
        "raw": df_log,
        "scan_points": scan_points,
        "time_s": df_log["elapsed"].to_numpy(),
        "force": df_log["force"].to_numpy(),
        "pressure_mpa": df_log["pressure_mpa"].to_numpy(),
        "state": df_log["state"].to_numpy(),
    }
    return control_log    


def attach_scan_timing(roc_map, control_log):
    """
    Привязка MCS-файлов к точному времени из periodic_control_log.

    Добавляет в roc_map:

    time_elapsed_s
        Абсолютное elapsed из лог-файла для каждого SCAN_START.
    
    time_s
        Точное время относительно первого скана (вместо времения из MCS-файлов, 
        которое пересохраняется в time_mcs_s)
        Используется как основная временная ось ROC-карты.
    
    time_mcs_s
        Исходная временная ось, восстановленная из заголовков MCS-файлов.

    force
        Сила в момент старта скана.

    pressure_mpa
        Давление в момент старта скана.

    В результате время сохраняется так:
    roc_map["time_mcs_s"]       - исходная ось из MCS metadata
    roc_map["time_elapsed_s"]   - абсолютный elapsed из periodic log
    roc_map["time_s"]           - рабочая ось для анализа    
    """
    scan_points = control_log["scan_points"].copy()

    n_scans = len(scan_points)
    n_files = len(roc_map["file_name"])
    if n_scans != n_files:
        raise ValueError(f"Число SCAN_START ({n_scans}) "
                         f"не совпадает с числом MCS файлов ({n_files})")

    scan_points = scan_points.sort_values("scan_id").reset_index(drop=True)
    
    # --- Время ---
    if "time_s" in roc_map:
        roc_map["time_mcs_s"] = roc_map["time_s"].copy()  # переназвать исходную ось времени (из MCS-файлов)
    roc_map["time_elapsed_s"] = scan_points["elapsed"].to_numpy(float)
    roc_map["time_s"] = roc_map["time_elapsed_s"] - roc_map["time_elapsed_s"][0]  # основная ось для дальнейшего анализа

    # --- Давление и сила ---
    roc_map["force"] = (scan_points["force"].to_numpy(float))
    roc_map["pressure_mpa"] = (scan_points["pressure_mpa"].to_numpy(float))
    
    return roc_map    