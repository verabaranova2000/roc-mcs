import numpy as np
from roc_mcs.processing.alignment import find_phase, extract_branch
from roc_mcs.processing.calibration import calibrate_phase_scan
from roc_mcs.io import load_mcs, find_mcs_files 
from roc_mcs.plots import plot_phase_scan

def process_mcs(mcs, branch="up"):
    """ Функция обработки одного файла """
    counts = mcs["counts"]
    n_channels = mcs["n_channels"]
    phi, _ = find_phase(counts, n_channels)
    s_branch, I_branch = extract_branch(counts, phi, n_channels, branch=branch)
    return mcs["header"]["file_info"]["datetime"], phi, s_branch, I_branch


def build_phase_scan(folder, branch="up"):
    """ Функция сборки всего эксперимента """
    phase_scan = {
        "file_name": [],
        "time": [],
        "time_s": [],
        "phi": [],
        "s_axis": None,
        "intensity": [],
    }

    files = find_mcs_files(folder)
    for file in files:
        mcs = load_mcs(file)
        t, phi, s_branch, I_branch = process_mcs(mcs, branch=branch)

        if phase_scan["s_axis"] is None:
            phase_scan["s_axis"] = s_branch.copy()                            # ось Y
        phase_scan["file_name"].append(mcs["file_name"])
        phase_scan["time"].append(t)
        phase_scan["phi"].append(phi)
        phase_scan["intensity"].append(I_branch)

    t = np.asarray(phase_scan["time"])
    t0 = t[0]
    phase_scan["time_s"] = np.array([(ti - t0).total_seconds() for ti in t])  # ось X 
    phase_scan["intensity"] = np.asarray(phase_scan["intensity"])             # ось Z
    phase_scan["phi"] = np.asarray(phase_scan["phi"])
    
    return phase_scan



def run_pipeline(folder, angle_scale):
    scan = build_phase_scan(folder)
    scan = calibrate_phase_scan(scan, angle_scale)
    fig = plot_phase_scan(scan)
    return scan, fig