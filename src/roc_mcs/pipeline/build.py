import numpy as np
from roc_mcs.processing.alignment import find_phase, extract_branch
from roc_mcs.processing.calibration import calibrate_roc_map
from roc_mcs.io.mcs import load_mcs, find_mcs_files 
from roc_mcs.plots import plot_roc_map

def process_mcs(mcs, branch="up"):
    """ Функция обработки одного файла """
    counts = mcs["counts"]
    n_channels = mcs["n_channels"]
    phi, _ = find_phase(counts, n_channels)
    s_branch, I_branch = extract_branch(counts, phi, n_channels, branch=branch)
    return mcs["header"]["file_info"]["datetime"], phi, s_branch, I_branch


def build_roc_map(folder, branch="up"):
    """ Функция сборки всего эксперимента """
    roc_map = {
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

        if roc_map["s_axis"] is None:
            roc_map["s_axis"] = s_branch.copy()                            # ось Y
        roc_map["file_name"].append(mcs["file_name"])
        roc_map["time"].append(t)
        roc_map["phi"].append(phi)
        roc_map["intensity"].append(I_branch)

    t = np.asarray(roc_map["time"])
    t0 = t[0]
    roc_map["time_s"] = np.array([(ti - t0).total_seconds() for ti in t])  # ось X 
    roc_map["intensity"] = np.asarray(roc_map["intensity"])             # ось Z
    roc_map["phi"] = np.asarray(roc_map["phi"])
    
    return roc_map



# def run_pipeline(folder, angle_scale):
#     scan = build_roc_map(folder)
#     scan = calibrate_roc_map(scan, angle_scale)
#     fig = plot_roc_map(scan)
#     return scan, fig

def run_pipeline(folder, amplitude, reference_angle, reference_amplitude=400.0):
    roc_map = build_roc_map(folder)
    roc_map = calibrate_roc_map(
        roc_map,
        amplitude=amplitude,        
        reference_angle=reference_angle,
        reference_amplitude=reference_amplitude,
    )
    fig = plot_roc_map(roc_map)
    return roc_map, fig