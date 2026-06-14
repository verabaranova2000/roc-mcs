import numpy as np
import matplotlib.pyplot as plt

from roc_mcs.processing.alignment import find_phase, extract_branch
from roc_mcs.processing.calibration import calibrate_roc_map
from roc_mcs.io.mcs import load_mcs, find_mcs_files 
from roc_mcs.plots import plot_roc_map, plot_model_evolution, save_figure
from roc_mcs.export import export_roc_map_excel
from roc_mcs.utils import resolve_output_folder, ensure_folder
from roc_mcs.pipeline.fit import fit_roc_map, run_fit_analysis
from roc_mcs.fitting.postprocessing import augment_results


def process_mcs(mcs, phi, branch="up"):
    """ Функция обработки одного файла """
    counts = mcs["counts"]
    n_channels = mcs["n_channels"]
    s_branch, I_branch = extract_branch(counts, phi, n_channels, branch=branch)
    return mcs["header"]["file_info"]["datetime"], s_branch, I_branch


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
    if not files:
        raise ValueError(f"В каталоге не обнаружены файлы .mcs: {folder}")

    # --- фаза только по первому файлу ---
    first_mcs = load_mcs(files[0])
    counts = first_mcs["counts"]
    n_channels = first_mcs["n_channels"]
    phi, scores = find_phase(counts, n_channels)    
    roc_map["phi"] = float(phi)

    for file in files:
        mcs = load_mcs(file)
        t, s_branch, I_branch = process_mcs(mcs, phi=phi, branch=branch)

        if roc_map["s_axis"] is None:
            roc_map["s_axis"] = s_branch.copy()                            # ось Y
        roc_map["file_name"].append(mcs["file_name"])
        roc_map["time"].append(t)
        roc_map["intensity"].append(I_branch)

    t = np.asarray(roc_map["time"])
    t0 = t[0]
    roc_map["time_s"] = np.array([(ti - t0).total_seconds() for ti in t])  # ось X 
    roc_map["intensity"] = np.asarray(roc_map["intensity"])             # ось Z

    qc_context = {
        "best_phi": float(phi),
        "scores": scores,
        "counts": counts,
        "n_channels": n_channels,
        "file_name": first_mcs["file_name"],
    }

    return roc_map, qc_context


def run_experiment(
    input_folder,
    amplitude,
    reference_amplitude,
    reference_angle,
    output_folder=None,
    save_excel_flag=True,
    save_figure_flag=True,
    diagnostics=(),
    fit_models=()
):
    output_folder = resolve_output_folder(output_folder)

    roc_map, qc = build_roc_map(input_folder)
    roc_map = calibrate_roc_map(
        roc_map,
        amplitude=amplitude,
        reference_amplitude=reference_amplitude,
        reference_angle=reference_angle,
    )

    roc_fig = plot_roc_map(roc_map)

    artifacts = []
    qc_folder = ensure_folder(output_folder / "qc") if diagnostics else None
    fit_folder = ensure_folder(output_folder / "fit") if fit_models else None
    
    # fit_tables = {}   
    # for model in fit_models:
    #     results = fit_roc_map(roc_map, model)
    #     fit_tables[model] = augment_results(results, 
    #                                         theta=roc_map["theta_axis"], time=roc_map["time_s"])
    fit_bundle = None
    if fit_models:
        fit_bundle = run_fit_analysis(roc_map, fit_models=fit_models, 
                                      show_progress=True, plot_residuals=True, plot_evolution=True)
    fit_tables = fit_bundle["fit_tables"] if fit_bundle else {}
    residual_fig = fit_bundle["residual_fig"] if fit_bundle else None


    # --- ROC figure ---
    if save_figure_flag:
        artifacts.append(
            save_figure(roc_fig, output_folder, "roc_map.png"))
        plt.close(roc_fig)
    # --- Excel ---
    if save_excel_flag:
        artifacts.append(
            export_roc_map_excel(roc_map, output_folder, "rocking_curve_dynamics.xlsx",
                                 fit_tables=fit_tables,))

    # --- diagnostics ---
    for diagnostic in diagnostics:
        diag_fig = diagnostic(qc)
        artifacts.append(
            save_figure(diag_fig, qc_folder, f"{diagnostic.__name__}.png"))

    # # --- model evolution figures ---
    # if save_figure_flag:    
    #     for model in fit_models:
    #         evol_fig = plot_model_evolution(fit_tables[model], y_dual=True)
    #         artifacts.append(
    #             save_figure(evol_fig, fit_folder, f"model_evolution_{model}.png"))           
    #         plt.close(evol_fig)

    # --- model evolution figures ---
    if save_figure_flag and fit_models:
        for model_name, evol_fig in fit_bundle["figures"].items():
            artifacts.append(
                save_figure(evol_fig, fit_folder, f"model_evolution_{model_name}.png"))
            plt.close(evol_fig)

        if fit_bundle["residual_fig"] is not None:
            artifacts.append(
                save_figure(fit_bundle["residual_fig"], fit_folder, "residual_maps.png"))
            plt.close(fit_bundle["residual_fig"])            
    return roc_map, roc_fig, fit_tables, artifacts

