import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from roc_mcs.processing.alignment import find_phase, extract_branch
from roc_mcs.processing.calibration import calibrate_roc_map
from roc_mcs.io.mcs import load_mcs, find_mcs_files 
from roc_mcs.plots import plot_roc_map, plot_model_evolution, plot_residual_maps, save_figure
from roc_mcs.plots import plot_trajectories
from roc_mcs.export import export_roc_map_excel
from roc_mcs.utils import resolve_output_folder, ensure_folder
from roc_mcs.pipeline.fit import fit_roc_map, run_fit_analysis
from roc_mcs.pipeline.trajectory import run_trajectory_analysis
from roc_mcs.fitting.postprocessing import augment_results
from roc_mcs.fitting.metrics import compute_map_metrics
from roc_mcs.fitting.results import ExperimentArtifact
from roc_mcs.diagnostics.registry import DIAGNOSTICS
from roc_mcs.fitting.registry import MODEL_SPECS

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
    save_artifact: bool = False,
    diagnostics=(),
    fit_models=(),
    trajectory_models=(),
):

    fit_models = tuple(fit_models)
    trajectory_models = tuple(trajectory_models)

    missing = set(trajectory_models) - set(fit_models)
    if missing:
        raise ValueError(
            f"trajectory_models must be subset of fit_models. Missing fits for: {sorted(missing)}"
        )

    output_folder = resolve_output_folder(output_folder)

    roc_map, qc = build_roc_map(input_folder)
    roc_map = calibrate_roc_map(
        roc_map,
        amplitude=amplitude,
        reference_amplitude=reference_amplitude,
        reference_angle=reference_angle,
    )

    roc_fig = plot_roc_map(roc_map)

    output_files = []
    qc_folder = ensure_folder(output_folder / "qc") if diagnostics else None
    fit_folder = ensure_folder(output_folder / "fit") if fit_models else None
    
    fit_results = {}
    fit_tables = {}
    trajectory_results = {}
    diff_maps = {}
    metrics = {}

    for model_name in fit_models:
        results = fit_roc_map(roc_map, model_name)
        df_fit = augment_results(results, theta=roc_map["theta_axis"], time=roc_map["time_s"])
        fit_results[model_name] = results
        fit_tables[model_name] = df_fit

        I_model = np.array([res.y_fit for res in results])
        diff_maps[model_name] = roc_map["intensity"] - I_model
        metrics[model_name] = compute_map_metrics(roc_map["intensity"], I_model)
        
        # fit построил таблицу; если модель есть в trajectory_models, запускаем ridge+Kalman
        if model_name in trajectory_models:
            traj = run_trajectory_analysis(roc_map=roc_map, df_fit=df_fit, results=results, model_name=model_name)
            trajectory_results[model_name] = traj  

    # --- ROC figure ---
    if save_figure_flag:
        output_files.append(
            save_figure(roc_fig, output_folder, "roc_map.png"))
        plt.close(roc_fig)
    # --- Excel ---
    if save_excel_flag:
        trajectory_tables = {model: traj.df_kf for model, traj in trajectory_results.items()}
        output_files.append(
            export_roc_map_excel(roc_map, output_folder, "rocking_curve_dynamics.xlsx",
                                 fit_tables=fit_tables, trajectory_tables=trajectory_tables))

    # --- diagnostics ---
    if diagnostics and isinstance(diagnostics[0], str):
        diagnostics = [DIAGNOSTICS[name] for name in diagnostics]    
    for diagnostic in diagnostics:
        diag_fig = diagnostic(qc)
        output_files.append(
            save_figure(diag_fig, qc_folder, f"{diagnostic.__name__}.png"))

    # --- model evolution figures ---
    if save_figure_flag:    
        for model in fit_models:
            evol_fig = plot_model_evolution(fit_tables[model], y_dual=True)
            output_files.append(
                save_figure(evol_fig, fit_folder, f"model_evolution_{model}.png"))
            plt.close(evol_fig)
        for model in trajectory_models:
            traj_fig = plot_trajectories(trajectory_results[model], fit_tables[model], fit_results[model],)
            output_files.append(
                save_figure(traj_fig, fit_folder, f"trajectory_{model}.png")
            )
            plt.close(traj_fig)

        theta = roc_map["theta_axis"]
        time = roc_map["time_s"]        
        residual_fig = plot_residual_maps(time, theta, diff_maps,  metrics=metrics)
        output_files.append(
            save_figure(residual_fig , fit_folder, f"residual_maps.png"))
    
    # --- model metadata used for downstream analysis ---
    model_config = {
        model_name: {
            "fit_parameters": list(MODEL_SPECS[model_name].param_names),
            "derived_parameters": ["FWHM"],
            "fit_sheet": f"Fit_{model_name}",
            "used_for_trajectory": model_name in trajectory_models,
        } for model_name in fit_models}
    
    artifact = ExperimentArtifact(
        roc_map=roc_map,
        fit_tables=fit_tables,
        fit_results=fit_results,
        trajectory_results=trajectory_results,
        metrics=metrics,
        metadata={
            "input_folder": str(input_folder),
            "amplitude": amplitude,
            "reference_amplitude": reference_amplitude,
            "reference_angle": reference_angle,
        },
        model_config=model_config,
        output_files=output_files
    )

    # --- save full experiment object (.pkl) ---
    if save_artifact:
        path = Path(output_folder / "experiment_artifact.pkl")
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(artifact, f)

    return artifact            
