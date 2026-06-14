from tqdm import tqdm
import numpy as np

from roc_mcs.fitting.optimize import FitConfig, fit_curve
from roc_mcs.fitting.registry import validate_model
from roc_mcs.fitting.postprocessing import augment_results
from roc_mcs.plots import plot_model_evolution, plot_residual_maps
from roc_mcs.fitting.metrics import compute_map_metrics

def fit_roc_map(roc_map, model_name="gauss", show_progress=True):
    validate_model(model_name)
    theta = roc_map["theta_axis"]
    time = roc_map["time_s"]
    intensity_map = roc_map["intensity"]

    results = []
    iterator = tqdm(time, desc=f"Model: {model_name:<14}") if show_progress else time
    for i, t in enumerate(iterator):
        I = intensity_map[i, :]
        res = fit_curve(theta, I, FitConfig(model_name=model_name))
        results.append(res)

    return results

    # FitResult(
    #     model=config.model_name,
    #     parameters=params,
    #     metrics=metrics,
    #     y_fit_global=y_fit_global,
    #     y_fit_local=y_fit_local,
    #     covariance=pcov,
    #     success=True,
    # ) 


def run_fit_analysis(
    roc_map,
    fit_models=(),
    show_progress=True,
    plot_residuals=True,
    plot_evolution=True,
):
    fit_tables = {}
    diff_maps = {}
    metrics = {}
    figures = {}

    for model_name in fit_models:
        results = fit_roc_map(roc_map, model_name=model_name, show_progress=show_progress)
        df_fit = augment_results(results, theta=roc_map["theta_axis"], time=roc_map["time_s"])
        fit_tables[model_name] = df_fit

        if plot_residuals:
            I_model = np.array([res.y_fit for res in results])
            diff_maps[model_name] = roc_map["intensity"] - I_model
            metrics[model_name] = compute_map_metrics(roc_map["intensity"], I_model)
            
        if plot_evolution:
            figures[model_name] = plot_model_evolution(df_fit, x="time")
    
    theta = roc_map["theta_axis"]
    time = roc_map["time_s"]

    residual_fig = None
    if plot_residuals and diff_maps:
        residual_fig = plot_residual_maps(time, theta, diff_maps,  metrics=metrics)
    
    return {
        "fit_tables": fit_tables,
        "diff_maps": diff_maps,
        "metrics": metrics,
        "figures": figures,
        "residual_fig": residual_fig
    }              