from tqdm import tqdm

from roc_mcs.fitting.optimize import FitConfig, fit_curve


def fit_roc_map(roc_map, model_name="gauss", show_progress=True):
    theta = roc_map["theta_axis"]
    time = roc_map["time_s"]
    intensity_map = roc_map["intensity"]

    results = []
    iterator = tqdm(time) if show_progress else time
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