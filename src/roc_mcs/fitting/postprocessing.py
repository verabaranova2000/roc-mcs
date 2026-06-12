import pandas as pd
from roc_mcs.fitting.analysis import MODEL_ANALYSIS


def augment_results(results, theta, time=None):
    """
    Использование
    -----
    results = fit_roc_map(roc_map, model_name="split_voigt")

    df_fit = augment_results(
        results,
        theta=roc_map["theta_axis"],
        time=roc_map["time_s"],
    )
    """
    rows = []
    for i, r in enumerate(results):
        params = {k: v.value for k, v in r.parameters.items()}
        row = {
            "index": i,
            "model": r.model,
            "success": r.success,
            "message": r.message,
            **params,
            **r.metrics,
        }
        if time is not None:
            row["time"] = time[i]
        spec = MODEL_ANALYSIS.get(r.model, {})
        for name, fn in spec.get("derived", {}).items():
            row[name] = fn(params)
        if "fwhm" in spec:
            row["FWHM"] = spec["fwhm"](r, theta)
        rows.append(row)
    return pd.DataFrame(rows)