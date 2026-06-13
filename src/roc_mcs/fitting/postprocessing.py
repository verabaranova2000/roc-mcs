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


def prepare_fit_table_for_export(df):
    df = df.copy()
    drop_columns = ["index", "model", "success", "message"]
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    preferred_order = ["time", "theta0", "FWHM"]
    ordered_cols = [c for c in preferred_order if c in df.columns]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols]
    return df