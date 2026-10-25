from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd

from roc_mcs.fitting.registry import MODEL_SPECS


# -----------------------------------------------------------------------------
# Demo parameter table
# -----------------------------------------------------------------------------

def make_demo_param_table() -> pd.DataFrame:
    """Return the parameter table from the prompt as a DataFrame.

    The values are intentionally close to your example and can be extended
    or replaced by any other table with the same columns.
    """
    rows = [
        {"time": 0,  "theta0": -2.434876943, "FWHM": 60.15541098, "S": 70432.27716, "eta": 0.894472062},
        {"time": 3,  "theta0": -2.382275051, "FWHM": 64.05198424, "S": 69945.12564, "eta": 0.869953416},
        {"time": 6,  "theta0": -1.305736129, "FWHM": 61.69819814, "S": 70345.82661, "eta": 0.913997630},
        {"time": 9,  "theta0": -0.226850242, "FWHM": 66.28956013, "S": 71477.01643, "eta": 0.783725372},
        {"time": 12, "theta0":  0.135339551, "FWHM": 63.85547174, "S": 70592.70908, "eta": 0.850828433},
        {"time": 15, "theta0":  1.175625020, "FWHM": 63.15038151, "S": 69593.88871, "eta": 0.868620536},
        {"time": 18, "theta0":  0.536113962, "FWHM": 64.04912711, "S": 70740.91597, "eta": 0.853480544},
        {"time": 21, "theta0":  1.432417177, "FWHM": 67.53421252, "S": 69946.70666, "eta": 0.837867973},
        {"time": 24, "theta0":  1.296779339, "FWHM": 66.01921090, "S": 70156.37599, "eta": 0.819903532},
        {"time": 27, "theta0":  1.649357522, "FWHM": 66.16053125, "S": 69998.97294, "eta": 0.818906782},
        {"time": 30, "theta0":  2.044035557, "FWHM": 60.18027025, "S": 71214.41999, "eta": 0.943307128},
        {"time": 33, "theta0":  1.784346070, "FWHM": 65.28872010, "S": 70569.90771, "eta": 0.816761271},
        {"time": 36, "theta0":  2.495730699, "FWHM": 65.16469625, "S": 69906.74212, "eta": 0.792706237},
        {"time": 40, "theta0":  1.665935590, "FWHM": 64.16197041, "S": 70410.97696, "eta": 0.894039695},
        {"time": 43, "theta0":  2.580854550, "FWHM": 64.77767839, "S": 69893.70460, "eta": 0.822413824},
        {"time": 46, "theta0":  2.684141284, "FWHM": 65.08965102, "S": 69351.97963, "eta": 0.858920905},
        {"time": 49, "theta0":  3.068107915, "FWHM": 63.29406230, "S": 71060.76516, "eta": 0.906998987},
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Utilities for model parameter packing and synthetic records
# -----------------------------------------------------------------------------

def _pack_model_params(row: pd.Series, spec) -> tuple[float, ...]:
    """Pack a row into the parameter tuple expected by the model registry.

    Special rule:
    - if the model expects `H` and the row contains `FWHM`, use `FWHM` as `H`.
    """
    values: list[float] = []
    for name in spec.param_names:
        if name in row.index:
            values.append(float(row[name]))
        elif name == "H" and "FWHM" in row.index:
            values.append(float(row["FWHM"]))
        else:
            raise KeyError(
                f"Cannot pack parameter '{name}'. Available columns: {list(row.index)}"
            )
    return tuple(values)


@dataclass(slots=True)
class SyntheticMcsRecord:
    """In-memory MCS-like record."""

    file_name: str
    header: dict[str, Any]
    counts: np.ndarray
    n_channels: int
    theta_axis: np.ndarray
    model_name: str
    params: dict[str, float]


def make_synthetic_mcs_records(
    param_table: pd.DataFrame | Iterable[dict[str, Any]],
    model_name: str = "pvoigt",
    theta_axis: np.ndarray | None = None,
    noise: str | float | None = "poisson",
    seed: int = 0,
    dt_s: float = 3.0,
    start_datetime: datetime | None = None,
) -> list[SyntheticMcsRecord]:
    """Create synthetic MCS-like records fully in memory."""
    spec = MODEL_SPECS[model_name]
    df = pd.DataFrame(param_table).copy()

    if theta_axis is None:
        theta_axis = np.linspace(-120.0, 120.0, 1000)
    theta_axis = np.asarray(theta_axis, dtype=float)

    rng = np.random.default_rng(seed)
    if start_datetime is None:
        start_datetime = datetime(2026, 1, 1, 0, 0, 0)

    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=float) * dt_s

    records: list[SyntheticMcsRecord] = []
    for i, row in df.iterrows():
        p = _pack_model_params(row, spec)
        y = np.asarray(spec.func(theta_axis, *p), dtype=float)
        y = np.clip(y, 0.0, None)

        if noise == "poisson":
            y = rng.poisson(y).astype(float)
        elif isinstance(noise, (int, float)) and float(noise) > 0:
            y = np.clip(y + rng.normal(0.0, float(noise), size=y.shape), 0.0, None)

        if pd.notna(row["time"]):
            if isinstance(row["time"], (int, float, np.integer, np.floating)):
                ts = start_datetime + timedelta(seconds=float(row["time"]))
            else:
                ts = pd.to_datetime(row["time"]).to_pydatetime()
        else:
            ts = start_datetime + timedelta(seconds=i * dt_s)

        records.append(
            SyntheticMcsRecord(
                file_name=f"synthetic_{i:03d}.mcs",
                header={
                    "file_info": {"datetime": ts},
                    "synthetic": True,
                    "index": int(i),
                },
                counts=y,
                n_channels=int(y.size),
                theta_axis=theta_axis.copy(),
                model_name=model_name,
                params={name: float(value) for name, value in zip(spec.param_names, p)},
            )
        )

    return records


def build_synthetic_roc_map(
    records: list[SyntheticMcsRecord],
) -> dict[str, Any]:
    """Build a ROC-map structure that mirrors the output of build_roc_map()."""
    if not records:
        raise ValueError("records is empty")

    times = [r.header["file_info"]["datetime"] for r in records]
    t0 = times[0]
    time_s = np.array([(t - t0).total_seconds() for t in times], dtype=float)

    roc_map = {
        "file_name": [r.file_name for r in records],
        "time": times,
        "time_s": time_s,
        "phi": float("nan"),
        "s_axis": records[0].theta_axis.copy(),
        "intensity": np.asarray([r.counts for r in records], dtype=float),
        "synthetic": True,
        "model_name": records[0].model_name,
        "records": records,
    }
    return roc_map