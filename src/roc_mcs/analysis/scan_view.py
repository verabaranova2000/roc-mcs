"""Renderer-neutral, read-only view state for source and processed scan artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifacts import ArtifactBundle
from .workflow_history import ProcessingNode


@dataclass(frozen=True, slots=True)
class ScanViewState:
    node_id: str
    operation_id: str
    label: str
    data_yxt: np.ndarray
    theta: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    map_data: np.ndarray
    map_title: str
    valid_mask_yxt: np.ndarray | None
    metadata: dict[str, Any]


def _axis(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size != length or not np.isfinite(result).all():
        raise ValueError(f"Ось {name} должна быть конечным 1D-массивом длины {length}")
    if result.size > 1:
        delta = np.diff(result)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"Ось {name} должна быть строго монотонной")
    return result


def _sum_map(data_yxt: np.ndarray, valid_mask_yxt: np.ndarray | None, chunk_rows: int = 32) -> np.ndarray:
    ny, nx, _ = data_yxt.shape
    result = np.empty((ny, nx), dtype=np.float64)
    for y0 in range(0, ny, max(1, int(chunk_rows))):
        y1 = min(ny, y0 + max(1, int(chunk_rows)))
        block = np.asarray(data_yxt[y0:y1])
        valid = np.isfinite(block)
        if valid_mask_yxt is not None:
            valid &= np.asarray(valid_mask_yxt[y0:y1], dtype=bool)
        sums = np.sum(np.where(valid, block, 0.0), axis=2, dtype=np.float64)
        sums[~np.any(valid, axis=2)] = np.nan
        result[y0:y1] = sums
    return result


def build_scan_view_state(
    node: ProcessingNode,
    scan: ArtifactBundle,
    spikes: ArtifactBundle | None = None,
) -> ScanViewState:
    """Build one consistent view of a workflow node without altering its artifacts."""
    if node.operation_id == "preprocess.spikes":
        if spikes is None:
            raise ValueError("Для spike-узла требуется выходной spikes-артефакт")
        expected_grid = str(scan.metadata.get("grid_id", ""))
        actual_grid = str(spikes.metadata.get("grid_id", ""))
        if not expected_grid or expected_grid != actual_grid:
            raise ValueError("Spike-артефакт относится к другой координатной сетке")
        arrays = spikes.arrays
        data = np.asarray(arrays["clean_data_yxt"])
        mask_value = arrays.get("detector_valid_mask_yxt", scan.arrays.get("valid_mask_yxt"))
        valid = None if mask_value is None else np.asarray(mask_value, dtype=bool)
        map_data = np.asarray(arrays["spike_count_map"])
        map_title = "Число обнаруженных и интерполированных спайков"
        label = f"{node.label} · интерполированные данные"
    else:
        arrays = scan.arrays
        data = np.asarray(arrays["data_yxt"])
        mask_value = arrays.get("valid_mask_yxt")
        valid = None if mask_value is None else np.asarray(mask_value, dtype=bool)
        map_data = None
        map_title = "Суммарная интенсивность по θ"
        label = node.label

    if data.ndim != 3:
        raise ValueError(f"Ожидался data_yxt с формой Y×X×θ, получено {data.shape}")
    ny, nx, ntheta = data.shape
    theta = _axis(arrays.get("theta", scan.arrays.get("theta")), ntheta, "θ")
    x = _axis(arrays.get("x_coords", scan.arrays.get("x_coords")), nx, "X")
    y = _axis(arrays.get("y_coords", scan.arrays.get("y_coords")), ny, "Y")
    if valid is not None and valid.shape != data.shape:
        raise ValueError("Маска валидности должна совпадать с data_yxt")
    if node.operation_id == "preprocess.spikes":
        if map_data.shape != (ny, nx):
            raise ValueError("spike_count_map не согласована с координатной сеткой")
    else:
        map_data = _sum_map(data, valid)

    return ScanViewState(
        node_id=node.id, operation_id=node.operation_id, label=label,
        data_yxt=data, theta=theta, x_coords=x, y_coords=y,
        map_data=map_data, map_title=map_title, valid_mask_yxt=valid,
        metadata={**scan.metadata, **(spikes.metadata if spikes is not None else {})},
    )
