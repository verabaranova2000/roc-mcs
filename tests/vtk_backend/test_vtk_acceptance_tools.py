from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "tools" / "vtk_acceptance"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import vtk_acceptance as acceptance


def _make_h5(path: Path, data_shape=(7, 4, 5), theta_count=7, dtype="float32") -> Path:
    with h5py.File(path, "w") as h5:
        group = h5.create_group("entry472/measurement")
        group.create_dataset("Frame", shape=data_shape, dtype=dtype)
        group.create_dataset("th", data=np.arange(theta_count, dtype=np.float64))
    return path


def test_metadata_preflight_detects_theta_first_layout_without_loading_frames(tmp_path):
    path = _make_h5(tmp_path / "raw.h5")
    meta = acceptance._scan_meta(path, "entry472")
    assert meta["dataset"] == "entry472/measurement/Frame"
    assert meta["theta_axis"] == 0
    assert meta["shape"] == (7, 4, 5)
    assert meta["source_bytes"] == 7 * 4 * 5 * 4


def test_metadata_preflight_detects_theta_last_layout(tmp_path):
    path = _make_h5(tmp_path / "theta_last.h5", data_shape=(4, 5, 7))
    meta = acceptance._scan_meta(path, "entry472")
    assert meta["theta_axis"] == 2
    assert meta["display_dtype_bytes"] == 4


def test_metadata_preflight_rejects_ambiguous_theta_axis(tmp_path):
    path = _make_h5(tmp_path / "ambiguous.h5", data_shape=(7, 5, 7))
    with pytest.raises(ValueError, match="неоднозначна"):
        acceptance._scan_meta(path, "entry472")


def test_metadata_preflight_rejects_unknown_scan(tmp_path):
    path = _make_h5(tmp_path / "unknown.h5")
    with pytest.raises(ValueError, match="нет группы"):
        acceptance._scan_meta(path, "entry999")


def test_viewer_preflight_formula_adds_spike_memory_reserve():
    meta = {"source_bytes": 1000, "voxels": 100}
    assert acceptance._viewer_estimate_bytes(meta, False) == 4200
    assert acceptance._viewer_estimate_bytes(meta, True) == 10600


def test_suggested_budgets_keep_system_reserve_and_cap_process_share():
    reserve, cap = acceptance._suggest_budgets(64 * acceptance.GIB, 40 * acceptance.GIB)
    assert reserve == pytest.approx(22.4)
    assert cap == pytest.approx(16.0)


def test_suggested_budgets_refuse_process_budget_when_system_is_busy():
    reserve, cap = acceptance._suggest_budgets(16 * acceptance.GIB, 5 * acceptance.GIB)
    assert reserve == 6.0
    assert cap == 0.0
