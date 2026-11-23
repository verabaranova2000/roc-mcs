from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from roc_mcs.analysis.artifacts import ArtifactStore
from roc_mcs.analysis.engine import WorkflowExecutor
from roc_mcs.analysis.operations import default_registry
from roc_mcs.analysis.workflow_history import WorkflowHistory


def _install_fake_scan_repository(monkeypatch):
    """Provide the repository API omitted from the partial source archive."""
    roi_package = types.ModuleType("roc_mcs.roi")
    roi_package.__path__ = []
    module = types.ModuleType("roc_mcs.roi.h5_scan_repository")

    class H5ScanRepository:
        def __init__(self, path):
            self.h5_path = Path(path).resolve()
            with h5py.File(self.h5_path, "r") as handle:
                self.scans = {
                    key: SimpleNamespace(raw_path=f"/{key}/data", th_path=f"/{key}/theta")
                    for key in handle
                    if isinstance(handle[key], h5py.Group) and "data" in handle[key] and "theta" in handle[key]
                }

        @staticmethod
        def _theta_1d(value):
            return np.asarray(value, dtype=float).reshape(-1)

        def load_source(self, scan_key):
            info = self.scans[scan_key]
            with h5py.File(self.h5_path, "r") as handle:
                data = np.asarray(handle[info.raw_path])
                theta = self._theta_1d(np.asarray(handle[info.th_path]))
            return SimpleNamespace(yxt=data, th=theta, axis=2)

        def centers_xy(self, scan_key):
            shape = self.load_source(scan_key).yxt.shape
            return np.arange(shape[1], dtype=float), np.arange(shape[0], dtype=float)

    module.H5ScanRepository = H5ScanRepository
    monkeypatch.setitem(sys.modules, "roc_mcs.roi", roi_package)
    monkeypatch.setitem(sys.modules, "roc_mcs.roi.h5_scan_repository", module)


def _make_h5(path: Path, data: np.ndarray):
    theta = np.linspace(0.0, 3.0, data.shape[-1], dtype=float)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry1")
        entry.create_dataset("data", data=data)
        entry.create_dataset("theta", data=theta)
    return theta


def _source_node(path: Path, data: np.ndarray, theta: np.ndarray):
    theta_hash = hashlib.sha256(np.asarray(theta, dtype=float).tobytes()).hexdigest()[:16]
    return SimpleNamespace(
        id="source-node-1",
        label="Скан · entry1",
        metadata={
            "source_file": str(path),
            "scan_key": "entry1",
            "raw_dataset": "/entry1/data",
            "theta_dataset": "/entry1/theta",
            "source_shape": list(data.shape),
            "source_shape_yx": list(data.shape[:2]),
            "dtype": str(data.dtype),
            "theta_count": int(theta.size),
            "theta_hash": theta_hash,
            "theta_min": float(theta.min()),
            "theta_max": float(theta.max()),
            "theta_axis": 2,
            "grid_id": "stable-grid",
        },
    )


def _append_second_scan(path: Path):
    with h5py.File(path, "a") as handle:
        entry = handle.create_group("entry2")
        entry.create_dataset("data", data=np.full((2, 3, 4), 17, dtype=np.float32))
        entry.create_dataset("theta", data=np.linspace(0, 3, 4, dtype=float))


def test_legacy_source_session_survives_append_without_copying_scan_data(tmp_path, monkeypatch):
    _install_fake_scan_repository(monkeypatch)
    path = tmp_path / "TPC_data.h5"
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    theta = _make_h5(path, data)
    node = _source_node(path, data, theta)
    _append_second_scan(path)

    store = ArtifactStore(tmp_path / "project" / "artifacts")
    executor = WorkflowExecutor(WorkflowHistory(), store, default_registry())
    bundle = executor._load_external_scan(node)

    np.testing.assert_array_equal(bundle.arrays["data_yxt"], data)
    np.testing.assert_array_equal(bundle.arrays["theta"], theta)
    assert bundle.metadata["source_reference"] is True
    assert bundle.metadata["source_snapshot"] is False
    assert list(store.directory.glob("*.h5")) == []
    assert not (store.directory / "source_snapshots").exists()


def test_selected_scan_structural_change_is_reported_not_silently_accepted(tmp_path, monkeypatch):
    _install_fake_scan_repository(monkeypatch)
    path = tmp_path / "TPC_data.h5"
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    theta = _make_h5(path, data)
    node = _source_node(path, data, theta)
    with h5py.File(path, "r+") as handle:
        del handle["/entry1/data"]
        handle["/entry1"].create_dataset("data", data=np.zeros((3, 3, 4), dtype=np.float32))

    executor = WorkflowExecutor(
        WorkflowHistory(), ArtifactStore(tmp_path / "project" / "artifacts"), default_registry()
    )
    with pytest.raises(RuntimeError, match="Структура выбранного скана"):
        executor._load_external_scan(node)
