from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

# In the delivery workspace the overlay is merged with the user's existing src/.
# This extra path lets the same tests run against the module archive supplied here.
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
OVERLAY_SRC = ROOT / "src"
if str(OVERLAY_SRC) not in sys.path:
    sys.path.insert(0, str(OVERLAY_SRC))

from roc_mcs.analysis.artifacts import ArtifactBundle, ArtifactStore
from roc_mcs.analysis.engine import (
    WorkflowExecutor, build_input_refs, create_source_node,
)
from roc_mcs.analysis.operations import default_registry
from roc_mcs.analysis.workflow_history import NodeStatus, WorkflowHistory


def _make_synthetic_h5(path: Path) -> np.ndarray:
    rng = np.random.default_rng(20261105)
    theta = np.linspace(10.0, 14.0, 31)
    data = rng.poisson(3.0, size=(theta.size, 9, 11)).astype(np.int32)
    theta_peak = np.exp(-0.5 * ((theta - 12.0) / 0.18) ** 2)
    data[:, 3:6, 4:7] += np.rint(35.0 * theta_peak[:, None, None]).astype(np.int32)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry42")
        measurement = group.create_group("measurement")
        measurement.create_dataset("Frame", data=data, compression="gzip")
        measurement.create_dataset("th", data=theta)
    return data


class WorkflowHistoryTests(unittest.TestCase):
    def test_branching_and_multiple_inputs(self):
        history = WorkflowHistory()
        source = history.add_node(
            "source.scan", "Scan", status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "@external-hdf5-scan"},
        )
        detector_qc = history.add_node(
            "preprocess.detector_bad_pixels", "Detector QC · none",
            parent_ids=[source.id],
            input_refs={"scan": {"node_id": source.id, "output": "scan"}},
            parameters={"pixels": []}, status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "d" * 32},
            review_status="accepted",
        )
        pool_a = history.add_node(
            "preprocess.pool", "Pool A", parent_ids=[detector_qc.id],
            input_refs={"scan": {"node_id": detector_qc.id, "output": "scan"}},
            parameters={"size": 5, "stride": 1},
            status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "a" * 32},
        )
        pool_b = history.add_node(
            "preprocess.pool", "Pool B", parent_ids=[detector_qc.id],
            input_refs={"scan": {"node_id": detector_qc.id, "output": "scan"}},
            parameters={"size": 5, "stride": 5},
        )
        spikes = history.add_node(
            "preprocess.spikes", "Spikes", parent_ids=[pool_a.id],
            input_refs={"scan": {"node_id": pool_a.id, "output": "scan"}},
            status=NodeStatus.SUCCEEDED, output_refs={"spikes": "b" * 32},
        )
        peaks = history.add_node(
            "peaks.detect", "Peaks", parent_ids=[spikes.id],
            input_refs={
                "scan": {"node_id": pool_a.id, "output": "scan"},
                "spikes": {"node_id": spikes.id, "output": "spikes"},
            },
        )
        self.assertEqual({n.id for n in history.children(source.id)}, {detector_qc.id})
        self.assertEqual({n.id for n in history.children(detector_qc.id)}, {pool_a.id, pool_b.id})
        self.assertEqual(set(peaks.parent_ids), {pool_a.id, spikes.id})
        self.assertEqual(history.find_ancestor_output(peaks.id, "scan").id, pool_a.id)
        from roc_mcs.analysis.engine import build_input_refs
        self.assertEqual(
            build_input_refs(history, spikes.id, "peaks.detect", {"use_interpolated_data": True})["spikes"]["node_id"],
            spikes.id,
        )
        with self.assertRaises(ValueError):
            build_input_refs(history, pool_a.id, "peaks.detect", {"use_interpolated_data": True})
        with self.assertRaises(ValueError):
            build_input_refs(history, spikes.id, "preprocess.pool")

        history.transition(peaks.id, NodeStatus.RUNNING)
        history.transition(peaks.id, NodeStatus.SUCCEEDED, output_refs={"peaks": "c" * 32})
        fit_refs = build_input_refs(history, peaks.id, "fit.scan")
        self.assertEqual(fit_refs["scan"]["node_id"], pool_a.id)
        self.assertEqual(fit_refs["peaks"]["node_id"], peaks.id)
        self.assertEqual(fit_refs["spikes"]["node_id"], spikes.id)
        fit_without_mask = build_input_refs(
            history, peaks.id, "fit.scan", {"use_spike_mask": False}
        )
        self.assertNotIn("spikes", fit_without_mask)

    def test_immutable_config_and_state_transitions(self):
        history = WorkflowHistory()
        node = history.add_node("preprocess.pool", "Pool", parameters={"size": 5, "stride": 1})
        history.transition(node.id, NodeStatus.RUNNING)
        done = history.transition(
            node.id, NodeStatus.SUCCEEDED, output_refs={"scan": "a" * 32}, summary="ok"
        )
        self.assertEqual(done.parameters, {"size": 5, "stride": 1})
        self.assertEqual(done.status, NodeStatus.SUCCEEDED)
        with self.assertRaises(ValueError):
            history.transition(node.id, NodeStatus.RUNNING)

    def test_bad_parent_and_cycle_are_rejected(self):
        with self.assertRaises(ValueError):
            WorkflowHistory().add_node("fit.scan", "Bad", parent_ids=["missing"])

        from roc_mcs.analysis.workflow_history import ProcessingNode
        first = ProcessingNode(
            id="a", operation_id="x", label="A", parent_ids=("b",),
            parameters_json="{}", created_at="now",
        )
        second = ProcessingNode(
            id="b", operation_id="x", label="B", parent_ids=("a",),
            parameters_json="{}", created_at="now",
        )
        with self.assertRaises(ValueError):
            WorkflowHistory([first, second])

    def test_manifest_round_trip(self):
        history = WorkflowHistory()
        source = history.add_node("source.scan", "Scan", status=NodeStatus.SUCCEEDED)
        history.add_node(
            "preprocess.pool", "Pool", parent_ids=[source.id],
            input_refs={"scan": {"node_id": source.id, "output": "scan"}},
            parameters={"size": 5, "stride": 2},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            history.save(path)
            loaded = WorkflowHistory.load(path)
        self.assertEqual(len(loaded.nodes), 2)
        self.assertEqual(loaded.nodes[1].parameters["stride"], 2)


class ArtifactStoreTests(unittest.TestCase):
    def test_array_table_and_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            artifact_id = store.write(ArtifactBundle(
                arrays={
                    "data": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                    "mask": np.array([True, False, True]),
                },
                tables={
                    "catalog": pd.DataFrame({
                        "x": [1, 2],
                        "theta": [10.5, 11.5],
                        "status": ["found", "not_found"],
                        "optional": [None, "value"],
                        "optional_string": pd.Series([pd.NA, "value"], dtype="string"),
                        "optional_count": pd.Series([pd.NA, 7], dtype="Int64"),
                        "label": np.asarray(["alpha", "beta"], dtype="U5"),
                    }),
                    "duplicate_columns": pd.DataFrame(
                        np.asarray([[1, 10], [2, 20]], dtype=np.int64),
                        columns=["value", "value"],
                    ),
                },
                metadata={"grid_id": "abc", "iterations": 3, "first_iteration_change": float("inf")},
            ))
            with h5py.File(store.path_for(artifact_id), "r") as handle:
                self.assertEqual(int(handle.attrs["format_version"]), 2)
            loaded = store.read(artifact_id)
            np.testing.assert_array_equal(loaded.arrays["data"], np.arange(24, dtype=np.float32).reshape(2, 3, 4))
            np.testing.assert_array_equal(loaded.arrays["mask"], [True, False, True])
            self.assertEqual(loaded.metadata["grid_id"], "abc")
            self.assertEqual(loaded.tables["catalog"]["status"].tolist(), ["found", "not_found"])
            self.assertEqual(loaded.tables["catalog"]["optional"].tolist(), [None, "value"])
            self.assertEqual(loaded.tables["catalog"]["optional_string"].tolist(), [None, "value"])
            self.assertEqual(loaded.tables["catalog"]["optional_string"].dtype, object)
            self.assertTrue(pd.isna(loaded.tables["catalog"]["optional_count"].iloc[0]))
            self.assertEqual(int(loaded.tables["catalog"]["optional_count"].iloc[1]), 7)
            self.assertEqual(str(loaded.tables["catalog"]["optional_count"].dtype), "Int64")
            self.assertEqual(loaded.tables["catalog"]["label"].tolist(), ["alpha", "beta"])
            self.assertEqual(
                loaded.tables["duplicate_columns"].columns.tolist(), ["value", "value"]
            )
            self.assertEqual(
                loaded.tables["duplicate_columns"].iloc[:, 1].tolist(), [10, 20]
            )
            self.assertIsNone(loaded.metadata["first_iteration_change"])

    def test_legacy_string_artifact_remains_readable(self):
        # Artifacts written by the initial Workbench MVP already used format v1,
        # UTF-8 datasets and explicit null masks. The repaired reader stays
        # backward-compatible with those project folders.
        import json
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            artifact_id = "a" * 32
            path = store.path_for(artifact_id)
            with h5py.File(path, "w") as handle:
                handle.attrs["format_version"] = 1
                handle.attrs["metadata_json"] = json.dumps({"legacy": True})
                handle.create_group("arrays")
                tables = handle.create_group("tables")
                catalog = tables.create_group("catalog")
                catalog.attrs["columns_json"] = json.dumps(["optional", "status"])
                catalog.attrs["row_count"] = 2
                string_dtype = h5py.string_dtype("utf-8")
                optional = catalog.create_dataset(
                    "c0000", data=np.asarray(["", "value"], dtype=object), dtype=string_dtype
                )
                optional.attrs["logical_type"] = "string"
                catalog.create_dataset("null_c0000", data=np.asarray([True, False]))
                status = catalog.create_dataset(
                    "c0001", data=np.asarray(["ok", "failed"], dtype=object), dtype=string_dtype
                )
                status.attrs["logical_type"] = "string"
                catalog.create_dataset("null_c0001", data=np.asarray([False, False]))

            loaded = store.read(artifact_id)
            self.assertEqual(loaded.tables["catalog"]["optional"].tolist(), [None, "value"])
            self.assertEqual(loaded.tables["catalog"]["status"].tolist(), ["ok", "failed"])
            self.assertTrue(loaded.metadata["legacy"])


class SpikeOperationTests(unittest.TestCase):
    def test_real_spike_adapter_and_artifact_round_trip(self):
        from roc_mcs.analysis.operations import default_registry

        rng = np.random.default_rng(20261105)
        theta = np.linspace(10.0, 14.0, 51)
        data = np.empty((5, 7, theta.size), dtype=float)
        for y in range(5):
            for x in range(7):
                data[y, x] = rng.poisson(5, theta.size) + 30.0 * np.exp(
                    -0.5 * ((theta - 12.0) / 0.12) ** 2
                )
        data[2, 3, 24] += 200.0
        source = ArtifactBundle(
            arrays={
                "data_yxt": data,
                "theta": theta,
                "x_coords": np.arange(7, dtype=float),
                "y_coords": np.arange(5, dtype=float),
            },
            metadata={"grid_id": "synthetic-grid", "theta_axis": 0},
        )
        result = default_registry().execute(
            "preprocess.spikes", {"scan": source},
            {"max_iter": 3, "tol": 1e-3, "collect_kappa_fits": True},
        )
        self.assertEqual(result.bundle.arrays["spike_mask_tyx"].shape, (51, 5, 7))
        self.assertEqual(result.bundle.arrays["valid_mask_yxt"].shape, data.shape)
        self.assertEqual(result.bundle.arrays["clean_data_yxt"].shape, data.shape)
        self.assertGreater(result.bundle.metadata["n_spikes"], 0)
        self.assertIn("spike_iterations", result.bundle.tables)
        self.assertIn("kappa_thresholds", result.bundle.tables)
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            artifact_id = store.write(result.bundle)
            restored = store.read(artifact_id)
        np.testing.assert_array_equal(
            restored.arrays["spike_mask_tyx"], result.bundle.arrays["spike_mask_tyx"]
        )
        self.assertEqual(restored.metadata["grid_id"], "synthetic-grid")

    def test_spike_adapter_without_kappa_curve_collection(self):
        theta = np.linspace(10.0, 14.0, 41)
        rng = np.random.default_rng(7)
        data = rng.poisson(5.0, size=(3, 5, theta.size)).astype(float)
        data[:, :, 20] += 100.0
        source = ArtifactBundle(
            arrays={
                "data_yxt": data, "theta": theta,
                "x_coords": np.arange(5, dtype=float),
                "y_coords": np.arange(3, dtype=float),
            },
            metadata={"grid_id": "small-grid", "theta_axis": 0},
        )
        result = default_registry().execute(
            "preprocess.spikes", {"scan": source},
            {"max_iter": 2, "tol": 1e-3, "collect_kappa_fits": False},
        )
        self.assertTrue(result.bundle.tables["kappa_thresholds"].empty)
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            artifact_id = store.write(result.bundle)
            restored = store.read(artifact_id)
        self.assertTrue(restored.tables["kappa_thresholds"].empty)


class PeakOperationTests(unittest.TestCase):
    @staticmethod
    def _source_bundle(theta_axis: int) -> ArtifactBundle:
        rng = np.random.default_rng(20261106)
        theta = np.linspace(10.0, 14.0, 51)
        data_yxt = rng.poisson(3.0, size=(5, 6, theta.size)).astype(np.float64)
        profile = 55.0 * np.exp(-0.5 * ((theta - 12.0) / 0.16) ** 2)
        data_yxt[2:4, 2:5, :] += profile[None, None, :]
        return ArtifactBundle(
            arrays={
                "data_yxt": data_yxt,
                "theta": theta,
                "x_coords": np.arange(data_yxt.shape[1], dtype=float),
                "y_coords": np.arange(data_yxt.shape[0], dtype=float),
            },
            metadata={"grid_id": "synthetic-grid", "theta_axis": theta_axis},
        )

    def test_empty_peak_catalog_keeps_downstream_schema(self):
        theta = np.linspace(10.0, 14.0, 21)
        source = ArtifactBundle(
            arrays={
                "data_yxt": np.ones((3, 4, theta.size), dtype=np.float64),
                "theta": theta,
                "x_coords": np.arange(4, dtype=float),
                "y_coords": np.arange(3, dtype=float),
            },
            metadata={"grid_id": "flat-grid", "theta_axis": 0},
        )
        result = default_registry().execute(
            "peaks.detect", {"scan": source}, {
                "window_size": 3,
                "smooth_window": 5,
                "smooth_polyorder": 2,
                "min_distance": 2,
                "prominence_mode": "sigma",
                "prominence_value": 3.0,
                "min_relative_prominence": 0.07,
                "use_interpolated_data": False,
            },
        ).bundle
        catalog = result.tables["peak_catalog"]
        self.assertTrue(catalog.empty)
        self.assertTrue({"peak_id", "x", "y", "theta_index", "theta", "prominence", "height"}.issubset(catalog.columns))
        self.assertEqual(result.arrays["peak_count_map"].shape, (3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            artifact_id = store.write(result)
            restored = store.read(artifact_id)
        self.assertTrue(restored.tables["peak_catalog"].empty)
        self.assertTrue({"x", "y", "theta"}.issubset(restored.tables["peak_catalog"].columns))

    def test_peak_adapter_is_independent_of_hdf5_theta_axis(self):
        registry = default_registry()
        parameters = registry.get("peaks.detect").validate({
            "window_size": 3,
            "smooth_window": 7,
            "smooth_polyorder": 2,
            "min_distance": 4,
            "prominence_mode": "sigma",
            "prominence_value": 2.0,
            "min_relative_prominence": 0.07,
            "use_interpolated_data": False,
        })
        axis0 = registry.execute(
            "peaks.detect", {"scan": self._source_bundle(0)}, parameters
        ).bundle
        axis2 = registry.execute(
            "peaks.detect", {"scan": self._source_bundle(2)}, parameters
        ).bundle
        self.assertGreater(axis0.metadata["n_peaks"], 0)
        np.testing.assert_array_equal(
            axis0.arrays["peak_count_map"], axis2.arrays["peak_count_map"]
        )
        left = axis0.tables["peak_catalog"].sort_values(["y", "x", "theta_index"]).reset_index(drop=True)
        right = axis2.tables["peak_catalog"].sort_values(["y", "x", "theta_index"]).reset_index(drop=True)
        self.assertEqual(left.shape, right.shape)
        for column in ("x", "y", "theta_index", "theta", "prominence", "height"):
            np.testing.assert_allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )


class FitOperationIntegrationTests(unittest.TestCase):
    def test_fit_adapter_uses_existing_fit_scan_and_round_trips(self):
        import importlib.util
        import os

        if importlib.util.find_spec("jax") is not None:
            backend = "jax"
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        elif importlib.util.find_spec("lmfit") is not None:
            backend = "lmfit"
        else:
            self.skipTest("Для интеграционного fit-теста требуется JAX или LMFit")

        registry = default_registry()
        scan = PeakOperationTests._source_bundle(0)
        peak_parameters = registry.get("peaks.detect").validate({
            "window_size": 3,
            "smooth_window": 7,
            "smooth_polyorder": 2,
            "min_distance": 4,
            "prominence_mode": "sigma",
            "prominence_value": 2.0,
            "min_relative_prominence": 0.07,
            "use_interpolated_data": False,
        })
        peaks = registry.execute("peaks.detect", {"scan": scan}, peak_parameters).bundle
        fit_parameters = registry.get("fit.scan").validate({
            "model": "pvoigt",
            "backend": backend,
            "center_window": 0.5,
            "add_background": True,
            "allow_center_shift": True,
            "batch_size": 8,
            "solver_max_iter": 30,
            "use_spike_mask": False,
        })
        deliberately_invalid_spikes = ArtifactBundle(
            arrays={"valid_mask_yxt": np.zeros(scan.arrays["data_yxt"].shape, dtype=bool)},
            metadata={"grid_id": scan.metadata["grid_id"]},
        )
        fit = registry.execute(
            "fit.scan",
            {"scan": scan, "peaks": peaks, "spikes": deliberately_invalid_spikes},
            fit_parameters,
        ).bundle
        self.assertFalse(fit.metadata["use_spike_mask"])
        self.assertNotIn("valid_mask", fit.arrays)
        self.assertIn("fit_cube", fit.arrays)
        self.assertEqual(fit.arrays["fit_cube"].shape, scan.arrays["data_yxt"].shape)
        self.assertIn("pixel_fit_catalog", fit.tables)
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            artifact_id = store.write(fit)
            restored = store.read(artifact_id)
        np.testing.assert_allclose(restored.arrays["fit_cube"], fit.arrays["fit_cube"])
        self.assertEqual(
            restored.tables["pixel_fit_catalog"].shape,
            fit.tables["pixel_fit_catalog"].shape,
        )



class OperationValidationTests(unittest.TestCase):
    def test_pool_rejects_even_and_fractional_window_sizes(self):
        registry = default_registry()
        with self.assertRaises(ValueError):
            registry.get("preprocess.pool").validate({"size": 4, "stride": 1})
        with self.assertRaises(ValueError):
            registry.get("preprocess.pool").validate({"size": 3.5, "stride": 1})

    def test_peak_parameters_reject_non_finite_prominence(self):
        registry = default_registry()
        with self.assertRaises(ValueError):
            registry.get("peaks.detect").validate({"prominence_value": float("nan")})
        with self.assertRaises(ValueError):
            registry.get("peaks.detect").validate({"smooth_window": 10})

    def test_jax_backend_is_limited_to_implemented_models(self):
        import importlib.util

        registry = default_registry()
        if importlib.util.find_spec("lmfit") is not None:
            normalized = registry.get("fit.scan").validate(
                {"model": "gauss", "backend": "lmfit"}
            )
            self.assertIsNone(normalized["batch_size"])
            self.assertIsNone(normalized["solver_max_iter"])
        else:
            with self.assertRaisesRegex(ValueError, "lmfit"):
                registry.get("fit.scan").validate(
                    {"model": "gauss", "backend": "lmfit"}
                )

        # JAX may be installed while the selected model has no JAX implementation.
        if importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None:
            with self.assertRaisesRegex(ValueError, "JAX-backend не реализован"):
                registry.get("fit.scan").validate(
                    {"model": "gauss", "backend": "jax"}
                )
        else:
            with self.assertRaisesRegex(ValueError, "jax"):
                registry.get("fit.scan").validate(
                    {"model": "gauss", "backend": "jax"}
                )


class PoolOperationTests(unittest.TestCase):
    def test_source_loader_normalizes_theta_axis_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "theta_last.h5"
            theta = np.linspace(10.0, 14.0, 31)
            data_yxt = np.arange(5 * 6 * theta.size, dtype=np.int32).reshape(5, 6, theta.size)
            with h5py.File(h5_path, "w") as handle:
                measurement = handle.create_group("entry_last").create_group("measurement")
                measurement.create_dataset("Frame", data=data_yxt)
                measurement.create_dataset("th", data=theta)
            history = WorkflowHistory()
            source = create_source_node(history, h5_path, "entry_last")
            self.assertEqual(source.metadata["theta_axis"], 2)
            bundle = WorkflowExecutor(
                history, ArtifactStore(Path(tmp) / "artifacts"), default_registry()
            )._load_external_scan(source)
            np.testing.assert_array_equal(bundle.arrays["data_yxt"], data_yxt)
            np.testing.assert_array_equal(bundle.arrays["theta"], theta)
            self.assertEqual(bundle.arrays["data_yxt"].shape, (5, 6, 31))

    def test_source_to_pool_runs_against_existing_repository(self):
        from roc_mcs.roi.spatial_pooling import PoolSpec, pool_yxt

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            h5_path = folder / "tiny.h5"
            raw = _make_synthetic_h5(h5_path)
            history = WorkflowHistory()
            source = create_source_node(history, h5_path, "entry42")
            pool_node = history.add_node(
                "preprocess.pool",
                "Pool 3/2",
                parent_ids=[source.id],
                input_refs={"scan": {"node_id": source.id, "output": "scan"}},
                parameters={"size": 3, "stride": 2},
            )
            history.transition(pool_node.id, NodeStatus.RUNNING)
            store = ArtifactStore(folder / "project" / "artifacts")
            executor = WorkflowExecutor(history, store, default_registry())
            outcome = executor.execute_node(pool_node.id)
            history.transition(
                pool_node.id,
                NodeStatus.SUCCEEDED,
                output_refs={port: outcome.artifact_id for port in outcome.output_ports},
                summary=outcome.summary,
            )
            result = store.read(outcome.artifact_id)
            expected = pool_yxt(np.moveaxis(raw, 0, -1), PoolSpec(3, 2))
            np.testing.assert_array_equal(result.arrays["data_yxt"], expected)
            self.assertEqual(result.metadata["pool"], {"size": 3, "stride": 2})
            self.assertEqual(result.arrays["data_yxt"].shape[:2], expected.shape[:2])

    def test_source_file_change_invalidates_registered_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "tiny.h5"
            _make_synthetic_h5(h5_path)
            history = WorkflowHistory()
            source = create_source_node(history, h5_path, "entry42")
            with h5py.File(h5_path, "a") as handle:
                handle.create_dataset("provenance_touch", data=np.arange(3))
            executor = WorkflowExecutor(
                history, ArtifactStore(Path(tmp) / "artifacts"), default_registry()
            )
            with self.assertRaisesRegex(RuntimeError, "HDF5 изменился"):
                executor._load_external_scan(source)

    def test_pool_branches_have_distinct_grid_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "tiny.h5"
            _make_synthetic_h5(h5_path)
            history = WorkflowHistory()
            source = create_source_node(history, h5_path, "entry42")
            registry = default_registry()
            source_bundle = WorkflowExecutor(
                history, ArtifactStore(Path(tmp) / "artifacts"), registry
            )._load_external_scan(source)
            first = registry.execute(
                "preprocess.pool", {"scan": source_bundle}, {"size": 3, "stride": 1}
            )
            second = registry.execute(
                "preprocess.pool", {"scan": source_bundle}, {"size": 3, "stride": 3}
            )
            self.assertNotEqual(
                first.bundle.metadata["grid_id"],
                second.bundle.metadata["grid_id"],
            )


class FitMaskPropagationTests(unittest.TestCase):
    def test_fit_receives_strict_detector_coverage_mask_even_without_spike_mask(self):
        import importlib.util
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import patch
        from roc_mcs.analysis.operations import fit_backend_availability
        from roc_mcs.fitting.registry import MODEL_SPECS

        availability = fit_backend_availability()
        backend = "jax" if availability["jax"] else ("lmfit" if availability["lmfit"] else None)
        if backend is None:
            self.skipTest("Для проверки Fit adapter требуется JAX или LMFit")

        registry = default_registry()
        theta = np.linspace(10.0, 14.0, 31)
        rng = np.random.default_rng(20261106)
        source_data = rng.poisson(4.0, size=(7, 7, theta.size)).astype(np.float64)
        source = ArtifactBundle(
            arrays={
                "data_yxt": source_data,
                "theta": theta,
                "x_coords": np.arange(7, dtype=float),
                "y_coords": np.arange(7, dtype=float),
            },
            metadata={
                "grid_id": "fit-mask-source-grid",
                "pool": {"size": 1, "stride": 1},
                "pool_signature": "raw",
            },
        )
        masked = registry.execute(
            "preprocess.detector_bad_pixels", {"scan": source}, {"pixels": [(3, 3)]}
        ).bundle
        pooled = registry.execute(
            "preprocess.pool", {"scan": masked}, {"size": 3, "stride": 1}
        ).bundle
        peaks = ArtifactBundle(
            tables={"peak_catalog": pd.DataFrame(columns=["peak_id", "x", "y", "theta", "theta_index"])},
            metadata={"grid_id": pooled.metadata["grid_id"]},
        )
        captured = {}
        data = np.asarray(pooled.arrays["data_yxt"], dtype=float)

        def fake_fit_scan(*args, **kwargs):
            captured["valid_mask"] = np.asarray(kwargs["valid_mask"], dtype=bool).copy()
            return SimpleNamespace(
                theta=np.asarray(args[1]),
                theta_valid=np.ones(np.asarray(args[1]).shape, dtype=bool),
                fit_cube=np.zeros_like(data),
                residual_cube=np.zeros_like(data),
                valid_mask=captured["valid_mask"],
                pixel_fit_catalog=pd.DataFrame(columns=["fit_status", "x", "y", "n_fitted"]),
                peak_catalog=pd.DataFrame(),
                fitted_peak_catalog=pd.DataFrame(),
                image_shape=data.shape[:2],
                model_id="pvoigt", model_version="test", backend=backend,
                run_id="synthetic-fit-run", fit_config={},
                component_param_names=[], pixel_param_names=[],
            )

        parameters = registry.get("fit.scan").validate({
            "model": "pvoigt", "backend": backend, "center_window": 0.5,
            "add_background": True, "allow_center_shift": True,
            "use_spike_mask": False,
        })
        with patch("roc_mcs.fitting.pipeline.fit_scan", side_effect=fake_fit_scan):
            result = registry.get("fit.scan").execute(
                {"scan": pooled, "peaks": peaks}, parameters
            ).bundle

        self.assertEqual(captured["valid_mask"].shape, data.shape)
        self.assertEqual(int(captured["valid_mask"].all(axis=-1).sum()), 16)
        self.assertFalse(np.any(captured["valid_mask"][:, :, :][1:4, 1:4, :]))
        self.assertEqual(result.metadata["detector_bad_pixel_count"], 1)
        self.assertEqual(result.metadata["detector_invalid_samples"], 31)
        self.assertEqual(result.metadata["analysis_invalid_bins"], 9)
        self.assertEqual(result.metadata["fit_excluded_samples"], 9 * 31)


if __name__ == "__main__":
    unittest.main(verbosity=2)
