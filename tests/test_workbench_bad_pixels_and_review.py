from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from roc_mcs.analysis.artifacts import ArtifactBundle, ArtifactStore
from roc_mcs.analysis.engine import build_input_refs
from roc_mcs.analysis.operations import default_registry
from roc_mcs.analysis.workflow_history import NodeStatus, ReviewStatus, WorkflowHistory


def _scan_bundle(shape=(7, 7, 31), *, high_bad_pixel=False):
    ny, nx, nt = shape
    theta = np.linspace(10.0, 14.0, nt)
    rng = np.random.default_rng(20261106)
    data = rng.poisson(4.0, size=shape).astype(np.float64)
    if high_bad_pixel:
        data[2, 3, :] = 250_000.0
    return ArtifactBundle(
        arrays={
            "data_yxt": data,
            "theta": theta,
            "x_coords": np.arange(nx, dtype=float),
            "y_coords": np.arange(ny, dtype=float),
        },
        metadata={
            "grid_id": "source-grid",
            "pool": {"size": 1, "stride": 1},
            "pool_signature": "raw",
            "source_hdf5_modified": False,
        },
    )


class DetectorBadPixelOperationTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_registry()

    def test_measurements_are_preserved_and_excluded_only_by_validity_mask(self):
        source = _scan_bundle(high_bad_pixel=True)
        original = source.arrays["data_yxt"].copy()

        result = self.registry.execute(
            "preprocess.detector_bad_pixels",
            {"scan": source},
            {"pixels": [{"x": 3, "y": 2}, (1, 5)]},
        )
        out = result.bundle

        np.testing.assert_array_equal(source.arrays["data_yxt"], original)
        # Counts from excluded detector sites remain untouched. Exclusion is
        # represented only by the validity masks, never by synthetic zeros.
        np.testing.assert_array_equal(out.arrays["data_yxt"], original)
        self.assertEqual(out.metadata["detector_bad_pixel_replacement"], "none; measured counts preserved")
        self.assertFalse(np.any(out.arrays["valid_mask_yxt"][2, 3, :]))
        self.assertFalse(np.any(out.arrays["valid_mask_yxt"][5, 1, :]))
        self.assertTrue(out.arrays["detector_bad_pixel_mask_yx"][2, 3])
        self.assertEqual(out.metadata["detector_bad_pixel_count"], 2)
        self.assertFalse(out.metadata["source_hdf5_modified"])
        self.assertEqual(result.output_ports, ("scan",))

        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp) / "artifacts")
            ref = store.write(out)
            restored = store.read(ref)
        np.testing.assert_array_equal(restored.arrays["valid_mask_yxt"], out.arrays["valid_mask_yxt"])
        np.testing.assert_array_equal(restored.arrays["data_yxt"], out.arrays["data_yxt"])

    def test_bad_pixel_operation_rejects_invalid_or_out_of_grid_coordinates(self):
        source = _scan_bundle()
        for pixels in ([{"x": -1, "y": 0}], [{"x": 1.5, "y": 1}], [{"x": 99, "y": 0}]):
            with self.subTest(pixels=pixels):
                with self.assertRaises(ValueError):
                    self.registry.execute(
                        "preprocess.detector_bad_pixels",
                        {"scan": source},
                        {"pixels": pixels},
                    )

    def test_bad_pixel_stage_must_precede_pooling_and_cannot_be_repeated(self):
        source = _scan_bundle()
        first = self.registry.execute(
            "preprocess.detector_bad_pixels",
            {"scan": source},
            {"pixels": [(3, 3)]},
        ).bundle
        with self.assertRaises(ValueError):
            self.registry.execute(
                "preprocess.detector_bad_pixels",
                {"scan": first},
                {"pixels": [(2, 2)]},
            )
        pooled = self.registry.execute(
            "preprocess.pool", {"scan": source}, {"size": 3, "stride": 1}
        ).bundle
        with self.assertRaises(ValueError):
            self.registry.execute(
                "preprocess.detector_bad_pixels",
                {"scan": pooled},
                {"pixels": [(2, 2)]},
            )

    def test_empty_bad_pixel_stage_is_an_explicit_auditable_noop(self):
        source = _scan_bundle(high_bad_pixel=True)
        original = source.arrays["data_yxt"].copy()
        result = self.registry.execute(
            "preprocess.detector_bad_pixels", {"scan": source}, {"pixels": []}
        )
        out = result.bundle
        np.testing.assert_array_equal(out.arrays["data_yxt"], original)
        self.assertTrue(np.all(out.arrays["valid_mask_yxt"]))
        self.assertTrue(np.all(out.arrays["analysis_valid_mask_yx"]))
        self.assertEqual(out.metadata["detector_bad_pixel_mode"], "confirm_none")
        self.assertEqual(out.metadata["detector_bad_pixel_count"], 0)
        self.assertTrue(out.metadata["detector_bad_pixel_stage_applied"])
        self.assertFalse(out.metadata["source_hdf5_modified"])

    def test_pool_preserves_valid_counts_and_tracks_partial_coverage(self):
        source = _scan_bundle()
        original = source.arrays["data_yxt"].copy()
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels",
            {"scan": source},
            {"pixels": [(3, 3)]},
        ).bundle
        pooled = self.registry.execute(
            "preprocess.pool",
            {"scan": masked},
            {"size": 3, "stride": 1},
        ).bundle
        valid = pooled.arrays["valid_mask_yxt"]
        affected = pooled.arrays["detector_bad_pixel_mask_yx"]
        self.assertEqual(valid.shape, (5, 5, 31))
        self.assertEqual(int(np.count_nonzero(affected)), 9)
        # The excluded detector value is not included in the derived sum, but
        # the raw source and QC artifact retain that measurement unchanged.
        expected = original[1:4, 1:4, :].sum(axis=(0, 1)) - original[3, 3, :]
        np.testing.assert_array_equal(pooled.arrays["data_yxt"][1, 1, :], expected)
        self.assertTrue(np.all(valid[1:4, 1:4, :]))
        self.assertFalse(np.any(pooled.arrays["analysis_valid_mask_yx"][1:4, 1:4]))
        self.assertTrue(np.all(pooled.arrays["valid_pixel_count_yx"][1:4, 1:4] == 8))
        self.assertTrue(np.all(pooled.arrays["represented_detector_pixel_count_yx"][1:4, 1:4] == 9))
        np.testing.assert_allclose(pooled.arrays["valid_fraction_yx"][1:4, 1:4], 8 / 9)
        self.assertGreater(float(pooled.arrays["data_yxt"][1, 1, :].sum()), 0.0)
        np.testing.assert_array_equal(masked.arrays["data_yxt"], original)
        self.assertTrue(np.all(valid[0, 0, :]))

    def test_pool_bin_with_no_valid_contributors_is_missing_not_zero(self):
        source = _scan_bundle(shape=(7, 7, 11))
        bad_pixels = [
            (x, y) for y in range(3) for x in range(3)
        ]
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels", {"scan": source}, {"pixels": bad_pixels}
        ).bundle
        np.testing.assert_array_equal(masked.arrays["data_yxt"], source.arrays["data_yxt"])
        pooled = self.registry.execute(
            "preprocess.pool", {"scan": masked}, {"size": 3, "stride": 1}
        ).bundle
        self.assertFalse(np.any(pooled.arrays["valid_mask_yxt"][0, 0, :]))
        self.assertFalse(bool(pooled.arrays["analysis_valid_mask_yx"][0, 0]))
        self.assertTrue(np.all(np.isnan(pooled.arrays["data_yxt"][0, 0, :])))
        self.assertEqual(float(pooled.arrays["valid_fraction_yx"][0, 0]), 0.0)

    def test_partial_coverage_is_propagated_through_repeated_pooling(self):
        source = _scan_bundle()
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels", {"scan": source}, {"pixels": [(3, 3)]}
        ).bundle
        first = self.registry.execute(
            "preprocess.pool", {"scan": masked}, {"size": 3, "stride": 1}
        ).bundle
        second = self.registry.execute(
            "preprocess.pool", {"scan": first}, {"size": 3, "stride": 1}
        ).bundle
        self.assertEqual(second.arrays["data_yxt"].shape, (3, 3, 31))
        self.assertTrue(np.all(second.arrays["valid_pixel_count_yx"] <= second.arrays["represented_detector_pixel_count_yx"]))
        self.assertTrue(np.all((second.arrays["valid_fraction_yx"] >= 0) & (second.arrays["valid_fraction_yx"] <= 1)))
        self.assertTrue(np.any(~second.arrays["analysis_valid_mask_yx"]))
        self.assertTrue(np.all(second.arrays["data_yxt"] >= 0))

    def test_partial_pool_bins_are_excluded_from_spike_background(self):
        source = _scan_bundle()
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels", {"scan": source}, {"pixels": [(3, 3)]}
        ).bundle
        pooled = self.registry.execute(
            "preprocess.pool", {"scan": masked}, {"size": 3, "stride": 1}
        ).bundle
        captured = {}

        def fake_refine(data, **kwargs):
            captured["shape"] = np.asarray(data).shape
            shape = np.asarray(data).shape
            return SimpleNamespace(
                spike_mask=np.zeros(shape, dtype=bool),
                clean_data=np.asarray(data, dtype=float).copy(),
                kappa=1.0,
                converged=True,
                iterations=1,
                history=[{"iteration": 0, "kappa": 1.0, "n_spikes": 0}],
                kappa_fits=[],
                background=SimpleNamespace(median=4.0, sigma=1.0, threshold=9.0),
            )

        with patch("roc_mcs.processing.spikes.refine_spike_mask", side_effect=fake_refine):
            result = self.registry.execute(
                "preprocess.spikes", {"scan": pooled},
                {"max_iter": 2, "tol": 1e-3, "collect_kappa_fits": False},
            ).bundle

        # 9 output windows have 8/9 detector coverage; only the 16 complete
        # windows are allowed into the global kappa/background estimate.
        self.assertEqual(captured["shape"], (31, 1, 16))
        self.assertEqual(int(np.count_nonzero(~result.arrays["valid_mask_yxt"].all(axis=-1))), 9)
        self.assertEqual(result.metadata["excluded_detector_pixels"], 1)
        self.assertEqual(result.metadata["excluded_analysis_bins"], 9)

    def test_partial_pool_bins_are_excluded_from_peak_candidates_and_neighbours(self):
        import pandas as pd
        from roc_mcs.peaks import catalog as catalog_module

        source = _scan_bundle()
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels", {"scan": source}, {"pixels": [(3, 3)]}
        ).bundle
        pooled = self.registry.execute(
            "preprocess.pool", {"scan": masked}, {"size": 3, "stride": 1}
        ).bundle
        captured = {}

        def fake_build(data_tyx, theta, **kwargs):
            captured["shape"] = np.asarray(data_tyx).shape
            captured["pixel_valid"] = np.asarray(kwargs["pixel_valid"], dtype=bool).copy()
            return pd.DataFrame(), 0.0, 0.0

        parameters = self.registry.get("peaks.detect").validate({
            "window_size": 3,
            "smooth_window": 7,
            "smooth_polyorder": 2,
            "min_distance": 4,
            "prominence_mode": "sigma",
            "prominence_value": 2.0,
            "min_relative_prominence": 0.07,
            "use_interpolated_data": False,
        })
        with patch.object(catalog_module, "build_peak_catalog", side_effect=fake_build):
            result = self.registry.get("peaks.detect").execute({"scan": pooled}, parameters)

        self.assertEqual(captured["shape"], (31, 5, 5))
        self.assertEqual(int(captured["pixel_valid"].sum()), 16)
        self.assertFalse(np.any(captured["pixel_valid"][1:4, 1:4]))
        self.assertEqual(result.bundle.arrays["peak_count_map"].shape, (5, 5))
        self.assertEqual(result.bundle.metadata["excluded_detector_pixels"], 1)
        self.assertEqual(result.bundle.metadata["excluded_analysis_bins"], 9)

    def test_all_detector_pixels_cannot_be_excluded(self):
        source = _scan_bundle(shape=(2, 2, 11))
        with self.assertRaisesRegex(ValueError, "все пиксели"):
            self.registry.execute(
                "preprocess.detector_bad_pixels",
                {"scan": source},
                {"pixels": [(0, 0), (1, 0), (0, 1), (1, 1)]},
            )

    def test_peak_detection_ignores_bad_pixel_with_extreme_intensity(self):
        theta = np.linspace(10.0, 14.0, 51)
        rng = np.random.default_rng(42)
        data = rng.poisson(3.0, size=(5, 6, theta.size)).astype(float)
        profile = 65.0 * np.exp(-0.5 * ((theta - 12.0) / 0.16) ** 2)
        data[2:4, 2:5, :] += profile[None, None, :]
        data[0, 0, :] = 500_000.0
        source = ArtifactBundle(
            arrays={
                "data_yxt": data,
                "theta": theta,
                "x_coords": np.arange(6, dtype=float),
                "y_coords": np.arange(5, dtype=float),
            },
            metadata={
                "grid_id": "peak-source-grid",
                "pool": {"size": 1, "stride": 1},
                "pool_signature": "raw",
            },
        )
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels",
            {"scan": source},
            {"pixels": [(0, 0)]},
        ).bundle
        peak_result = self.registry.execute(
            "peaks.detect",
            {"scan": masked},
            {
                "window_size": 3,
                "smooth_window": 7,
                "smooth_polyorder": 2,
                "min_distance": 4,
                "prominence_mode": "sigma",
                "prominence_value": 2.0,
                "min_relative_prominence": 0.07,
                "use_interpolated_data": False,
            },
        ).bundle
        catalog = peak_result.tables["peak_catalog"]
        self.assertEqual(peak_result.metadata["excluded_detector_pixels"], 1)
        if not catalog.empty:
            self.assertFalse(np.any((catalog["x"] == 0) & (catalog["y"] == 0)))
        self.assertGreater(peak_result.metadata["n_peaks"], 0)

    def test_spike_estimator_receives_only_valid_detector_traces(self):
        source = _scan_bundle(high_bad_pixel=True)
        masked = self.registry.execute(
            "preprocess.detector_bad_pixels",
            {"scan": source},
            {"pixels": [(3, 2)]},
        ).bundle
        captured = {}

        def fake_refine(data, **kwargs):
            captured["shape"] = np.asarray(data).shape
            captured["max"] = float(np.max(data))
            shape = np.asarray(data).shape
            return SimpleNamespace(
                spike_mask=np.zeros(shape, dtype=bool),
                clean_data=np.asarray(data, dtype=float).copy(),
                kappa=1.0,
                converged=True,
                iterations=1,
                history=[{"iteration": 0, "kappa": 1.0, "n_spikes": 0}],
                kappa_fits=[],
                background=SimpleNamespace(median=4.0, sigma=1.0, threshold=9.0),
            )

        with patch("roc_mcs.processing.spikes.refine_spike_mask", side_effect=fake_refine):
            result = self.registry.execute(
                "preprocess.spikes",
                {"scan": masked},
                {"max_iter": 2, "tol": 1e-3, "collect_kappa_fits": False},
            ).bundle

        self.assertEqual(captured["shape"], (31, 1, 48))
        self.assertLess(captured["max"], 250_000.0)
        self.assertFalse(np.any(result.arrays["valid_mask_yxt"][2, 3, :]))
        # The interpolation result retains the original trace at an excluded
        # detector site; consumers must obey valid_mask_yxt and not display/use it.
        np.testing.assert_array_equal(result.arrays["clean_data_yxt"][2, 3, :], source.arrays["data_yxt"][2, 3, :])
        self.assertEqual(result.metadata["excluded_detector_pixels"], 1)


class BadPixelPeakExclusionTests(unittest.TestCase):
    def test_invalid_pixel_is_not_a_candidate_or_neighbour_contributor(self):
        from roc_mcs.processing.background import BackgroundStats
        from roc_mcs.peaks import catalog as catalog_module

        theta = np.linspace(10.0, 14.0, 21)
        data = np.ones((theta.size, 3, 3), dtype=float) * 10.0
        data[:, 1, 1] = 1_000_000.0
        pixel_valid = np.ones((3, 3), dtype=bool)
        pixel_valid[1, 1] = False
        seen_means = []

        def fake_detect(mean_curve, _config, *, noise_sigma):
            seen_means.append(np.asarray(mean_curve).copy())
            return (
                np.asarray(mean_curve), np.asarray([], dtype=int),
                {"prominences": [], "widths": [], "left_ips": [], "right_ips": []},
                np.asarray([], dtype=bool),
            )

        with patch.object(catalog_module, "detect_peaks", side_effect=fake_detect), patch.object(
            catalog_module, "estimate_background_stats",
            return_value=BackgroundStats(median=1.0, sigma=1.0, threshold=0.5),
        ) as background_estimator:
            catalog, _sigma, _threshold = catalog_module.build_peak_catalog(
                data, theta, window_size=3, pixel_valid=pixel_valid,
            )

        self.assertTrue(catalog.empty)
        self.assertEqual(len(seen_means), 8)  # all valid pixels are candidates
        self.assertTrue(all(np.max(curve) < 100.0 for curve in seen_means))
        masked_input = background_estimator.call_args.args[0]
        self.assertTrue(np.all(np.isnan(masked_input[:, 1, 1])))


class WorkflowReviewTests(unittest.TestCase):
    def _source(self):
        history = WorkflowHistory()
        source = history.add_node(
            "source.scan", "Source", status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "@external-hdf5-scan"},
        )
        return history, source

    def test_review_state_round_trips_and_legacy_nodes_default_to_accepted(self):
        import json

        history, source = self._source()
        candidate = history.add_node(
            "preprocess.detector_bad_pixels", "Bad pixels",
            parent_ids=[source.id],
            input_refs={"scan": {"node_id": source.id, "output": "scan"}},
            status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "d" * 32},
            review_status=ReviewStatus.CANDIDATE,
        )
        history.set_review_status(candidate.id, ReviewStatus.ACCEPTED)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            history.save(path)
            loaded = WorkflowHistory.load(path)
        self.assertEqual(loaded.get(candidate.id).review_status, ReviewStatus.ACCEPTED)

        legacy_payload = {
            "schema_version": 2,
            "nodes": [{
                "id": "legacy-source", "operation_id": "source.scan", "label": "legacy",
                "parent_ids": [], "parameters": {}, "status": "succeeded",
                "inputs": {}, "output_refs": {"scan": "@external-hdf5-scan"},
                "metadata": {},
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(legacy_payload), encoding="utf-8")
            legacy = WorkflowHistory.load(path)
        self.assertEqual(legacy.get("legacy-source").review_status, ReviewStatus.ACCEPTED)

    def test_detector_exclusion_only_from_raw_source(self):
        history, source = self._source()
        refs = build_input_refs(
            history, source.id, "preprocess.detector_bad_pixels",
            {"pixels": [{"x": 1, "y": 2}]},
        )
        self.assertEqual(refs["scan"]["node_id"], source.id)
        pooled = history.add_node(
            "preprocess.pool", "Pool", parent_ids=[source.id],
            input_refs={"scan": {"node_id": source.id, "output": "scan"}},
            status=NodeStatus.SUCCEEDED, output_refs={"scan": "a" * 32},
        )
        with self.assertRaises(ValueError):
            build_input_refs(
                history, pooled.id, "preprocess.detector_bad_pixels",
                {"pixels": [{"x": 1, "y": 2}]},
            )

    def test_all_scientific_stages_require_an_accepted_detector_qc_stage(self):
        history, source = self._source()
        with self.assertRaisesRegex(ValueError, "Битые пиксели"):
            build_input_refs(history, source.id, "preprocess.pool", {"size": 3, "stride": 1})
        with self.assertRaisesRegex(ValueError, "Битые пиксели"):
            build_input_refs(history, source.id, "preprocess.spikes", {})

    def test_candidate_must_be_accepted_before_downstream_stage(self):
        history, source = self._source()
        candidate = history.add_node(
            "preprocess.detector_bad_pixels", "Bad pixels",
            parent_ids=[source.id],
            input_refs={"scan": {"node_id": source.id, "output": "scan"}},
            status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "b" * 32},
            review_status=ReviewStatus.CANDIDATE,
        )
        with self.assertRaises(ValueError):
            build_input_refs(history, candidate.id, "preprocess.pool", {"size": 3, "stride": 1})
        history.set_review_status(candidate.id, ReviewStatus.ACCEPTED)
        refs = build_input_refs(history, candidate.id, "preprocess.pool", {"size": 3, "stride": 1})
        self.assertEqual(refs["scan"]["node_id"], candidate.id)

    def test_rejected_result_remains_in_history_but_cannot_be_an_input(self):
        history, source = self._source()
        candidate = history.add_node(
            "preprocess.pool", "Pool",
            parent_ids=[source.id],
            input_refs={"scan": {"node_id": source.id, "output": "scan"}},
            status=NodeStatus.SUCCEEDED,
            output_refs={"scan": "c" * 32},
            review_status=ReviewStatus.CANDIDATE,
        )
        history.set_review_status(candidate.id, ReviewStatus.REJECTED)
        self.assertEqual(history.get(candidate.id).review_status, ReviewStatus.REJECTED)
        with self.assertRaises(ValueError):
            build_input_refs(history, candidate.id, "preprocess.pool", {"size": 3, "stride": 1})
        self.assertEqual(len(history.nodes), 2)


if __name__ == "__main__":
    unittest.main()
