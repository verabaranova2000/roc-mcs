from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from roc_mcs.analysis.artifacts import ArtifactBundle, ArtifactStore
from roc_mcs.analysis.operations import (
    _theta_axis_sampling_metadata,
    _validate_branch_correlation,
    default_registry,
)


def _catalog() -> pd.DataFrame:
    source_grid = np.asarray(
        [[0, 0, 0], [1, 0, 1], [2, 1, 1], [3, 1, 2]], dtype=np.int64
    )
    partner_grid = source_grid + np.asarray([20, 1, 2], dtype=np.int64)
    source = pd.DataFrame({
        "peak_id": np.arange(100, 104), "branch_id": 10,
        "theta_index": source_grid[:, 0], "theta": source_grid[:, 0] * 15.0,
        "x": source_grid[:, 2], "y": source_grid[:, 1],
    })
    partner = pd.DataFrame({
        "peak_id": np.arange(200, 204), "branch_id": 20,
        "theta_index": partner_grid[:, 0], "theta": partner_grid[:, 0] * 15.0 + 5.0,
        "x": partner_grid[:, 2], "y": partner_grid[:, 1],
    })
    return pd.concat([source, partner], ignore_index=True)


def test_branch_correlation_is_registered_as_next_stage_after_branch_detection():
    spec = default_registry().get("peaks.branch_correlation")
    assert spec.output_ports == ("branch_correlation",)
    assert "Определение дифракционных ветвей" in spec.description


def test_theta_sampling_metadata_preserves_signed_uniform_step_and_detects_irregular_axis():
    regular = _theta_axis_sampling_metadata(np.asarray([2.0, 17.0, 32.0, 47.0]))
    assert regular["theta_index_step_arcsec"] == pytest.approx(15.0)
    assert regular["theta_index_step_uniform"] is True

    irregular = _theta_axis_sampling_metadata(np.asarray([2.0, 17.0, 33.0, 48.0]))
    assert irregular["theta_index_step_uniform"] is False

    descending = _theta_axis_sampling_metadata(np.asarray([47.0, 32.0, 17.0, 2.0]))
    assert descending["theta_index_step_arcsec"] == pytest.approx(-15.0)
    assert descending["theta_index_step_uniform"] is True


def test_branch_correlation_validator_accepts_manual_ids_without_guessing_theta_step():
    result = _validate_branch_correlation({
        "source_branch_id": "34",
        "partner_branch_ids": "72; 32",
        "xy_tolerance_px": 1.5,
        "theta_tolerance_arcsec": 45.0,
        "assignment_margin": 0.2,
    })
    assert result["source_branch_id"] == 34
    assert result["partner_branch_ids"] == [72, 32]
    assert "theta_step_arcsec" not in result


def test_branch_correlation_validator_rejects_duplicate_or_self_partner():
    with pytest.raises(ValueError, match="повторяющиеся"):
        _validate_branch_correlation({"source_branch_id": 34, "partner_branch_ids": "72,72"})
    with pytest.raises(ValueError, match="не может быть указана"):
        _validate_branch_correlation({"source_branch_id": 34, "partner_branch_ids": "34"})


def test_branch_correlation_operation_uses_measured_theta_step_and_roundtrips_tables(tmp_path):
    registry = default_registry()
    parent = ArtifactBundle(
        tables={"branch_catalog": _catalog()},
        metadata={
            "grid_id": "synthetic-grid",
            "branch_algorithm": "test",
            "n_branches": 2,
            "theta_index_step_arcsec": 15.0,
            "theta_index_step_uniform": True,
        },
    )
    result = registry.execute(
        "peaks.branch_correlation",
        {"branches": parent},
        {
            "source_branch_id": "10",
            "partner_branch_ids": "20",
            "xy_tolerance_px": 1.5,
            "theta_tolerance_arcsec": 20.0,
            "assignment_margin": 0.2,
        },
    )
    assert result.output_ports == ("branch_correlation",)
    assert result.bundle.metadata["theta_step_source"] == "parent_scan_axis_metadata"
    assert result.bundle.metadata["theta_index_step_arcsec"] == pytest.approx(15.0)
    assert len(result.bundle.tables["branch_correlation_pairs"]) == 1
    assert len(result.bundle.tables["branch_correlation_assignments"]) == 4
    assert set(result.bundle.tables["branch_correlation_assignments"]["assignment_class"]) == {"partner"}

    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id = store.write(result.bundle)
    loaded = store.read(artifact_id)
    assert set(loaded.tables) == {
        "branch_correlation_pairs",
        "branch_correlation_assignments",
        "branch_correlation_candidates",
    }
    assert loaded.tables["branch_correlation_assignments"]["assigned_partner_branch_id"].dtype.name == "Int64"
    pd.testing.assert_frame_equal(
        loaded.tables["branch_correlation_assignments"],
        result.bundle.tables["branch_correlation_assignments"],
    )


def test_branch_correlation_rejects_nonuniform_or_missing_theta_axis_metadata():
    params = {
        "source_branch_id": 10,
        "partner_branch_ids": [20],
        "xy_tolerance_px": 1.5,
        "theta_tolerance_arcsec": 20.0,
        "assignment_margin": 0.2,
    }
    with pytest.raises(ValueError, match="равномерную ось θ"):
        default_registry().execute(
            "peaks.branch_correlation",
            {"branches": ArtifactBundle(
                tables={"branch_catalog": _catalog()},
                metadata={"theta_index_step_arcsec": 15.0, "theta_index_step_uniform": False},
            )},
            params,
        )
    with pytest.raises(ValueError, match="отсутствуют метаданные равномерности"):
        default_registry().execute(
            "peaks.branch_correlation",
            {"branches": ArtifactBundle(tables={"branch_catalog": _catalog()}, metadata={})},
            params,
        )


def test_branch_correlation_rejects_catalog_with_only_one_distinct_branch():
    single_branch = _catalog().query("branch_id == 10").copy()
    parent = ArtifactBundle(
        tables={"branch_catalog": single_branch},
        metadata={
            "theta_index_step_arcsec": 15.0,
            "theta_index_step_uniform": True,
        },
    )
    with pytest.raises(ValueError, match="как минимум две различные дифракционные ветви"):
        default_registry().execute(
            "peaks.branch_correlation",
            {"branches": parent},
            {
                "source_branch_id": 10,
                "partner_branch_ids": [20],
                "xy_tolerance_px": 1.5,
                "theta_tolerance_arcsec": 20.0,
                "assignment_margin": 0.2,
            },
        )


def test_branch_correlation_rejects_ids_not_present_in_parent_catalog():
    parent = ArtifactBundle(
        tables={"branch_catalog": _catalog()},
        metadata={
            "theta_index_step_arcsec": 15.0,
            "theta_index_step_uniform": True,
        },
    )
    with pytest.raises(ValueError, match="отсутствуют branch_id"):
        default_registry().execute(
            "peaks.branch_correlation",
            {"branches": parent},
            {
                "source_branch_id": 10,
                "partner_branch_ids": [99],
                "xy_tolerance_px": 1.5,
                "theta_tolerance_arcsec": 20.0,
                "assignment_margin": 0.2,
            },
        )


def test_artifact_store_reads_metadata_without_reading_payload(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact_id = store.write(ArtifactBundle(
        arrays={"large": np.arange(1000)},
        tables={"values": pd.DataFrame({"value": np.arange(10)})},
        metadata={"n_branches": 3, "grid_id": "test-grid"},
    ))

    def fail_read(_artifact_id):
        raise AssertionError("read_metadata must not materialize the complete artifact")

    monkeypatch.setattr(store, "read", fail_read)
    assert store.read_metadata(artifact_id) == {"grid_id": "test-grid", "n_branches": 3}
