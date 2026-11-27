from __future__ import annotations

import json

import pytest

from roc_mcs.gui.roi_explorer.vtk_backend.scan_view_state import (
    load_view_state,
    normalize_view_state,
    save_view_state,
)


def _valid_state():
    return {
        "schema_version": 1,
        "axis_scales": [1.25, 0.8, 2.0],
        "transfer": {
            "low_percent": 12.5,
            "high_percent": 91.0,
            "opacity": 0.24,
            "gamma": 1.8,
            "sample_factor": 0.9,
            "unit_distance": 1.2,
            "palette": "magma",
            "projection_enabled": True,
            "projection_opacity": 0.7,
            "projection_palette": "viridis",
        },
        "camera": {
            "position": [10.0, 20.0, 30.0],
            "focal_point": [0.0, 0.0, 0.0],
            "view_up": [0.0, 1.0, 0.0],
            "parallel_projection": False,
            "parallel_scale": 42.0,
            "view_angle": 30.0,
        },
        "scene": {
            "axes_visible": True,
            "bounding_box_visible": False,
            "axis_ticks_visible": True,
            "grid_visible": False,
        },
    }


def test_scan_view_state_round_trip_and_atomic_replace(tmp_path):
    target = tmp_path / "scan.roi-analysis" / "view_state.json"
    state = _valid_state()
    save_view_state(target, state)
    loaded = load_view_state(target)
    assert loaded == normalize_view_state(state)
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1
    assert list(target.parent.iterdir()) == [target]


def test_scan_view_state_discards_invalid_optional_values(tmp_path):
    state = _valid_state()
    state["axis_scales"] = [1.0, float("nan"), 2.0]
    state["transfer"]["opacity"] = 9.0
    state["camera"]["position"] = [0.0, 0.0, 0.0]
    normalized = normalize_view_state(state)
    assert "axis_scales" not in normalized
    assert "opacity" not in normalized["transfer"]
    assert "camera" not in normalized

    target = tmp_path / "missing.json"
    assert load_view_state(target) is None


def test_unknown_view_state_schema_is_rejected():
    with pytest.raises(ValueError, match="Неподдерживаемая версия"):
        normalize_view_state({"schema_version": 999})
