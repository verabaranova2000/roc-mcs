"""Unit tests for the ROI Explorer's lightweight Plotly display updates."""
from __future__ import annotations

from roc_mcs.gui.roi_explorer.volume_rendering_runtime import (
    build_live_display_patch,
    can_update_live_display,
    changed_setting_keys,
)


def test_palette_and_contrast_are_live_display_changes() -> None:
    previous = {
        "volume_colormap": "magma",
        "isomin_fraction": 0.16,
        "isomax_fraction": 0.95,
    }
    current = {
        "volume_colormap": "viridis",
        "isomin_fraction": 0.22,
        "isomax_fraction": 0.91,
    }
    assert can_update_live_display(previous, current)


def test_voxel_sampling_changes_require_a_rebuild() -> None:
    assert not can_update_live_display(
        {"smooth_sigma": 0.5}, {"smooth_sigma": 0.75}
    )
    assert not can_update_live_display(
        {"volume_stride": 2}, {"volume_stride": 3}
    )
    assert not can_update_live_display(
        {"quality": "balanced"}, {"quality": "fast"}
    )


def test_unknown_future_setting_is_conservatively_rebuilt() -> None:
    assert not can_update_live_display(
        {"future_scientific_parameter": 1},
        {"future_scientific_parameter": 2},
    )


def test_patch_is_bounded_and_handles_malformed_settings() -> None:
    patch = build_live_display_patch(
        {
            "surface_count": "not-an-int",
            "tick_size": "not-an-int",
            "label_size": 999,
            "isomin_fraction": 0.9,
            "isomax_fraction": 0.1,
            "rotation_speed": float("inf"),
        },
        theme="dark",
    )
    assert patch["surface_count"] == 25
    assert patch["tick_size"] == 12
    assert patch["label_size"] == 18
    assert patch["isomin_fraction"] < patch["isomax_fraction"]
    assert patch["rotation_speed"] == 14.0
    assert patch["projection_colorscale"] == "Greys_r"
    assert len(patch["volume_colorscale"]) == 32


def test_projection_and_grid_settings_are_reflected_in_patch() -> None:
    patch = build_live_display_patch(
        {"show_projection": False, "show_grid": False, "show_colorbar": True}
    )
    assert patch["show_projection"] is False
    assert patch["show_grid"] is False
    assert patch["show_colorbar"] is True


def test_contrast_patch_contains_only_threshold_properties() -> None:
    patch = build_live_display_patch(
        {"volume_colormap": "viridis", "isomin_fraction": 0.2, "isomax_fraction": 0.9},
        changed_settings={"isomin_fraction", "isomax_fraction"},
    )
    assert set(patch) == {"isomin_fraction", "isomax_fraction"}


def test_theme_patch_contains_only_theme_dependent_properties() -> None:
    patch = build_live_display_patch(
        {"projection_colorscale": "Greys"},
        theme="dark",
        changed_settings={"theme"},
    )
    assert set(patch) == {"background", "foreground", "grid", "axisline", "projection_colorscale"}
    assert patch["projection_colorscale"] == "Greys_r"


def test_changed_setting_keys_reports_exact_differences() -> None:
    assert changed_setting_keys(
        {"volume_colormap": "magma", "isomin_fraction": 0.16},
        {"volume_colormap": "magma", "isomin_fraction": 0.2},
    ) == {"isomin_fraction"}
