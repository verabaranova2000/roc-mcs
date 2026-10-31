from __future__ import annotations

import ast
import re
from pathlib import Path

from roc_mcs.gui.roi_explorer.volume_rendering_runtime import (
    LIVE_DISPLAY_SETTINGS,
    build_live_display_patch,
    can_update_live_display,
    volume_opacity_scale,
)


def test_default_opacity_mode_preserves_legacy_plotly_appearance():
    assert volume_opacity_scale() == "uniform"
    assert volume_opacity_scale(None) == "uniform"
    assert volume_opacity_scale("unknown") == "uniform"


def test_soft_opacity_transfer_is_valid_monotonic_and_continuous_at_ends():
    scale = volume_opacity_scale("soft")
    assert len(scale) >= 4
    assert scale[0] == [0.0, 0.0]
    assert scale[-1] == [1.0, 1.0]
    assert all(0.0 <= x <= 1.0 and 0.0 <= alpha <= 1.0 for x, alpha in scale)
    assert all(scale[i][0] < scale[i + 1][0] for i in range(len(scale) - 1))
    assert all(scale[i][1] <= scale[i + 1][1] for i in range(len(scale) - 1))
    assert scale[2][1] < scale[-2][1]


def test_opacity_mode_is_display_only_and_does_not_rebuild_voxel_lattice():
    assert "volume_opacity_mode" in LIVE_DISPLAY_SETTINGS
    before = {"smooth_sigma": 0.5, "volume_stride": 2, "volume_opacity_mode": "uniform"}
    after = {**before, "volume_opacity_mode": "soft"}
    assert can_update_live_display(before, after)


def test_soft_opacity_live_patch_changes_only_volume_transfer_function():
    patch = build_live_display_patch(
        {"volume_opacity_mode": "soft", "projection_colorscale": "Greys"},
        theme="dark",
        changed_settings={"volume_opacity_mode"},
    )
    assert patch["volume_opacityscale"] == volume_opacity_scale("soft")
    assert "projection_colorscale" not in patch
    assert "volume_colorscale" not in patch


def test_theme_and_volume_palette_do_not_modify_projection_palette():
    theme_patch = build_live_display_patch(
        {"volume_colormap": "cyan", "projection_colorscale": "Greys"},
        theme="dark",
        changed_settings={"theme"},
    )
    volume_patch = build_live_display_patch(
        {"volume_colormap": "cyan", "projection_colorscale": "Greys"},
        theme="dark",
        changed_settings={"volume_colormap"},
    )
    assert "projection_colorscale" not in theme_patch
    assert "projection_colorscale" not in volume_patch


def _widget_html() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "src/roc_mcs/gui/roi_explorer/volume_plot_widget.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_HTML"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_HTML template was not found")


def test_middle_pan_is_coalesced_to_animation_frames_and_prevents_plotly_double_handling():
    html = _widget_html()
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    js = scripts[-1]
    assert "stopImmediatePropagation()" in js
    assert "requestAnimationFrame(flushPanFrame)" in js
    assert "let panRaf=0" in js
    assert "pendingCameraApply" in js
    assert "cameraApplyBusy||renderBusy||resizeBusy||displayBusy" in js


def test_fit_camera_resets_translation_but_keeps_direction_and_up_vector():
    js = re.findall(r"<script>(.*?)</script>", _widget_html(), flags=re.S)[-1]
    assert "center={x:0,y:0,z:0}" in js
    assert "setCamera({center:center,eye:{x:d.x*r,y:d.y*r,z:d.z*r}})" in js
    # fitView must not replace camera.up: the current up vector is preserved.
    fit_body = re.search(r"function fitView\(\)\{(.*?)\n\}", js, flags=re.S)
    assert fit_body
    assert "camera.up" not in fit_body.group(1)


def test_benchmark_no_longer_embeds_large_figure_in_sethtml_document():
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "src/roc_mcs/gui/roi_explorer/volume_rendering_benchmark.py"
    code = source.read_text(encoding="utf-8")
    assert "window.beginRoi3dBenchTransfer" in code
    assert "window.receiveRoi3dBenchChunk" in code
    assert "transfer_chunk_chars = 64 * 1024" in code
    assert "view.setHtml(html, base_url)" in code
    assert "figure_literal" not in code
    assert "camera_relayout_ms" in code
    assert "p95:percentile(frameIntervals,0.95)" in code
