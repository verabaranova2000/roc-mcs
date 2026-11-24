from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

vtk = pytest.importorskip("vtk")
from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import VTKVolumeScene


def test_peak_glyph_picking_returns_branch_id_count_and_matching_color():
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(400, 400)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    scene = VTKVolumeScene(render_window, interactor, render_callback=lambda: None)
    try:
        scene._physical_x = np.arange(20, dtype=np.float64)
        scene._physical_y = np.arange(20, dtype=np.float64)
        scene._physical_theta = np.arange(10, dtype=np.float64)
        catalog = pd.DataFrame({
            "peak_id": [11, 12, 13],
            "branch_id": [4, 4, 5],
            "x": [5, 10, 15],
            "y": [5, 10, 15],
            "theta": [0.0, 1.0, 2.0],
        })
        assert scene.set_peak_markers(
            catalog, np.arange(20), np.arange(20), visible=True, render=False
        ) == 3
        legend = scene.branch_legend_data()
        assert legend["total_branches"] == 2
        assert legend["total_events"] == 3
        assert legend["entries"][0]["branch_id"] == 4
        assert legend["entries"][0]["event_count"] == 2

        renderer = scene.renderer
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(10.0, 10.0, 1.0)
        camera.SetPosition(10.0, 10.0, 100.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetParallelProjection(1)
        camera.SetParallelScale(20.0)
        renderer.ResetCameraClippingRange()
        render_window.Render()
        renderer.SetWorldPoint(5.0, 5.0, 0.0, 1.0)
        renderer.WorldToDisplay()
        display_x, display_y, _depth = renderer.GetDisplayPoint()

        info = scene.pick_peak_at_display(display_x, display_y, tolerance=0.05)
        assert info is not None
        assert info["peak_id"] == 11
        assert info["branch_id"] == 4
        assert info["branch_event_count"] == 2
        assert info["color"] == legend["entries"][0]["color"]
        assert scene.diagnostics()["peak_marker_pickable"] is True
    finally:
        scene.close()
        render_window.Finalize()


def test_peak_tooltip_color_matches_uniform_marker_color_when_branch_coloring_is_off():
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(300, 300)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    scene = VTKVolumeScene(render_window, interactor, render_callback=lambda: None)
    try:
        scene._physical_x = np.arange(12, dtype=np.float64)
        scene._physical_y = np.arange(12, dtype=np.float64)
        scene._physical_theta = np.arange(4, dtype=np.float64)
        catalog = pd.DataFrame({
            "peak_id": [7], "branch_id": [3],
            "x": [5], "y": [5], "theta": [1.0],
        })
        scene.set_peak_markers(
            catalog, np.arange(12), np.arange(12), visible=True,
            color_by_branch=False, uniform_color="#C04080", render=False,
        )
        camera = scene.renderer.GetActiveCamera()
        camera.SetFocalPoint(5.0, 5.0, 1.0)
        camera.SetPosition(5.0, 5.0, 100.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetParallelProjection(1)
        camera.SetParallelScale(10.0)
        scene.renderer.ResetCameraClippingRange()
        render_window.Render()
        scene.renderer.SetWorldPoint(5.0, 5.0, 1.0, 1.0)
        scene.renderer.WorldToDisplay()
        display_x, display_y, _ = scene.renderer.GetDisplayPoint()
        info = scene.pick_peak_at_display(display_x, display_y, tolerance=0.05)
        assert info is not None
        assert info["branch_id"] == 3
        assert info["color"] == "#C04080"
    finally:
        scene.close()
        render_window.Finalize()
