from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

vtk = pytest.importorskip("vtk")

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import TransferSettings, VTKVolumeScene


def _payload():
    x = np.linspace(0.0, 4.0, 5)
    y = np.linspace(0.0, 4.0, 5)
    theta = np.linspace(0.0, 4.0, 5)
    volume = np.ones((5, 5, 5), dtype=np.float32)
    return SimpleNamespace(
        x=x, y=y, theta=theta, volume=volume,
        projection=np.sum(volume, axis=2), coord_system="xy", roi=None,
        scan_key="correspondence-test", view_mode="full", scalar_dtype="float32",
        finite_count=volume.size, invalid_count=0, nbytes=volume.nbytes,
        masked_detector_pixels=(), smoothing_sigma=0.0,
        spike_report=SimpleNamespace(enabled=False, kappa=0.0, invalid_count=0,
                                     invalid_fraction=0.0, sigma=0.0),
    )


def test_stored_correspondences_render_as_separate_layers_and_emphasise_core():
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    interactor = vtk.vtkGenericRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    scene = VTKVolumeScene(render_window, interactor, render_callback=lambda: None)
    try:
        scene.set_payload(_payload(), TransferSettings())
        catalog = pd.DataFrame({
            "peak_id": [101, 102, 201, 202, 301],
            "branch_id": [10, 10, 20, 20, 30],
            "x": [1, 2, 1, 2, 3],
            "y": [1, 1, 1, 1, 3],
            "theta": [1.0, 2.0, 1.5, 2.5, 2.0],
        })
        assert scene.set_peak_markers(
            catalog, np.asarray(_payload().x), np.asarray(_payload().y),
            visible=True, render=False,
        ) == 5
        matched = np.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.5]]])
        candidates = np.asarray([[[2.0, 1.0, 2.0], [2.0, 1.0, 2.5]]])
        info = scene.set_branch_correspondences(
            matched,
            np.asarray([20]),
            candidate_segments=candidates,
            matched_peak_ids=np.asarray([101, 201]),
            focus_branch_ids=np.asarray([10, 20]),
            visible=True,
            show_candidates=True,
            residual_opacity=0.25,
            render=False,
        )
        assert info == {
            "matched_links": 1, "candidate_links": 1,
            "matched_peaks": 2, "focus_branches": 2,
        }
        assert scene._branch_correlation_matched_actor.GetVisibility() == 1
        assert scene._branch_correlation_candidate_actor.GetVisibility() == 1
        assert scene._branch_correlation_matched_polydata.GetNumberOfLines() == 1
        # Dashed alternatives are represented by three short VTK line cells.
        assert scene._branch_correlation_candidate_polydata.GetNumberOfLines() == 3

        peak_ids = scene._peak_marker_peak_ids
        colors = scene._peak_marker_colors
        alpha_by_id = {int(pid): int(color[3]) for pid, color in zip(peak_ids, colors)}
        assert alpha_by_id[101] == 255
        assert alpha_by_id[201] == 255
        assert alpha_by_id[102] in (63, 64)
        assert alpha_by_id[202] in (63, 64)
        assert alpha_by_id[301] == 255  # a branch outside the selected pair is untouched

        # The correlation actor is independent of the intensity volume.
        scene.set_volume_visible(False, render=False)
        assert scene.volume_visible is False
        assert scene._branch_correlation_matched_actor.GetVisibility() == 1

        # Axis-scale changes rebuild the correlation line in the scene display
        # coordinate system without changing stored physical endpoints.
        physical_before = scene._branch_correlation_matched_segments_physical.copy()
        scene.set_axis_scales(2.0, 1.0, 1.0, render=False)
        np.testing.assert_array_equal(scene._branch_correlation_matched_segments_physical, physical_before)
        assert scene._branch_correlation_matched_polydata.GetNumberOfLines() == 1

        scene.set_branch_correspondences_visible(False, render=False)
        assert scene._branch_correlation_matched_actor.GetVisibility() == 0
        assert scene._branch_correlation_candidate_actor.GetVisibility() == 0
        # Hiding the correlation restores the original point alpha values.
        assert np.all(scene._peak_marker_colors[:, 3] == 255)

        # Exercise VTK's actual rendering pipeline as well as its data objects.
        # This is a renderer smoke test (Qt integration is tested separately).
        render_window.SetSize(420, 320)
        scene.renderer.ResetCamera()
        render_window.Render()
        assert render_window.GetNeverRendered() == 0
    finally:
        scene.close()
        interactor.SetRenderWindow(None)
        render_window.Finalize()
