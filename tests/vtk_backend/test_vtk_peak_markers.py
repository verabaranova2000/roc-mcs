from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import VTKVolumeScene


def _marker_scene():
    scene = object.__new__(VTKVolumeScene)
    scene.vtk = vtk
    scene._scene_user_matrix = vtk.vtkMatrix4x4()
    scene._scene_user_matrix.Identity()
    scene.renderer = vtk.vtkRenderer()
    scene._peak_marker_polydata = vtk.vtkPolyData()
    scene._peak_marker_mapper = vtk.vtkPolyDataMapper()
    scene._peak_marker_mapper.SetInputData(scene._peak_marker_polydata)
    scene._peak_marker_mapper.SetScalarModeToUsePointData()
    scene._peak_marker_mapper.SetColorModeToDirectScalars()
    scene._peak_marker_mapper.ScalarVisibilityOn()
    scene._peak_marker_actor = vtk.vtkActor()
    scene._peak_marker_actor.SetMapper(scene._peak_marker_mapper)
    scene._peak_marker_actor.SetUserMatrix(scene._scene_user_matrix)
    scene._peak_marker_actor.GetProperty().SetRepresentationToPoints()
    scene._peak_marker_actor.GetProperty().SetPointSize(5.0)
    scene._peak_marker_actor.GetProperty().SetOpacity(0.95)
    scene.renderer.AddActor(scene._peak_marker_actor)
    scene._peak_marker_positions_physical = np.empty((0, 3), dtype=np.float64)
    scene._peak_marker_colors = np.empty((0, 4), dtype=np.uint8)
    scene._peak_marker_size = 5.0
    scene._peak_marker_opacity = 0.95
    scene._peak_marker_visible = False
    scene._physical_x = np.arange(4, dtype=np.float64)
    scene._physical_y = np.arange(3, dtype=np.float64)
    scene._physical_theta = np.arange(6, dtype=np.float64)
    scene._axis_scales = (1.0, 1.0, 1.0)
    scene._request_render = lambda: None
    return scene


def test_peak_markers_map_catalog_indices_to_physical_xyz_and_branch_rgba():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": [10, 11, 12],
        "x": [0, 2, 3],
        "y": [1, 1, 2],
        "theta": [1.5, 2.5, 4.5],
        "branch_id": [0, 1, 1],
    })
    count = scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        color_by_branch=True, render=False,
    )
    assert count == 3
    assert scene._peak_marker_actor.GetVisibility() == 1
    assert scene._peak_marker_polydata.GetNumberOfPoints() == 3
    assert scene._peak_marker_polydata.GetNumberOfVerts() == 3
    np.testing.assert_allclose(
        vtk_to_numpy(scene._peak_marker_polydata.GetPoints().GetData()),
        [[0.0, 1.0, 1.5], [2.0, 1.0, 2.5], [3.0, 2.0, 4.5]],
    )
    rgba = vtk_to_numpy(scene._peak_marker_polydata.GetPointData().GetScalars())
    assert rgba.shape == (3, 4)
    assert not np.array_equal(rgba[0], rgba[1])
    np.testing.assert_array_equal(rgba[1], rgba[2])


def test_peak_marker_positions_follow_display_axis_scales_without_mutating_physical_catalog():
    scene = _marker_scene()
    catalog = pd.DataFrame({"peak_id": [1], "x": [0], "y": [1], "theta": [1.5]})
    scene.set_peak_markers(catalog, np.arange(4.0), np.arange(3.0), visible=True, render=False)
    scene._axis_scales = (2.0, 1.0, 0.5)
    scene._update_peak_marker_polydata()
    # The display transform is centered on each physical axis' midpoint.
    np.testing.assert_allclose(
        vtk_to_numpy(scene._peak_marker_polydata.GetPoints().GetData()),
        [[-1.5, 1.0, 2.0]],
    )
    np.testing.assert_allclose(scene._peak_marker_positions_physical, [[0.0, 1.0, 1.5]])


def test_peak_markers_filter_events_outside_the_current_roi_grid():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": [1, 2, 3],
        "x": [0, 3, 8],
        "y": [1, 1, 1],
        "theta": [1.5, 2.5, 2.0],
        "branch_id": [0, 1, 2],
    })
    count = scene.set_peak_markers(
        catalog, np.arange(9.0), np.arange(3.0), visible=True, render=False
    )
    assert count == 2
    assert scene._peak_marker_polydata.GetNumberOfPoints() == 2
