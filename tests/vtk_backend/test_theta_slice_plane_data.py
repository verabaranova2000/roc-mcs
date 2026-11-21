from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import VTKVolumeScene


def _marker_only_scene(*, enabled=True, index=1):
    scene = object.__new__(VTKVolumeScene)
    scene._theta_slice_marker_source = vtk.vtkPlaneSource()
    scene._theta_slice_marker_source.SetXResolution(1)
    scene._theta_slice_marker_source.SetYResolution(1)
    scene._theta_slice_marker_mapper = vtk.vtkPolyDataMapper()
    scene._theta_slice_marker_mapper.SetInputConnection(
        scene._theta_slice_marker_source.GetOutputPort()
    )
    scene._theta_slice_marker_actor = vtk.vtkActor()
    scene._theta_slice_marker_actor.SetMapper(scene._theta_slice_marker_mapper)
    scene._theta_slice_marker_actor.SetUserMatrix(vtk.vtkMatrix4x4())
    scene._theta_slice_marker_enabled = bool(enabled)
    scene._theta_slice_marker_opacity = 0.12
    scene._scene_user_matrix = vtk.vtkMatrix4x4()
    scene._scene_user_matrix.Identity()
    scene._display_x = np.asarray([-2.0, 0.5, 3.0], dtype=np.float64)
    scene._display_y = np.asarray([-1.0, 1.5], dtype=np.float64)
    # Descending theta verifies that the plane follows the selected physical
    # coordinate, not an assumed monotonically increasing axis.
    scene._display_theta = np.asarray([10.0, 4.0, -3.0], dtype=np.float64)
    scene._theta_slice_index = int(index)
    scene.payload = SimpleNamespace()
    scene.theme = "dark"
    scene._style_theta_slice_marker = VTKVolumeScene._style_theta_slice_marker.__get__(scene)
    scene._style_theta_slice_marker()
    return scene


def test_theta_marker_is_one_abstract_plane_at_selected_physical_theta():
    scene = _marker_only_scene(enabled=True, index=1)

    scene._update_theta_slice_marker_geometry()

    source = scene._theta_slice_marker_source
    source.Update()
    points = vtk_to_numpy(source.GetOutput().GetPoints().GetData())
    np.testing.assert_allclose(
        points,
        [
            [-2.0, -1.0, 4.0],
            [3.0, -1.0, 4.0],
            [-2.0, 1.5, 4.0],
            [3.0, 1.5, 4.0],
        ],
    )
    actor = scene._theta_slice_marker_actor
    assert actor.GetVisibility() == 1
    assert actor.GetMapper().GetInput().GetNumberOfCells() == 1
    assert actor.GetProperty().GetOpacity() == 0.12
    # It is a geometric plane with no intensity scalars/LUT.
    assert actor.GetMapper().GetInput().GetPointData().GetScalars() is None


def test_theta_marker_is_hidden_by_default_and_geometry_does_not_touch_volume_data():
    scene = _marker_only_scene(enabled=False, index=2)
    scene._update_theta_slice_marker_geometry()
    scene._theta_slice_marker_source.Update()
    assert scene._theta_slice_marker_actor.GetVisibility() == 0
    assert scene._theta_slice_marker_source.GetOutput().GetNumberOfCells() == 1

    before = scene._display_theta.copy()
    scene._theta_slice_index = 0
    scene._update_theta_slice_marker_geometry()
    np.testing.assert_array_equal(scene._display_theta, before)
    assert scene._theta_slice_marker_source.GetOrigin()[2] == 10.0
    assert scene._theta_slice_marker_actor.GetVisibility() == 0


def test_theta_marker_opacity_range_matches_existing_theta_boundary_planes():
    scene = _marker_only_scene()
    scene.set_theta_slice_marker_opacity(0.20, render=False)
    prop = scene._theta_slice_marker_actor.GetProperty()
    assert prop.GetOpacity() == 0.20
    assert prop.GetEdgeVisibility() == 1
    assert prop.GetEdgeOpacity() == 0.92
    assert prop.GetLineWidth() == 1.5
    assert prop.GetColor() == (0.25, 0.74, 0.69)
