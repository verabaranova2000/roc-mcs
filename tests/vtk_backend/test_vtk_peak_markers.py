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
    scene._peak_marker_glyph_source = vtk.vtkSphereSource()
    scene._peak_marker_glyph_source.SetRadius(1.0)
    scene._peak_marker_glyph_source.SetThetaResolution(12)
    scene._peak_marker_glyph_source.SetPhiResolution(8)
    scene._peak_marker_mapper = vtk.vtkGlyph3DMapper()
    scene._peak_marker_mapper.SetInputData(scene._peak_marker_polydata)
    scene._peak_marker_mapper.SetSourceConnection(
        scene._peak_marker_glyph_source.GetOutputPort()
    )
    scene._peak_marker_mapper.SetScaleModeToNoDataScaling()
    scene._peak_marker_mapper.SetScaleFactor(0.1)
    scene._peak_marker_mapper.SetScalarModeToUsePointData()
    scene._peak_marker_mapper.SetColorModeToDirectScalars()
    scene._peak_marker_mapper.ScalarVisibilityOn()
    scene._peak_marker_actor = vtk.vtkActor()
    scene._peak_marker_actor.SetMapper(scene._peak_marker_mapper)
    scene._peak_marker_actor.SetUserMatrix(scene._scene_user_matrix)
    scene._peak_marker_actor.GetProperty().SetOpacity(0.95)
    scene.renderer.AddActor(scene._peak_marker_actor)
    scene._peak_marker_positions_physical = np.empty((0, 3), dtype=np.float64)
    scene._peak_marker_colors = np.empty((0, 4), dtype=np.uint8)
    scene._peak_marker_size = 5.0
    scene._peak_marker_opacity = 0.95
    scene._peak_marker_world_radius = 0.1
    scene._peak_marker_visible = False
    scene._branch_edge_polydata = vtk.vtkPolyData()
    scene._branch_edge_mapper = vtk.vtkPolyDataMapper()
    scene._branch_edge_mapper.SetInputData(scene._branch_edge_polydata)
    scene._branch_edge_mapper.SetScalarModeToUseCellData()
    scene._branch_edge_mapper.SetColorModeToDirectScalars()
    scene._branch_edge_mapper.ScalarVisibilityOn()
    scene._branch_edge_actor = vtk.vtkActor()
    scene._branch_edge_actor.SetMapper(scene._branch_edge_mapper)
    scene._branch_edge_actor.SetUserMatrix(scene._scene_user_matrix)
    scene.renderer.AddActor(scene._branch_edge_actor)
    scene._branch_edge_positions_physical = np.empty((0, 2, 3), dtype=np.float64)
    scene._branch_edge_colors = np.empty((0, 4), dtype=np.uint8)
    scene._branch_edge_visible = False
    scene._branch_edge_width = 2.0
    scene._branch_edge_opacity = 0.92
    scene._volume_visible = True
    scene.volume = vtk.vtkVolume()
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


def test_uniform_marker_color_is_applied_when_branch_coloring_is_disabled():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": [10, 11, 12],
        "x": [0, 1, 2],
        "y": [1, 1, 1],
        "theta": [1.0, 2.0, 3.0],
        "branch_id": [0, 1, 2],
    })
    count = scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        color_by_branch=False, uniform_color="#2468AC", render=False,
    )
    assert count == 3
    colors = vtk_to_numpy(scene._peak_marker_polydata.GetPointData().GetScalars())
    expected = np.array([0x24, 0x68, 0xAC, 255], dtype=np.uint8)
    np.testing.assert_array_equal(colors, np.tile(expected, (3, 1)))


def test_missing_branch_id_keeps_peak_marker_with_uniform_fallback_color():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": [10, 11, 12],
        "x": [0, 1, 2],
        "y": [1, 1, 1],
        "theta": [1.0, 2.0, 3.0],
        "branch_id": [0.0, np.nan, 1.0],
    })
    count = scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        color_by_branch=True, uniform_color="#123456", render=False,
    )
    assert count == 3
    colors = vtk_to_numpy(scene._peak_marker_polydata.GetPointData().GetScalars())
    np.testing.assert_array_equal(colors[1], [0x12, 0x34, 0x56, 255])
    assert scene._peak_marker_actor.GetVisibility() == 1


def test_branch_edges_are_rendered_from_peak_ids_and_follow_axis_scaling():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": [10, 11, 12, 13],
        "x": [0, 1, 0, 1],
        "y": [0, 0, 1, 1],
        "theta": [1.0, 2.0, 3.0, 4.0],
        "branch_id": [0, 0, 1, 1],
    })
    edges = pd.DataFrame({
        "peak_id_a": [10, 12],
        "peak_id_b": [11, 13],
        "delta_theta": [1.0, 1.0],
    })
    count = scene.set_branch_edges(
        catalog, edges, np.arange(4.0), np.arange(3.0),
        visible=True, color_by_branch=True, render=False,
    )
    assert count == 2
    assert scene._branch_edge_actor.GetVisibility() == 1
    assert scene._branch_edge_polydata.GetNumberOfPoints() == 4
    assert scene._branch_edge_polydata.GetNumberOfLines() == 2
    colors = vtk_to_numpy(scene._branch_edge_polydata.GetCellData().GetScalars())
    assert not np.array_equal(colors[0], colors[1])

    scene._axis_scales = (2.0, 1.0, 1.0)
    scene._update_branch_edge_polydata()
    display = vtk_to_numpy(scene._branch_edge_polydata.GetPoints().GetData())
    np.testing.assert_allclose(display[:, 0], [-1.5, 0.5, -1.5, 0.5])


def test_volume_visibility_is_independent_from_marker_and_edge_visibility():
    scene = _marker_scene()
    scene.set_peak_markers(
        pd.DataFrame({"peak_id": [1], "x": [1], "y": [1], "theta": [2.0]}),
        np.arange(4.0), np.arange(3.0), visible=True, render=False,
    )
    scene.set_volume_visible(False, render=False)
    assert scene.volume.GetVisibility() == 0
    assert scene._peak_marker_actor.GetVisibility() == 1
    scene.set_volume_visible(True, render=False)
    assert scene.volume.GetVisibility() == 1


def test_nullable_catalogue_columns_keep_valid_markers_and_skip_missing_rows():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": pd.Series([10, 11, 12, 13], dtype="Int64"),
        "x": pd.Series([0, 1, 2, 3], dtype="Int64"),
        "y": pd.Series([1, 1, 1, 1], dtype="Int64"),
        "theta": pd.Series([1.0, 2.0, 3.0, pd.NA], dtype="Float64"),
        "branch_id": pd.Series([0, pd.NA, 1, 2], dtype="Int64"),
    })
    count = scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        color_by_branch=True, uniform_color="#123456", render=False,
    )
    assert count == 3
    colors = vtk_to_numpy(scene._peak_marker_polydata.GetPointData().GetScalars())
    np.testing.assert_array_equal(colors[1], [0x12, 0x34, 0x56, 255])


def test_nullable_peak_ids_are_ignored_when_building_branch_edges():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": pd.Series([10, 11, pd.NA], dtype="Int64"),
        "x": [0, 1, 2], "y": [0, 0, 1], "theta": [1.0, 2.0, 3.0],
        "branch_id": pd.Series([0, 0, pd.NA], dtype="Int64"),
    })
    edges = pd.DataFrame({"peak_id_a": pd.Series([10, pd.NA], dtype="Int64"),
                          "peak_id_b": pd.Series([11, 10], dtype="Int64")})
    count = scene.set_branch_edges(
        catalog, edges, np.arange(4.0), np.arange(3.0), visible=True, render=False,
    )
    assert count == 1
    assert scene._branch_edge_polydata.GetNumberOfLines() == 1


def test_peak_markers_use_real_3d_sphere_glyphs_not_driver_dependent_point_sprites():
    scene = _marker_scene()
    catalog = pd.DataFrame({
        "peak_id": [1, 2], "x": [0, 1], "y": [1, 1],
        "theta": [1.0, 2.0], "branch_id": [0, 1],
    })
    scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True, render=False
    )
    assert scene._peak_marker_mapper.IsA("vtkGlyph3DMapper")
    assert scene._peak_marker_glyph_source.IsA("vtkSphereSource")
    assert scene._peak_marker_mapper.GetScaleMode() == vtk.vtkGlyph3DMapper.SCALE_BY_MAGNITUDE
    assert scene._peak_marker_mapper.GetScaleFactor() > 0.0
    scale_array = scene._peak_marker_polydata.GetPointData().GetArray("PeakMarkerGlyphScale")
    assert scale_array is not None
    assert scale_array.GetNumberOfTuples() == 2
    assert scene._peak_marker_actor.GetVisibility() == 1
    assert scene._peak_marker_actor.GetBounds()[0] < scene._peak_marker_actor.GetBounds()[1]


def test_peak_marker_size_changes_world_glyph_radius_and_remains_bounded():
    scene = _marker_scene()
    catalog = pd.DataFrame({"peak_id": [1], "x": [1], "y": [1], "theta": [2.0]})
    scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        point_size=5.0, render=False,
    )
    default_radius = scene._peak_marker_mapper.GetScaleFactor()
    scene.set_peak_marker_settings(point_size=10.0, render=False)
    larger_radius = scene._peak_marker_mapper.GetScaleFactor()
    scene.set_peak_marker_settings(point_size=20.0, render=False)
    maximum_radius = scene._peak_marker_mapper.GetScaleFactor()
    assert default_radius > 0.0
    assert larger_radius > default_radius
    assert maximum_radius >= larger_radius
    # No sphere grows beyond 0.45 of the detector's spatial pixel pitch.
    assert maximum_radius <= 0.45 + 1e-12


def test_peak_marker_radius_uses_xy_pitch_not_finely_sampled_theta_axis():
    scene = _marker_scene()
    scene._physical_x = np.arange(4, dtype=np.float64)
    scene._physical_y = np.arange(3, dtype=np.float64)
    scene._physical_theta = np.linspace(0.0, 0.0005, 6, dtype=np.float64)
    catalog = pd.DataFrame({
        "peak_id": [1, 2], "x": [0, 1], "y": [1, 1],
        "theta": [0.0001, 0.0002],
    })
    count = scene.set_peak_markers(
        catalog, scene._physical_x, scene._physical_y,
        visible=True, point_size=5.0, render=False,
    )

    assert count == 2
    # X/Y pitch is 1; θ pitch is 1e-4.  The old min(dx,dy,dθ)
    # rule produced a 3.5e-5 radius, effectively invisible in the scene.
    assert scene._peak_marker_reference_pitch == 1.0
    assert np.isclose(scene._peak_marker_world_radius, 0.35)
    assert scene._peak_marker_actor.GetVisibility() == 1


def _attach_projection_window(scene, *, width=800, height=600):
    window = vtk.vtkRenderWindow()
    window.SetSize(int(width), int(height))
    window.AddRenderer(scene.renderer)
    scene.render_window = window
    return window


def test_peak_marker_size_is_real_screen_pixels_in_parallel_projection():
    scene = _marker_scene()
    _attach_projection_window(scene)
    camera = scene.renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(10.0)
    camera.SetPosition(0.0, 0.0, -10.0)
    camera.SetFocalPoint(0.0, 0.0, 0.0)

    catalog = pd.DataFrame({
        "peak_id": [1, 2], "x": [1, 2], "y": [1, 1], "theta": [1.0, 2.0],
    })
    scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        point_size=10.0, render=False,
    )
    expected_radius = 5.0 * (2.0 * 10.0) / 600.0
    assert np.isclose(scene._peak_marker_world_radius, expected_radius)
    assert np.allclose(scene._peak_marker_scale_values, 1.0)

    # Doubling the parallel viewing height must double world radius so the
    # rendered sphere diameter remains exactly 10 pixels.
    camera.SetParallelScale(20.0)
    scene._on_peak_marker_view_changed()
    assert np.isclose(scene._peak_marker_world_radius, 2.0 * expected_radius)
    assert scene._peak_marker_size == 10.0


def test_perspective_marker_radii_compensate_for_each_event_depth():
    scene = _marker_scene()
    _attach_projection_window(scene)
    camera = scene.renderer.GetActiveCamera()
    camera.ParallelProjectionOff()
    camera.SetViewAngle(30.0)
    camera.SetPosition(0.0, 0.0, -10.0)
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    scene._physical_theta = np.arange(11, dtype=np.float64)

    catalog = pd.DataFrame({
        "peak_id": [1, 2], "x": [1, 1], "y": [1, 1], "theta": [0.0, 10.0],
    })
    scene.set_peak_markers(
        catalog, np.arange(4.0), np.arange(3.0), visible=True,
        point_size=10.0, render=False,
    )
    low_radius, high_radius = scene._peak_marker_world_radius_range
    assert high_radius > low_radius > 0.0
    assert np.isclose(high_radius / low_radius, 2.0, rtol=1e-6)
    assert np.isclose(float(np.median(scene._peak_marker_world_radius_range)), scene._peak_marker_world_radius)
    scales = scene._peak_marker_polydata.GetPointData().GetArray("PeakMarkerGlyphScale")
    assert scales is not None and scales.GetNumberOfTuples() == 2

    old_median = scene._peak_marker_world_radius
    camera.SetViewAngle(45.0)
    scene._on_peak_marker_view_changed()
    assert scene._peak_marker_world_radius > old_median
