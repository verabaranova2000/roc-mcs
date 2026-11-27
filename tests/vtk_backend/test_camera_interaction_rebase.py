from __future__ import annotations

from types import SimpleNamespace

import pytest
import numpy as np

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
    VTKVolumeScene,
    _rebase_vtk_mouse_move,
    _stable_axis_corner_bits,
)


class _Interactor:
    def __init__(self, current=(420, 270), previous=(12, 19)):
        self.current = tuple(current)
        self.previous = tuple(previous)

    def GetEventPosition(self):
        return self.current

    def SetLastEventPosition(self, x, y):
        self.previous = (int(x), int(y))


class _Style:
    def __init__(self, state=1):
        self.state = int(state)
        self.stopped = 0
        self.released = 0
        self.default_renderer = None
        self.current_renderer = None

    def GetState(self):
        return self.state

    def StopState(self):
        self.stopped += 1
        self.state = 0

    def ReleaseFocus(self):
        self.released += 1

    def SetDefaultRenderer(self, renderer):
        self.default_renderer = renderer

    def SetCurrentRenderer(self, renderer):
        self.current_renderer = renderer


def test_trackball_recovery_stops_active_state_releases_focus_and_rebases_pointer():
    renderer = object()
    interactor = _Interactor(current=(777, 333), previous=(2, 4))
    style = _Style(state=2)
    scene_stub = SimpleNamespace(interactor_style=style, renderer=renderer)

    VTKVolumeScene._recover_trackball_interaction(scene_stub, interactor)

    assert style.stopped == 1
    assert style.released == 1
    assert style.default_renderer is renderer
    assert style.current_renderer is renderer
    assert interactor.current == (777, 333)
    assert interactor.previous == interactor.current


def test_trackball_recovery_is_idempotent_for_inactive_style_but_releases_focus():
    interactor = _Interactor(current=(15, 25), previous=(300, 400))
    style = _Style(state=0)
    VTKVolumeScene._recover_trackball_interaction(
        SimpleNamespace(interactor_style=style, renderer=object()), interactor
    )
    assert style.stopped == 0
    assert style.released == 1
    assert interactor.previous == (15, 25)


def test_qvtk_adapter_contract_rebases_after_coordinate_conversion_before_dispatch():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "roc_mcs" / "gui" / "roi_explorer" / "vtk_backend" / "embedded_widget.py"
    ).read_text(encoding="utf-8")
    adapter = source[source.index("class _TrackedQVTK") : source.index("return _TrackedQVTK")]
    assert "def _setEventInformation(" in adapter
    assert "super()._setEventInformation(x, y, ctrl, shift, key, repeat, keysum)" in adapter
    assert "_rebase_vtk_mouse_move(" in adapter
    event_info = adapter[
        adapter.index("def _setEventInformation") : adapter.index("def mousePressEvent")
    ]
    mouse_move = adapter[
        adapter.index("def mouseMoveEvent") : adapter.index("def mouseReleaseEvent")
    ]
    assert event_info.index("super()._setEventInformation") < event_info.index("_rebase_vtk_mouse_move(")
    assert "self._roc_mouse_move_dispatch = True" in mouse_move
    assert mouse_move.index("self._roc_mouse_move_dispatch = True") < mouse_move.index("super().mouseMoveEvent(event)")


def test_pointer_rebase_helper_updates_last_position_only_during_active_drag():
    interactor = _Interactor(current=(430, 300), previous=(7, 9))
    style = _Style(state=1)
    interactor.GetInteractorStyle = lambda: style

    current = _rebase_vtk_mouse_move(interactor, (400, 300))

    assert current == (430, 300)
    assert interactor.previous == (400, 300)

    style.state = 0
    assert _rebase_vtk_mouse_move(interactor, (430, 300)) is None


def test_real_vtk_trackball_uses_current_drag_samples_not_stale_last_event_position():
    vtk = pytest.importorskip("vtk")
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(800, 600)
    renderer = vtk.vtkRenderer()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)
    renderer.ResetCamera()
    interactor.Initialize()
    camera = renderer.GetActiveCamera()
    focal = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    preset_direction = np.asarray([1.7, -2.3, 1.6], dtype=np.float64)
    preset_direction /= np.linalg.norm(preset_direction)
    camera.SetFocalPoint(*focal.tolist())
    camera.SetPosition(*(focal + preset_direction * 10.0).tolist())
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.OrthogonalizeViewUp()
    renderer.ResetCameraClippingRange()
    before_direction = np.asarray(camera.GetPosition(), dtype=np.float64) - np.asarray(
        camera.GetFocalPoint(), dtype=np.float64
    )
    before_direction /= np.linalg.norm(before_direction)

    def set_pointer(x, y):
        interactor.SetEventInformation(x, y, 0, 0, "\0", 0, "")

    # A previous cursor location far from the next press must not influence
    # the first actual drag movement.
    set_pointer(15, 20)
    interactor.MouseMoveEvent()
    set_pointer(400, 300)
    interactor.LeftButtonPressEvent()
    assert int(style.GetState()) != 0
    previous = tuple(map(int, interactor.GetEventPosition()))

    # Simulate the stale LastEventPosition that caused the camera jump. The
    # adapter shim must restore this gesture's previous sample before dispatch.
    set_pointer(430, 300)
    interactor.SetLastEventPosition(15, 20)
    previous = _rebase_vtk_mouse_move(interactor, previous)
    interactor.MouseMoveEvent()
    first_position = tuple(float(value) for value in camera.GetPosition())
    after_first_direction = np.asarray(camera.GetPosition(), dtype=np.float64) - np.asarray(
        camera.GetFocalPoint(), dtype=np.float64
    )
    after_first_direction /= np.linalg.norm(after_first_direction)
    # A 30-pixel horizontal gesture produces the ordinary small TrackballCamera
    # rotation from the newly preset pose, not a jump from the old cursor.
    assert float(np.dot(before_direction, after_first_direction)) > 0.98

    # A second move must use (430, 300), not the preceding stale point either.
    set_pointer(440, 300)
    interactor.SetLastEventPosition(5, 5)
    previous = _rebase_vtk_mouse_move(interactor, previous)
    interactor.MouseMoveEvent()
    second_position = tuple(float(value) for value in camera.GetPosition())
    second_direction = np.asarray(camera.GetPosition(), dtype=np.float64) - np.asarray(
        camera.GetFocalPoint(), dtype=np.float64
    )
    second_direction /= np.linalg.norm(second_direction)
    assert float(np.dot(after_first_direction, second_direction)) > 0.999
    assert second_position != first_position

    set_pointer(440, 300)
    interactor.LeftButtonReleaseEvent()
    assert int(style.GetState()) == 0
    render_window.Finalize()


def test_camera_facing_axis_corner_has_a_dead_band_around_center_planes():
    ranges = ((0.0, 10.0), (-5.0, 5.0), (100.0, 140.0))
    initial = _stable_axis_corner_bits((9.0, 0.0, 120.0), ranges, None)
    assert initial == (1, 1, 1)

    # Sub-threshold crossing from the center must not flip the face.
    near_crossing = _stable_axis_corner_bits((9.0, -0.10, 120.10), ranges, initial)
    assert near_crossing == initial

    # Only a clear crossing beyond 2.5% of the axis span flips the corner.
    crossed = _stable_axis_corner_bits((9.0, -0.30, 121.10), ranges, near_crossing)
    assert crossed == (1, 0, 1)
    returned_near = _stable_axis_corner_bits((9.0, 0.10, 120.0), ranges, crossed)
    assert returned_near == crossed
    returned_far = _stable_axis_corner_bits((9.0, 0.30, 120.0), ranges, returned_near)
    assert returned_far == (1, 1, 1)
