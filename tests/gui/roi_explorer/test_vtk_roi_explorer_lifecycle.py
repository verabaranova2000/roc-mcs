from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import VTKVolumeScene

ROOT = Path(__file__).resolve().parents[3]
# Поддерживаем оба layout исходников: src/roc_mcs/gui и src/gui.
_SRC_CANDIDATES = (ROOT / "src" / "roc_mcs", ROOT / "src")
SRC = next((p for p in _SRC_CANDIDATES if (p / "gui" / "roi_explorer").is_dir()), None)
if SRC is None:
    expected = ", ".join(str(p / "gui" / "roi_explorer") for p in _SRC_CANDIDATES)
    raise RuntimeError(f"Не найдены исходники ROI Explorer. Проверены: {expected}")


def _method(path: Path, cls: str, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    return next(node for node in klass.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name), tree


def _calls(node):
    values = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            fn = child.func
            values.append(fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else "")
    return values


def test_workspace_starts_with_map_and_curves_and_volume_is_explicit():
    path = SRC / "gui/roi_explorer/plot_workspace_widget.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    visible = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == "DEFAULT_VISIBLE_PANELS" for t in n.targets))
    assert ast.literal_eval(visible.value) == ("map", "curve")
    assert "EmbeddedVTKVolumeWidget" in text
    assert "initial_visible=False" in text
    assert "on_close=self.volume_widget.release_vtk_scene" in text


def test_vtk_imports_are_deferred_until_first_visible_3d_request():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("vtkmodules" in ast.dump(n) or "import vtk" in ast.dump(n) for n in module_imports)
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "_load_vtk_modules")
    assert any(isinstance(n, ast.Import) and any(a.name.startswith("vtkmodules.") for a in n.names) for n in ast.walk(method))


def test_release_keeps_lightweight_hud_but_finalizes_qvtk_and_invalidates_workers():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "release_vtk_scene")
    calls = _calls(method)
    assert "close" in calls
    assert "deleteLater" in calls
    assert "set_scene_available" in calls
    assert "scene" in ast.unparse(method)
    source = ast.unparse(method)
    assert "self.display = None" not in source
    assert "self._generation += 1" in source
    assert "self._pending_request = None" in source


def test_vtk_scene_releases_graphics_before_detaching_props_and_renderer():
    events = []

    class Window:
        def MakeCurrent(self): events.append("make-current")
        def RemoveRenderer(self, renderer): events.append("remove-renderer")

    class Renderer:
        def __init__(self, name): self.name = name
        def ReleaseGraphicsResources(self, window): events.append(f"{self.name}-release")
        def RemoveAllViewProps(self): events.append(f"{self.name}-remove-props")

    class Interactor:
        def RemoveObserver(self, tag): events.append("remove-observer")
        def SetInteractorStyle(self, style): events.append("detach-style")

    class Style:
        def SetDefaultRenderer(self, renderer): events.append("detach-default-renderer")

    class Axes:
        def __init__(self, renderer): self.renderer = renderer
        def GetRenderer(self): return self.renderer
        def SetEnabled(self, value): events.append("axes-disable")
        def SetInteractor(self, value): events.append("axes-detach-interactor")
        def SetOrientationMarker(self, value): events.append("axes-detach-marker")
        def SetRenderer(self, value): events.append("axes-detach-renderer")

    scene = VTKVolumeScene.__new__(VTKVolumeScene)
    scene.__dict__.update({
        "_closed": False, "render_window": Window(), "renderer": Renderer("main"),
        "interactor": Interactor(), "interactor_style": Style(), "axes_widget": Axes(Renderer("marker")),
        "_camera_observer_id": 7, "_render_callback": lambda: None,
        "payload": object(), "grid": object(), "mapper": object(), "volume": object(),
        "volume_property": object(), "projection_grid": None, "projection_mapper": None,
        "projection_actor": None, "projection_lut": None, "outline_filter": None,
        "outline_mapper": None, "outline_actor": None, "axis_tick_polydata": None,
        "axis_tick_mapper": None, "axis_tick_actor": None, "grid_polydata": None,
        "grid_mapper": None, "grid_actor": None, "_axis_label_actors": [],
        "_axis_label_title_flags": [],
    })
    scene.close()
    assert events.index("make-current") < events.index("marker-release") < events.index("axes-disable")
    assert events.index("main-release") < events.index("main-remove-props") < events.index("remove-renderer")
    assert "detach-default-renderer" in events and "detach-style" in events
    assert scene.render_window is None and scene.renderer is None and scene.interactor is None
    count = len(events)
    scene.close()
    assert len(events) == count


def test_explorer_only_refreshes_vtk_when_panel_is_visible_and_uses_selected_view():
    path = SRC / "gui/roi_explorer/explorer_window.py"
    method, _ = _method(path, "RockingCurveExplorerWindow", "refresh_volume")
    source = ast.unparse(method)
    assert "is_panel_visible" in source
    assert "state.data_yxt" in source
    assert "state.valid_mask_yxt" in source
    assert "self.repo.load_scan" in source
    assert "H5VolumeLoader" not in source


def test_minimize_restore_refresh_signal_is_separate_from_close_release():
    path = SRC / "gui/roi_explorer/plot_workspace_widget.py"
    text = path.read_text(encoding="utf-8")
    assert "windowStateChanged" in text
    assert "panel_restored" in text
    assert "panel_restored.emit" in text
    subwindow, _ = _method(path, "_PlotSubWindow", "closeEvent")
    assert "_on_explicit_close" in ast.unparse(subwindow)
    # Minimize handling must not call the close/release callback.
    restore, _ = _method(path, "_PlotMdiArea", "_on_window_state_changed")
    assert "panel_restored" in ast.unparse(restore)
    assert "release_vtk_scene" not in ast.unparse(restore)


def test_persistent_hud_is_child_of_qvtk_and_rebound_across_scene_lifecycle():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "_initialize_vtk")
    source = ast.unparse(method)
    assert "VTKVolumeDisplayWidget(widget)" in source
    assert "self.display.set_viewport(widget)" in source
    release, _ = _method(path, "EmbeddedVTKVolumeWidget", "release_vtk_scene")
    assert "self.display.set_viewport(self.viewport)" in ast.unparse(release)


def test_map_cursor_and_roi_preview_are_wired_to_embedded_vtk_widget():
    path = SRC / "gui/roi_explorer/explorer_window.py"
    method, _ = _method(path, "RockingCurveExplorerWindow", "_connect_signals")
    source = ast.unparse(method)
    assert "cursor_position_changed.connect(self.volume_widget.set_map_cursor_position)" in source
    assert "preview_roi_changed.connect(self.volume_widget.set_roi_preview_geometry)" in source
    assert "preview_roi_finished.connect(self.volume_widget.set_roi_preview_geometry)" in source


def test_deferred_vtk_initialization_uses_generation_tickets_to_reject_stale_timers():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    schedule, _ = _method(path, "EmbeddedVTKVolumeWidget", "_schedule_vtk_init")
    ensure, _ = _method(path, "EmbeddedVTKVolumeWidget", "_ensure_vtk")
    initialize, _ = _method(path, "EmbeddedVTKVolumeWidget", "_initialize_vtk")
    release, _ = _method(path, "EmbeddedVTKVolumeWidget", "release_vtk_scene")
    assert "_init_generation += 1" in ast.unparse(schedule)
    assert "ticket != self._init_generation" in ast.unparse(ensure)
    assert "ticket != self._init_generation" in ast.unparse(initialize)
    assert "_init_generation += 1" in ast.unparse(release)


def test_new_payload_request_clears_previous_volume_before_background_build():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "_submit_payload")
    source = ast.unparse(method)
    assert "self._payload = None" in source
    assert "self._scene.clear(render=False)" in source
    assert "self.tools.set_scene_available(False)" in source


def test_restore_refreshes_only_when_data_changed_or_scene_was_closed():
    path = SRC / "gui/roi_explorer/explorer_window.py"
    shown, _ = _method(path, "RockingCurveExplorerWindow", "_on_plot_panel_visibility_changed")
    restored, _ = _method(path, "RockingCurveExplorerWindow", "_on_plot_panel_restored")
    refresh, _ = _method(path, "RockingCurveExplorerWindow", "refresh_volume")
    assert "_volume_refresh_pending" in ast.unparse(shown)
    assert "has_dataset_request" in ast.unparse(shown)
    assert "_volume_refresh_pending" in ast.unparse(restored)
    assert "has_dataset_request" in ast.unparse(restored)
    assert "self._volume_refresh_pending = True" in ast.unparse(refresh)
    assert "self._volume_refresh_pending = False" in ast.unparse(refresh)


def test_scan_pooling_and_workbench_switches_mark_vtk_data_dirty():
    path = SRC / "gui/roi_explorer/explorer_window.py"
    for method_name in ("_on_scan_changed", "_on_pool_changed", "show_analysis_preview", "refresh_all"):
        method, _ = _method(path, "RockingCurveExplorerWindow", method_name)
        assert "_request_volume_refresh" in ast.unparse(method), method_name
    request, _ = _method(path, "RockingCurveExplorerWindow", "_request_volume_refresh")
    source = ast.unparse(request)
    assert "self._volume_refresh_pending = True" in source
    assert "is_panel_visible" in source
    assert "self._volume_refresh_timer.start" in source


def test_display_smoothing_requests_refresh_through_explorer_not_local_hidden_worker():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "apply_display_settings")
    source = ast.unparse(method)
    assert "data_rebuild_requested.emit()" in source
    assert "self._submit_payload(request)" not in source
    explorer_path = SRC / "gui/roi_explorer/explorer_window.py"
    connect, _ = _method(explorer_path, "RockingCurveExplorerWindow", "_connect_signals")
    assert "data_rebuild_requested.connect(self._request_volume_refresh)" in ast.unparse(connect)


def test_failed_vtk_payload_application_clears_partial_scene():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "_payload_ready")
    source = ast.unparse(method)
    assert "self._scene.clear(render=False)" in source
    assert "self.tools.set_scene_available(False)" in source
    assert "self.display.set_scene_available(False)" in source
    assert "self._payload = None" in source
