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


def test_persistent_hud_is_a_layout_managed_native_overlay_sibling_of_qvtk():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    text = path.read_text(encoding="utf-8")
    assert "self.viewport_layout = QtWidgets.QGridLayout(self.viewport)" in text
    assert "self.viewport_layout.addWidget(widget, 0, 0)" in text
    assert "self.display, 0, 0" in text
    assert "AlignTop | QtCore.Qt.AlignmentFlag.AlignRight" in text
    display_path = SRC / "gui/roi_explorer/vtk_backend/vtk_volume_display_widget.py"
    display_text = display_path.read_text(encoding="utf-8")
    assert "WA_NativeWindow" in display_text
    assert "WA_AlwaysStackOnTop" in display_text
    assert "self.setFixedSize(86, content_height)" in display_text
    assert "def restack_over_viewport" in display_text
    position, _ = _method(display_path, "VTKVolumeDisplayWidget", "_position_overlay")
    assert "layout.setAlignment" in ast.unparse(position)


def test_map_cursor_and_roi_preview_are_wired_to_embedded_vtk_widget():
    path = SRC / "gui/roi_explorer/explorer_window.py"
    method, _ = _method(path, "RockingCurveExplorerWindow", "_connect_signals")
    source = ast.unparse(method)
    assert "cursor_position_changed.connect(self.volume_widget.set_map_cursor_position)" in source
    assert "preview_roi_changed.connect(self.volume_widget.set_roi_preview_geometry)" in source
    assert "preview_roi_finished.connect(self.volume_widget.set_roi_preview_geometry)" in source


def test_embedded_theta_slider_is_bidirectionally_linked_by_physical_theta_value():
    embedded_path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    explorer_path = SRC / "gui/roi_explorer/explorer_window.py"
    embedded = embedded_path.read_text(encoding="utf-8")
    explorer = explorer_path.read_text(encoding="utf-8")
    assert "theta_slice_value_changed = QtCore.pyqtSignal(float)" in embedded
    assert "Связать с 2D-картой" in embedded
    assert "def set_map_theta_context" in embedded
    assert "def set_theta_index_from_map" in embedded
    assert "np.sort(volume_axis), np.sort(map_axis)" in embedded
    assert "self.volume_widget.theta_slice_value_changed.connect(self._on_vtk_theta_slice_changed)" in explorer
    assert "self.volume_widget.set_theta_index_from_map(float(axis[self._theta_index]))" in explorer
    assert "self.volume_widget.set_map_theta_index(self._theta_index)" in explorer
    assert "self.volume_widget.set_map_theta_index(index)" in explorer
    assert "self.map_widget.set_theta_index(index, emit=False)" in explorer
    assert "def set_map_theta_index" in embedded


def test_embedded_source_selection_is_an_icon_menu_and_vtk_colorbar_stays_pinned():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    source = path.read_text(encoding="utf-8")
    assert "vtkExplorerSourceButton" in source
    assert "self.source_button.setMenu(self.source_menu)" in source
    assert "self.source_combo.hide()" in source
    assert 'QtWidgets.QLabel("Источник"' not in source
    assert "VTKVolumeDisplayWidget(self.viewport)" in source
    assert "_SourceDropdownButton(self.toolbar)" in source
    apply_payload, _ = _method(path, "EmbeddedVTKVolumeWidget", "_apply_payload")
    assert "self.display.set_scene_available(True)" in ast.unparse(apply_payload)


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


def test_rectangle_roi_aliases_use_shared_model_normalizer_without_cross_importing_vtk():
    model = (SRC / "roi/roi_types.py").read_text(encoding="utf-8")
    embedded = (SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py").read_text(encoding="utf-8")
    plotly = (SRC / "gui/roi_explorer/volume_plot_widget.py").read_text(encoding="utf-8")
    adapter = (SRC / "gui/roi_explorer/vtk_backend/scan_view_adapter.py").read_text(encoding="utf-8")
    assert "def normalize_rectangular_roi" in model
    assert "normalize_rectangular_roi(item.roi)" in embedded
    assert "normalize_rectangular_roi(saved.roi)" in plotly
    assert "return normalize_rectangular_roi(saved.roi)" in plotly
    assert "from roc_mcs.gui.roi_explorer.vtk_backend" not in plotly
    assert "from roc_mcs.roi.roi_types import normalize_rectangular_roi" in adapter


def test_saving_roi_refreshes_only_source_catalogue_not_active_volume():
    explorer_path = SRC / "gui/roi_explorer/explorer_window.py"
    embedded_path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    pin, _ = _method(explorer_path, "RockingCurveExplorerWindow", "pin_preview_roi")
    pin_source = ast.unparse(pin)
    assert "update_saved_rois" in pin_source
    assert "refresh_volume" not in pin_source

    refresh, _ = _method(embedded_path, "EmbeddedVTKVolumeWidget", "update_saved_rois")
    refresh_source = ast.unparse(refresh)
    assert "_rebuild_source_choices" in refresh_source
    assert "_submit_payload" in refresh_source  # only for removal of the active ROI
    assert "active_roi_removed" in refresh_source
    assert "set_dataset" not in refresh_source


def test_native_volume_hud_has_explicit_native_window_and_restacking_contract():
    display_path = SRC / "gui/roi_explorer/vtk_backend/vtk_volume_display_widget.py"
    display_text = display_path.read_text(encoding="utf-8")
    assert "WA_NativeWindow" in display_text
    assert "self.winId()" in display_text
    sync, _ = _method(display_path, "VTKVolumeDisplayWidget", "_sync_visibility")
    assert "raise_" in ast.unparse(sync)

    embedded_path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    render, _ = _method(embedded_path, "EmbeddedVTKVolumeWidget", "_render")
    assert "self._qvtk.Render()" in ast.unparse(render)
    assert "restack_over_viewport" not in ast.unparse(render)
    loader, _ = _method(embedded_path, "EmbeddedVTKVolumeWidget", "_load_vtk_modules")
    tracked = next(
        node for node in ast.walk(loader)
        if isinstance(node, ast.ClassDef) and node.name == "_TrackedQVTK"
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "paintEvent"
        for node in tracked.body
    )


def test_source_toolbar_button_is_smart_dropdown_and_nested_gap_is_removed():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    widget = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EmbeddedVTKVolumeWidget")
    init = next(n for n in widget.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignments = [n for n in ast.walk(init) if isinstance(n, ast.Assign)]
    source_button = next(
        n for n in assignments
        if any(isinstance(t, ast.Attribute) and t.attr == "source_button" for t in n.targets)
    )
    assert isinstance(source_button.value, ast.Call)
    assert isinstance(source_button.value.func, ast.Name)
    assert source_button.value.func.id == "_SourceDropdownButton"
    text = path.read_text(encoding="utf-8")
    assert "tools_layout.setContentsMargins(0, 2, 5, 2)" in text
    assert "row.setSpacing(0)" in text


def test_theta_link_is_compact_and_disabled_as_a_whole_row_when_unavailable():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    text = path.read_text(encoding="utf-8")
    assert "vtkThetaSliceLinkButton" in text
    assert "vtk_theta_link_" in text
    assert "self.theta_link_row.setEnabled(available)" in text
    assert "self.theta_link_label.setToolTip(tip)" in text
    assert "self._theta_link_preference" in text
    for variant in (1, 2, 3):
        assert (SRC / "gui" / "resources" / "icons" / f"vtk_theta_link_{variant}.svg").is_file()


def test_optional_theta_slice_marker_is_abstract_geometry_not_a_scalar_surface():
    scene_path = SRC / "gui/roi_explorer/vtk_backend/vtk_scene.py"
    text = scene_path.read_text(encoding="utf-8")
    assert "self._theta_slice_marker_source = vtk.vtkPlaneSource()" in text
    assert "self.renderer.AddActor(self._theta_slice_marker_actor)" in text
    assert "def set_theta_slice_marker_visible" in text
    assert "def set_theta_slice_marker_opacity" in text
    setter, _ = _method(scene_path, "VTKVolumeScene", "set_theta_slice_index")
    source = ast.unparse(setter)
    assert "_update_theta_slice_marker_geometry" in source
    assert "_update_theta_slice_shader_uniforms" in source
    assert "vtkGPUVolumeRayCastMapper" not in source
    assert "_theta_slice_plane_grid" not in text


def test_vtk_colorbar_defaults_visible_is_anchored_and_retains_settings_control():
    display_path = SRC / "gui/roi_explorer/vtk_backend/vtk_volume_display_widget.py"
    text = display_path.read_text(encoding="utf-8")
    assert "self._show_colorbar = True" in text
    assert "self.setFixedWidth(86)" in text
    assert "self.setFixedSize(86, content_height)" in text
    assert "WA_AlwaysStackOnTop" in text
    assert "margin = 12" in text
    assert "self.options_button.clicked.connect(self._open_settings_menu)" in text
    visible, _ = _method(display_path, "VTKVolumeDisplayWidget", "_sync_visibility")
    assert "self._scene_available and self._show_colorbar" in ast.unparse(visible)

    embedded_path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    apply_payload, _ = _method(embedded_path, "EmbeddedVTKVolumeWidget", "_apply_payload")
    assert "self.display.set_scene_available(True)" in ast.unparse(apply_payload)
    settings, _ = _method(embedded_path, "EmbeddedVTKVolumeWidget", "apply_display_settings")
    assert "show_vtk_colorbar" in ast.unparse(settings) and "True" in ast.unparse(settings)


def test_vtk_colorbar_has_independent_true_default_from_legacy_plotly_flag():
    settings_path = SRC / "gui/roi_explorer/plot_display_settings.py"
    settings_text = settings_path.read_text(encoding="utf-8")
    assert '"show_vtk_colorbar":True' in settings_text
    assert '"show_vtk_colorbar",True' in settings_text
    assert '"show_vtk_colorbar", "Компактная шкала в окне VTK"' in settings_text
    embedded_path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    embedded_text = embedded_path.read_text(encoding="utf-8")
    assert 'get("show_vtk_colorbar", True)' in embedded_text
    assert 'get("show_colorbar", True)' not in embedded_text


def test_theta_link_icon_is_recolored_to_muted_theme_when_parent_row_is_disabled():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    method, _ = _method(path, "EmbeddedVTKVolumeWidget", "_update_theta_link_icon")
    source = ast.unparse(method)
    assert "muted" in source
    assert 'self.theta_link_row.isEnabled()' in source
    assert "self.theta_link_checkbox.setIcon" in source


def test_source_dropdown_has_only_custom_corner_triangle_and_compact_menu_typography():
    path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    text = path.read_text(encoding="utf-8")
    assert "class _SourceDropdownButton(SmartDropdownButton)" in text
    button_paint, _ = _method(path, "_SourceDropdownButton", "paintEvent")
    paint_source = ast.unparse(button_paint)
    assert "ToolButtonFeature.HasMenu" in paint_source
    assert "features" in paint_source and "drawComplexControl" in paint_source
    assert "super().paintEvent" not in paint_source
    assert "QToolButton::menu-indicator" in text
    assert "QMenu#vtkExplorerSourceMenu" in text
    assert "font-size: 7pt" in text
    assert "setPointSizeF(8.0)" in text
    assert "min-height: 16px" in text


def test_theta_slice_marker_is_abstract_geometry_and_does_not_replace_shader_highlight():
    scene_path = SRC / "gui/roi_explorer/vtk_backend/vtk_scene.py"
    scene_text = scene_path.read_text(encoding="utf-8")
    assert "vtkPlaneSource()" in scene_text
    assert "_theta_slice_marker_actor" in scene_text
    assert "def set_theta_slice_marker_visible" in scene_text
    assert "vtkRectilinearGrid()" not in scene_text[scene_text.index("# Optional abstract marker"):scene_text.index("# The map cursor")]
    assert "_update_theta_slice_shader_uniforms()" in scene_text
    assert "_update_theta_slice_marker_geometry()" in scene_text
    assert "_theta_slice_plane_grid" not in scene_text

    embedded_path = SRC / "gui/roi_explorer/vtk_backend/embedded_widget.py"
    embedded_text = embedded_path.read_text(encoding="utf-8")
    assert "Плоскость среза θ" in embedded_text
    assert "set_theta_slice_marker_visible" in embedded_text
    assert "set_theta_slice_marker_opacity" in embedded_text
    assert "theta_slice_plane_checkbox" not in embedded_text
