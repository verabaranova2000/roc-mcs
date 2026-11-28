from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "src" / "roc_mcs" / "gui"
WORKBENCH = GUI / "processing_workbench" / "workbench_widget.py"
GRAPH = GUI / "processing_workbench" / "graph_canvas.py"
EXPLORER = GUI / "roi_explorer" / "explorer_window.py"
THEME = GUI / "roi_explorer" / "theme.py"
ICONS = GUI / "resources" / "icons"


def _method(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_pipeline_header_buttons_share_vtk_global_icon_geometry_policy():
    workbench = WORKBENCH.read_text(encoding="utf-8")
    vtk = (GUI / "roi_explorer" / "vtk_backend" / "vtk_tools_widget.py").read_text(encoding="utf-8")
    explorer = EXPLORER.read_text(encoding="utf-8")
    helper = _method(
        workbench,
        "    def _apply_graph_header_button_metrics(",
        "    def apply_global_icon_size(",
    )
    vtk_helper = _method(vtk, "    def _set_icon_sizes(self) -> None:", "    def _color_swatch_icon")

    for token in (
        "button_width = max(22, min(30, icon + 10))",
        "button_height = max(18, min(26, icon + 6))",
        "button.setFixedSize(button_width, button_height)",
    ):
        assert token in helper
        assert token in vtk_helper
    assert 'self.setObjectName("processingWorkbenchWidget")' in workbench
    assert "self.fit_graph_button" in workbench
    assert "self.graph_scroll_button" in workbench
    assert "button.setIconSize(QSize(icon, icon))" in helper
    assert "size = QSize(self._icon_size(), self._icon_size())" in vtk_helper
    assert "button.setIconSize(size)" in vtk_helper
    assert "apply_global_icon_size" in explorer
    assert 'is_vtk_widget = widget.objectName() == "EmbeddedVTKVolumeWidget"' in explorer
    assert 'is_workbench_widget = widget.objectName() == "processingWorkbenchWidget"' in explorer
    assert "if callable(apply_icon_size):" in explorer


def test_spatial_pooling_form_is_compact_icon_only_and_restores_canonical_layout():
    source = WORKBENCH.read_text(encoding="utf-8")
    pool_form = _method(source, "    def _build_pool_form(self) -> None:", "    def _build_spike_form")
    popup = _method(source, "    def _open_stage_configuration(", "    def _run_selected")

    assert "QGridLayout()" in pool_form
    assert "control.setFixedSize(48, 21)" in pool_form
    assert "preview.setFixedSize(82, 62)" in pool_form
    assert 'content.setObjectName("compactSpatialPoolingForm")' in pool_form
    assert "if is_pooling_stage:" in popup
    assert 'ICON_DIR / "run_icon.svg"' in popup
    assert 'run_button.setToolTip("Создать и запустить пространственный pooling")' in popup
    assert 'run_button.setAccessibleName("Создать и запустить пространственный pooling")' in popup
    assert "self._apply_graph_header_button_metrics(run_button)" in popup
    assert "elif isinstance(field, QSpinBox) and not is_pooling_stage:" in popup
    assert "compact_pool_form = is_pooling_stage" in popup
    assert "page_layout.setVerticalSpacing(0 if compact_pool_form" in popup
    assert "if (is_hkl_stage or is_pooling_stage) and page_layout is not None" in popup
    assert "page_layout.setHorizontalSpacing(horizontal)" in popup


def test_tree_wheel_is_captured_at_inner_scroll_limits_and_bubbles_only_without_range():
    source = GRAPH.read_text(encoding="utf-8")
    start = source.index("    def wheelEvent(self, event) -> None:")
    wheel = source[start:]
    assert "if self._vertical_scroll_enabled:" in wheel
    assert "has_internal_range = bar.maximum() > bar.minimum()" in wheel
    assert "if has_internal_range:" in wheel
    assert "bar.setValue(int(bar.value()) - scroll_delta)" in wheel
    assert "event.accept()" in wheel
    assert wheel.rstrip().endswith("event.ignore()")
    assert "self.setObjectName(\"processingGraphView\")" in source


def test_scrollbars_are_scoped_theme_aware_and_have_documented_style_variants():
    source = THEME.read_text(encoding="utf-8")
    for variant in ('"instrument"', '"minimal"', '"classic"'):
        assert variant in source
    assert 'if variant_name == "native":' in source
    assert "QScrollArea#leftSidebarScroll QScrollBar:vertical" in source
    assert "QGraphicsView#processingGraphView QScrollBar:vertical" in source
    assert "background:{idle}" in source
    assert "background:{hover}" in source
    assert "background:{pressed}" in source
    # The stylesheet receives the active theme's resolved palette, not fixed colors.
    assert "_workflow_scrollbar_stylesheet(c)" in source
    assert '"light": {' in source and '"dark": {' in source


def test_hkl_information_is_visible_and_close_control_has_no_external_asset_dependency():
    source = WORKBENCH.read_text(encoding="utf-8")
    panel_source = (GUI / "roi_explorer" / "hkl_results_panel.py").read_text(encoding="utf-8")
    hkl = _method(source, "    def _build_hkl_identification_form(self) -> None:", "    def _request_branch_stats")
    popup = _method(source, "    def _open_stage_configuration(", "    def _run_selected")
    info_svg = ICONS / "info_circle_icon.svg"

    assert "information_text = (" in hkl
    assert "ЧТЕНИЕ ГРАФИКА И ТАБЛИЦЫ" in hkl
    assert "layout.addRow(explanation)" not in hkl
    assert 'info_button.setObjectName("processingStageInfoButton")' in popup
    assert 'ICON_DIR / "info_circle_icon.svg"' in popup
    assert "SP_MessageBoxInformation" in popup
    assert "QMessageBox(popup)" in popup
    assert "help_dialog.exec()" in popup
    assert 'info_button.setFixedSize(24, 24)' in popup
    assert "title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)" in popup
    assert 'close_button.style().standardIcon(' in popup
    assert "QStyle.StandardPixmap.SP_TitleBarCloseButton" in popup
    assert 'close_button.setIconSize(QSize(13, 13))' in popup
    assert '"close_icon.svg"' not in popup
    assert info_svg.is_file()
    ET.parse(info_svg)

    # The long explanatory paragraph is no longer permanently appended to
    # the result panel; all of its content is available from the header Info.
    assert "self.method_note" not in panel_source
    assert "hklMethodNote" not in panel_source
    assert "QLabel#hklControlLabel" in panel_source
