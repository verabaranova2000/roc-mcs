from __future__ import annotations

from pathlib import Path


PANEL = (
    Path(__file__).resolve().parents[3]
    / "src" / "roc_mcs" / "gui" / "roi_explorer" / "hkl_results_panel.py"
)


def test_histogram_picking_uses_shared_selection_state_and_table_mirroring():
    source = PANEL.read_text(encoding="utf-8")
    press = source[source.index("    def _on_histogram_press") : source.index("    def _on_histogram_release")]
    assert "self._set_selected_reflection_key(reflection_key)" in press
    assert "self._clear_selected_reflection()" in press
    selection = source[
        source.index("    def _set_selected_reflection_key") :
        source.index("    def _draw_selected_reflection_overlay")
    ]
    assert "self.reflection_table.selectRow(row)" in selection
    assert "self._selected_reflection_key = key" in selection
    assert "QSignalBlocker(self.reflection_table)" in selection


def test_selected_line_is_keyed_by_reflection_not_only_visible_ranking_row():
    source = PANEL.read_text(encoding="utf-8")
    overlay = source[
        source.index("    def _draw_selected_reflection_overlay") :
        source.index("    def _build_reflection_hover_index")
    ]
    assert "self._selected_reflection_key is None" in overlay
    assert "x_value = float(self._selected_reflection_key)" in overlay
    assert "row = self._selected_reflection_row" not in overlay


def test_outside_click_dismissal_clears_highlight_but_table_row_click_does_not():
    source = PANEL.read_text(encoding="utf-8")
    event_filter = source[
        source.index("    def eventFilter") : source.index("    def _on_histogram_press")
    ]
    assert "in_selected_table_row" in event_filter
    assert "self._hide_reflection_tooltip(clear_pinned=True)" in event_filter
    assert "self._clear_selected_reflection()" in event_filter
