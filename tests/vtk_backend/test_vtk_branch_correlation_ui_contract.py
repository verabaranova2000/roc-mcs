from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src" / "roc_mcs"


def test_correlation_layer_is_registered_and_has_an_independent_settings_popup():
    tools = (ROOT / "gui/roi_explorer/vtk_backend/vtk_tools_widget.py").read_text(encoding="utf-8")
    embedded = (ROOT / "gui/roi_explorer/vtk_backend/embedded_widget.py").read_text(encoding="utf-8")
    assert '"correspondences"' in tools
    assert "vtk_branch_correlation.svg" in tools
    assert "correspondences: QWidget | None = None" in tools
    assert "set_branch_correspondences_visible" in embedded
    assert "correlation_pair_combo" in embedded
    assert "correlation_show_candidates" in embedded


def test_workflow_preview_is_passed_as_a_view_model_not_worker_payload():
    explorer = (ROOT / "gui/roi_explorer/explorer_window.py").read_text(encoding="utf-8")
    embedded = (ROOT / "gui/roi_explorer/vtk_backend/embedded_widget.py").read_text(encoding="utf-8")
    assert "branch_correlation_preview=getattr(" in explorer
    signature = embedded[embedded.index("def set_dataset("):embedded.index("def update_saved_rois(")]
    assert "branch_correlation_preview=None" in signature
    assert '"branch_correlation_preview"' not in embedded[embedded.index('self._dataset_request = {'):embedded.index('self.source_button.setEnabled(True)')]


def test_correlation_visual_semantics_are_spelled_out_in_controls():
    tools = (ROOT / "gui/roi_explorer/vtk_backend/vtk_tools_widget.py").read_text(encoding="utf-8")
    embedded = (ROOT / "gui/roi_explorer/vtk_backend/embedded_widget.py").read_text(encoding="utf-8")
    assert "принятые соответствия peak_id" in tools
    assert "Поддержанный альтернативный кандидат" in embedded
    assert "сдвиг партнёра → source" in embedded
    assert "Непрозрачность связей" in embedded


def test_correlation_svg_is_valid_and_present():
    icon = ROOT / "gui/resources/icons/vtk_branch_correlation.svg"
    assert icon.is_file()
    text = icon.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "</svg>" in text
