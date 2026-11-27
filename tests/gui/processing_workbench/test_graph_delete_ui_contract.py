from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "src" / "roc_mcs" / "gui"


def test_graph_card_exposes_compact_delete_action_and_canvas_signal():
    source = (ROOT / "processing_workbench" / "graph_canvas.py").read_text(encoding="utf-8")
    assert "delete_requested = pyqtSignal(str)" in source
    assert "def _delete_rect(self) -> QRectF" in source
    assert "remove_color = QColor(\"#D84F5B\")" in source
    assert "self.delete_requested.emit(self.node.id)" in source
    assert "item.delete_requested.connect(self.delete_requested)" in source
    assert "can_delete=(" in source
    assert 'node.operation_id != "source.scan"' in source


def test_workbench_removes_archive_control_and_routes_delete_through_project_transaction():
    source = (ROOT / "processing_workbench" / "workbench_widget.py").read_text(encoding="utf-8")
    assert "show_rejected_check" not in source
    assert "def _on_graph_delete_requested(self, node_id: str) -> None:" in source
    assert "self.project.remove_subgraph(str(node_id))" in source
    assert "self._active_branch_size_task is not None" in source
    assert "Удалить этап" in source
    # The rejected review status is retained as provenance, not exposed via an archive view.
    assert "show_rejected=False" in source
