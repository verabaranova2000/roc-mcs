from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "src" / "roc_mcs" / "gui" / "processing_workbench"


def test_branch_preview_prepares_hungarian_candidates_outside_drag_path():
    preview = (ROOT / "branch_detection_preview.py").read_text(encoding="utf-8")
    data = (ROOT / "branch_preview_data.py").read_text(encoding="utf-8")
    assert "prepare_branch_preview_graph(self._catalog)" in preview
    assert "prepared=self._prepared_graph" in preview
    assert "build_peak_branch_candidate_graph(peak_catalog)" in data
    assert "match_neighbor_peaks" not in data


def test_branch_distribution_reuses_array_and_histogram_bins_are_cached():
    workbench = (ROOT / "workbench_widget.py").read_text(encoding="utf-8")
    histogram = (ROOT / "branch_size_histogram.py").read_text(encoding="utf-8")
    assert 'np.asarray(distribution.get("sizes", ()), dtype=np.int64)' in workbench
    assert "np.count_nonzero(included_mask)" in workbench
    assert "np.histogram(values, bins=edges)" in histogram
    # Histogram binning belongs to set_distribution, not paintEvent.
    paint = histogram[histogram.index("    def paintEvent") :]
    assert "np.histogram(" not in paint


def test_branch_status_is_single_line_and_fixed_height():
    workbench = (ROOT / "workbench_widget.py").read_text(encoding="utf-8")
    assert "class _ElidingStatusLabel(QLabel)" in workbench
    status_class = workbench[
        workbench.index("class _ElidingStatusLabel(QLabel)") :
        workbench.index("class _PopupDragTitle(QLabel)")
    ]
    assert "setWordWrap(False)" in status_class
    assert "setFixedHeight" in status_class
    assert "fontMetrics().elidedText" in status_class
    assert 'selection_status = _ElidingStatusLabel(' in workbench


def test_theta_gate_reference_is_visually_distinct_from_draggable_marker():
    source = (ROOT / "theta_gate_sensitivity.py").read_text(encoding="utf-8")
    reference_block = source[
        source.index("# The reference estimate is deliberately") :
        source.index("# Selected gate is the only draggable control")
    ]
    assert "Qt.PenStyle.DotLine" in reference_block
    assert 'label = "оценка"' in reference_block
    assert "drawPolygon" not in reference_block
    assert "thin" not in reference_block.lower()
