from __future__ import annotations

import numpy as np
import pandas as pd

from roc_mcs.gui.processing_workbench.branch_preview_data import (
    build_branch_preview_graph,
    select_preview_patch,
)


def _two_sheet_catalog() -> pd.DataFrame:
    # Two smooth spatial-angular sheets.  The second sheet has a 40″ jump
    # between detector columns x=1 and x=2, deliberately exposing theta_gate.
    rows = []
    peak_id = 0
    for y in range(2):
        for x in range(3):
            for base, jump in ((25.0, 0.0), (140.0, 40.0)):
                theta = base + 2.0 * x + 1.5 * y + (jump if base > 100 and x >= 2 else 0.0)
                rows.append((peak_id, x, y, theta))
                peak_id += 1
    return pd.DataFrame(rows, columns=("peak_id", "x", "y", "theta"))


def test_preview_graph_uses_production_matcher_and_theta_gate_changes_components():
    catalog = _two_sheet_catalog()
    narrow = build_branch_preview_graph(catalog, theta_gate=5.0)
    wide = build_branch_preview_graph(catalog, theta_gate=45.0)

    assert len(narrow["candidate_edges"]) > len(narrow["accepted_edges"])
    assert len(wide["accepted_edges"]) > len(narrow["accepted_edges"])
    assert wide["n_branches"] < narrow["n_branches"]
    assert any(not accepted for *_edge, accepted in narrow["candidate_edges"])
    assert all(accepted for *_edge, accepted in wide["candidate_edges"])


def test_preview_patch_is_contiguous_compact_and_does_not_modify_source():
    rows = []
    peak_id = 0
    for y in range(5):
        for x in range(7):
            n_peaks = 1 + ((x + 2 * y) % 5)
            for index in range(n_peaks):
                rows.append((peak_id, x, y, float(20 * index + x + y)))
                peak_id += 1
    catalog = pd.DataFrame(rows, columns=("peak_id", "x", "y", "theta"))
    original = catalog.copy(deep=True)

    preview, info = select_preview_patch(catalog)

    assert info["available"] is True
    assert len(preview) <= 48
    assert info["occupied_pixels"] <= info["width"] * info["height"] <= 4 * 3
    assert preview["x"].max() - preview["x"].min() < info["width"]
    assert preview["y"].max() - preview["y"].min() < info["height"]
    assert info["all_events_preserved"] is True
    assert info["truncated"] is False
    # Every event in every displayed detector pixel is kept: thinning would
    # change the exact one-to-one assignment shown by the preview.
    chosen = catalog[
        catalog["x"].between(info["x_min"], info["x_max"])
        & catalog["y"].between(info["y_min"], info["y_max"])
    ]
    assert len(preview) == len(chosen)
    pd.testing.assert_frame_equal(catalog, original)


def test_preview_patch_reports_missing_catalog_without_fabricating_events():
    preview, info = select_preview_patch(pd.DataFrame())
    assert preview.empty
    assert info["available"] is False
    assert "пуст" in info["message"].lower()


def test_prepared_preview_reuses_hungarian_candidates_across_gate_updates(monkeypatch):
    import roc_mcs.peaks.branches as branch_algorithms
    from roc_mcs.gui.processing_workbench.branch_preview_data import prepare_branch_preview_graph

    catalog = _two_sheet_catalog()
    original_builder = branch_algorithms.build_peak_branch_candidate_graph
    calls = 0

    def counted_builder(frame):
        nonlocal calls
        calls += 1
        return original_builder(frame)

    monkeypatch.setattr(branch_algorithms, "build_peak_branch_candidate_graph", counted_builder)
    prepared = prepare_branch_preview_graph(catalog)
    narrow = build_branch_preview_graph(catalog, theta_gate=5.0, prepared=prepared)
    wide = build_branch_preview_graph(catalog, theta_gate=45.0, prepared=prepared)
    assert calls == 1

    # Cached and standalone materialization must display the same exact graph.
    reference = build_branch_preview_graph(catalog, theta_gate=45.0)
    assert wide["accepted_edges"] == reference["accepted_edges"]
    assert wide["candidate_edges"] == reference["candidate_edges"]
    assert wide["branch_by_peak_id"] == reference["branch_by_peak_id"]
    assert narrow["n_branches"] > wide["n_branches"]
