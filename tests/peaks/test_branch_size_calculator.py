from __future__ import annotations

import numpy as np
import pandas as pd

from roc_mcs.peaks.branches import (
    BranchSizeCalculator,
    build_peak_branch_candidate_graph,
    calculate_branch_size_distribution,
)


def test_prepared_branch_calculator_matches_reference_at_exact_gate_boundaries():
    catalog = pd.DataFrame([
        {"peak_id": 101, "x": 0, "y": 0, "theta": 10.0},
        {"peak_id": 102, "x": 0, "y": 0, "theta": 40.0},
        {"peak_id": 201, "x": 1, "y": 0, "theta": 10.5},
        {"peak_id": 202, "x": 1, "y": 0, "theta": 42.0},
        {"peak_id": 301, "x": 2, "y": 0, "theta": 11.5},
        {"peak_id": 302, "x": 2, "y": 0, "theta": 45.0},
        {"peak_id": 999, "x": 5, "y": 5, "theta": 100.0},
    ])
    candidate = build_peak_branch_candidate_graph(catalog)
    prepared = BranchSizeCalculator.prepare(catalog, candidate)

    assert prepared.edge_rows.dtype == np.int64
    assert prepared.sorted_deltas.dtype == np.float64
    assert not prepared.edge_rows.flags.writeable
    assert not prepared.sorted_deltas.flags.writeable

    gates = {1.0e-12, 0.5, 1.0, 2.0, 5.0}
    for delta in prepared.sorted_deltas:
        gates.add(float(np.nextafter(delta, -np.inf)))
        gates.add(float(delta))
        gates.add(float(np.nextafter(delta, np.inf)))

    for gate in sorted(gates):
        reference = calculate_branch_size_distribution(
            catalog, gate, candidate_graph=candidate
        )
        optimized = calculate_branch_size_distribution(
            None, gate, candidate_graph=prepared
        )
        assert sorted(optimized["sizes"]) == sorted(reference["sizes"])
        assert optimized["n_events"] == len(catalog)
        assert optimized["n_branches"] == reference["n_branches"]


def test_prepared_merge_forest_matches_reference_for_random_multigraphs():
    """The reconstruction forest must preserve exact components, including duplicates and ties."""
    from roc_mcs.peaks.branches import label_peak_branches

    rng = np.random.default_rng(20261127)
    n_events = 37
    catalog = pd.DataFrame({
        "peak_id": np.arange(500, 500 + n_events, dtype=np.int64),
        "x": np.zeros(n_events, dtype=np.int64),
        "y": np.zeros(n_events, dtype=np.int64),
        "theta": np.linspace(0.0, 100.0, n_events, dtype=np.float64),
    })
    edge_pairs = rng.integers(0, n_events, size=(95, 2), dtype=np.int64)
    weights = rng.choice(np.asarray([0.0, 0.125, 1.0, 1.0, 3.75, 12.0]), size=95)
    edges = [(int(catalog.peak_id[a]), int(catalog.peak_id[b])) for a, b in edge_pairs]
    candidate = (edges, weights.astype(np.float64))
    prepared = BranchSizeCalculator.prepare(catalog, candidate)

    np.testing.assert_array_equal(prepared.edge_rows, np.asarray(edge_pairs, dtype=np.int64)[np.argsort(weights, kind="stable")])
    gates = {0.01, 0.125, 1.0, 3.75, 12.0, 20.0}
    for value in np.unique(weights):
        if value <= 0.0:
            continue
        gates.add(float(np.nextafter(value, -np.inf)))
        gates.add(float(value))
        gates.add(float(np.nextafter(value, np.inf)))

    for gate in sorted(gates):
        accepted = [edge for edge, delta in zip(edges, weights) if float(delta) <= gate]
        count, labels = label_peak_branches(catalog, accepted)
        expected = np.bincount(labels.astype(np.int64), minlength=count)
        actual = np.sort(prepared.sizes(gate))
        np.testing.assert_array_equal(actual, np.sort(expected))


def test_prepared_merge_forest_handles_empty_graph_and_isolated_events():
    catalog = pd.DataFrame({
        "peak_id": [9, 2, 7], "x": [0, 1, 2], "y": [0, 0, 0], "theta": [1.0, 2.0, 3.0],
    })
    prepared = BranchSizeCalculator.prepare(catalog, ([], np.empty(0, dtype=np.float64)))
    np.testing.assert_array_equal(prepared.sizes(100.0), np.ones(3, dtype=np.int64))
    assert prepared.edge_rows.shape == (0, 2)


def test_vectorized_branch_threshold_recommendation_matches_legacy_rule_exactly():
    from roc_mcs.peaks.branches import recommend_min_branch_events

    def legacy(values, min_multiplicative_gap=4.0):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return 1, {"method": "largest_multiplicative_gap_v1", "status": "empty",
                       "ratio": None, "lower_size": None, "upper_size": None}
        values_int = values.astype(np.int64)
        unique = np.unique(values_int)
        gap_min = float(min_multiplicative_gap)
        if unique.size < 2:
            return 1, {"method": "largest_multiplicative_gap_v1", "status": "no_size_gap",
                       "ratio": None, "lower_size": int(unique[0]),
                       "upper_size": None, "min_multiplicative_gap": gap_min}
        candidates = []
        for index in range(unique.size - 1):
            lower = int(unique[index])
            upper = int(unique[index + 1])
            n_lower = int(np.count_nonzero(values_int <= lower))
            n_upper = int(values_int.size - n_lower)
            if n_lower < 2 or n_upper < 1:
                continue
            ratio = float(upper) / float(lower)
            candidates.append((ratio, lower, upper, n_lower, n_upper))
        if not candidates:
            return 1, {"method": "largest_multiplicative_gap_v1", "status": "insufficient_components",
                       "ratio": None, "lower_size": None, "upper_size": None,
                       "min_multiplicative_gap": gap_min}
        ratio, lower, upper, n_lower, n_upper = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        if ratio < gap_min:
            return 1, {"method": "largest_multiplicative_gap_v1", "status": "no_clear_gap",
                       "ratio": ratio, "lower_size": lower, "upper_size": upper,
                       "branches_below": n_lower, "branches_above": n_upper,
                       "min_multiplicative_gap": gap_min}
        threshold = int(np.floor(np.sqrt(float(lower) * float(upper)))) + 1
        threshold = max(lower + 1, min(upper, threshold))
        return threshold, {"method": "largest_multiplicative_gap_v1", "status": "clear_gap",
                           "ratio": ratio, "lower_size": lower, "upper_size": upper,
                           "branches_below": n_lower, "branches_above": n_upper,
                           "min_multiplicative_gap": gap_min, "threshold": threshold}

    rng = np.random.default_rng(20261127)
    samples = [np.asarray([], dtype=np.int64), np.asarray([1]), np.asarray([1, 1, 5])]
    for _ in range(80):
        values = rng.choice(
            np.asarray([1, 2, 3, 4, 5, 8, 16, 32, 64, 128], dtype=np.int64),
            size=int(rng.integers(1, 250)),
        )
        samples.append(values)
    samples.extend([
        np.asarray([1, 1, 2, 2, 8, 8, 32, 32]),
        np.asarray([1, 1, 2, 2, 4, 4, 8, 8]),
    ])

    for values in samples:
        assert recommend_min_branch_events(values) == legacy(values)
