from dataclasses import dataclass

import pytest

from roc_mcs.gui.processing_workbench.graph_layout import (
    vertical_dag_edge_ports,
    vertical_dag_positions,
)


@dataclass(frozen=True)
class Node:
    id: str
    parent_ids: tuple[str, ...] = ()


def test_linear_history_is_one_vertical_column():
    nodes = [Node("raw"), Node("qc", ("raw",)), Node("pool", ("qc",)), Node("spikes", ("pool",))]
    positions = vertical_dag_positions(nodes, width=240, height=78)
    assert len({round(point[0], 6) for point in positions.values()}) == 1
    assert positions["raw"][1] < positions["qc"][1] < positions["pool"][1] < positions["spikes"][1]


def test_alternative_branches_are_side_by_side_and_descendants_follow_branch():
    nodes = [Node("raw"), Node("a", ("raw",)), Node("b", ("raw",)),
             Node("a2", ("a",)), Node("b2", ("b",))]
    positions = vertical_dag_positions(nodes, width=240, height=78)
    assert positions["a"][1] == positions["b"][1]
    assert positions["a"][0] != positions["b"][0]
    assert positions["a2"][0] == positions["a"][0]
    assert positions["b2"][0] == positions["b"][0]
    assert positions["a2"][1] > positions["a"][1]


def test_independent_merge_keeps_both_parent_edges():
    nodes = [Node("raw"), Node("a", ("raw",)), Node("b", ("raw",)), Node("merge", ("a", "b"))]
    positions = vertical_dag_positions(nodes, width=240, height=78)
    ports = vertical_dag_edge_ports(nodes, positions, width=240, height=78)
    assert ("a", "merge") in ports
    assert ("b", "merge") in ports
    assert positions["merge"][1] > max(positions["a"][1], positions["b"][1])
    assert ports[("a", "merge")][1] != ports[("b", "merge")][1]


def test_duplicate_ids_and_cycles_are_rejected():
    with pytest.raises(ValueError, match="повторяющиеся"):
        vertical_dag_positions([Node("x"), Node("x")], width=240, height=78)
    with pytest.raises(ValueError, match="цикл"):
        vertical_dag_positions([Node("a", ("b",)), Node("b", ("a",))], width=240, height=78)


def test_sequential_workflow_removes_redundant_ancestor_shortcut():
    # Peaks consumes the pooled Scan and (optionally) the Spike artifact.
    # The structural graph also has Pool -> Peaks, but Pool -> Spikes -> Peaks
    # already preserves this ancestry; drawing the shortcut made a fake branch.
    nodes = [
        Node("scan"),
        Node("bad_pixels", ("scan",)),
        Node("pool", ("bad_pixels",)),
        Node("spikes", ("pool",)),
        Node("peaks", ("spikes", "pool")),
    ]
    positions = vertical_dag_positions(nodes, width=242, height=78)
    ports = vertical_dag_edge_ports(nodes, positions, width=242, height=78)

    expected_edges = {
        ("scan", "bad_pixels"),
        ("bad_pixels", "pool"),
        ("pool", "spikes"),
        ("spikes", "peaks"),
    }
    assert set(ports) == expected_edges
    assert len({round(point[0], 6) for point in positions.values()}) == 1
    assert [positions[key][1] for key in ("scan", "bad_pixels", "pool", "spikes", "peaks")] == sorted(
        positions[key][1] for key in ("scan", "bad_pixels", "pool", "spikes", "peaks")
    )


def test_branch_ports_are_compact_and_deterministic():
    nodes = [Node("root"), Node("left", ("root",)), Node("middle", ("root",)), Node("right", ("root",))]
    positions = vertical_dag_positions(nodes, width=242, height=78)
    first = vertical_dag_edge_ports(nodes, positions, width=242, height=78)
    second = vertical_dag_edge_ports(nodes, positions, width=242, height=78)

    starts = [first[("root", child)][0][0] for child in ("left", "middle", "right")]
    assert starts == [second[("root", child)][0][0] for child in ("left", "middle", "right")]
    assert len(set(starts)) == 3
    # Branch ports stay close to the card centre instead of being pushed to
    # the far left/right edges, which caused the old broad fan of arrows.
    assert max(starts) - min(starts) <= 32.0
    assert all(positions["root"][0] + 105.0 <= x <= positions["root"][0] + 137.0 for x in starts)
