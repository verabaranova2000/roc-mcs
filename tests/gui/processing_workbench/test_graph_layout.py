from dataclasses import dataclass

import pytest

from roc_mcs.gui.processing_workbench.graph_layout import vertical_dag_positions


@dataclass(frozen=True)
class Node:
    id: str
    parent_ids: tuple[str, ...] = ()


def test_linear_history_is_one_vertical_column():
    nodes = [Node("raw"), Node("qc", ("raw",)), Node("spikes", ("qc",))]
    positions = vertical_dag_positions(nodes, width=240, height=78)
    assert len({round(point[0], 6) for point in positions.values()}) == 1
    assert positions["raw"][1] < positions["qc"][1] < positions["spikes"][1]


def test_alternative_branches_are_side_by_side_and_descendants_follow_branch():
    nodes = [Node("raw"), Node("a", ("raw",)), Node("b", ("raw",)),
             Node("a2", ("a",)), Node("b2", ("b",))]
    positions = vertical_dag_positions(nodes, width=240, height=78)
    assert positions["a"][1] == positions["b"][1]
    assert positions["a"][0] != positions["b"][0]
    assert positions["a2"][0] == positions["a"][0]
    assert positions["b2"][0] == positions["b"][0]
    assert positions["a2"][1] > positions["a"][1]


def test_merge_node_is_below_both_parents():
    nodes = [Node("raw"), Node("a", ("raw",)), Node("b", ("raw",)), Node("merge", ("a", "b"))]
    positions = vertical_dag_positions(nodes, width=240, height=78)
    assert positions["merge"][1] > max(positions["a"][1], positions["b"][1])


def test_duplicate_ids_and_cycles_are_rejected():
    with pytest.raises(ValueError, match="повторяющиеся"):
        vertical_dag_positions([Node("x"), Node("x")], width=240, height=78)
    with pytest.raises(ValueError, match="цикл"):
        vertical_dag_positions([Node("a", ("b",)), Node("b", ("a",))], width=240, height=78)
