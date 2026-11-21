"""Чистая компоновка DAG для вертикального представления Workbench."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable


def _reduced_graph(nodes: Iterable[object]):
    """Return nodes, transitive-reduced parents and a deterministic topological order.

    A workflow node may reference an older ancestor directly (for example, a
    peak detector consumes both the pooled Scan and the Spike artifact made
    from that same Scan). Drawing both edges creates a visual shortcut on top
    of the actual sequential path. The transitive reduction removes only that
    redundant visual edge; the immutable ``input_refs`` remain untouched.
    """
    items = list(nodes)
    by_id = {str(node.id): node for node in items}
    if len(by_id) != len(items):
        raise ValueError("Workflow содержит повторяющиеся ID узлов.")

    order = {key: index for index, key in enumerate(by_id)}
    parents_full: dict[str, tuple[str, ...]] = {}
    children: dict[str, list[str]] = {key: [] for key in by_id}
    indegree: dict[str, int] = {}

    for key, node in by_id.items():
        parent_ids = tuple(dict.fromkeys(
            str(parent) for parent in node.parent_ids if str(parent) in by_id
        ))
        parents_full[key] = parent_ids
        indegree[key] = len(parent_ids)
        for parent in parent_ids:
            children[parent].append(key)

    # Kahn's algorithm validates acyclicity before reachability is evaluated.
    queue = deque(key for key in by_id if indegree[key] == 0)
    topological: list[str] = []
    while queue:
        parent = queue.popleft()
        topological.append(parent)
        for child in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(topological) != len(by_id):
        raise ValueError("Workflow содержит цикл; DAG не может быть отображён.")

    # Descendant sets are computed bottom-up in reverse topological order.
    descendants: dict[str, set[str]] = {}
    for parent in reversed(topological):
        reachable: set[str] = set()
        for child in children[parent]:
            reachable.add(child)
            reachable.update(descendants[child])
        descendants[parent] = reachable

    reduced_parents: dict[str, tuple[str, ...]] = {}
    for child, parent_ids in parents_full.items():
        reduced_parents[child] = tuple(
            parent for parent in parent_ids
            if not any(
                other != parent and other in descendants[parent]
                for other in parent_ids
            )
        )

    return by_id, reduced_parents, topological, order


def vertical_dag_positions(
    nodes: Iterable[object], *, width: float, height: float,
    horizontal_gap: float = 34.0, vertical_gap: float = 36.0,
    origin_x: float = 20.0, origin_y: float = 20.0,
) -> dict[str, tuple[float, float]]:
    """Place a DAG top-to-bottom; linear workflows stay in one clean column.

    Layout uses the transitive reduction of the parent graph. This prevents
    direct references to an already represented ancestor from pulling a
    sequential chain sideways or creating shortcut arrows across later nodes.
    """
    by_id, parents, topological, order = _reduced_graph(nodes)

    depths: dict[str, int] = {}
    for key in topological:
        depths[key] = max((depths[parent] + 1 for parent in parents[key]), default=0)

    layers: dict[int, list[str]] = defaultdict(list)
    for key in by_id:
        layers[depths[key]].append(key)

    lanes: dict[str, float] = {}
    for level in sorted(layers):
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for key in layers[level]:
            canonical_parents = tuple(sorted(parents[key], key=order.__getitem__))
            groups[canonical_parents].append(key)

        ordered_groups = sorted(
            groups.items(),
            key=lambda entry: (
                sum(lanes[parent] for parent in entry[0]) / len(entry[0])
                if entry[0] else 0.0,
                min(order[key] for key in entry[1]),
            ),
        )
        previous_right = float("-inf")
        for parent_ids, child_ids in ordered_groups:
            anchor = (
                sum(lanes[parent] for parent in parent_ids) / len(parent_ids)
                if parent_ids else 0.0
            )
            count = len(child_ids)
            offsets = [index - (count - 1) / 2.0 for index in range(count)]
            desired = [anchor + offset for offset in offsets]
            shift = (
                max(0.0, previous_right + 1.0 - desired[0])
                if previous_right != float("-inf") else 0.0
            )
            for key, lane in zip(child_ids, desired):
                lanes[key] = lane + shift
            previous_right = max(lanes[key] for key in child_ids)

    min_lane = min(lanes.values(), default=0.0)
    return {
        key: (
            origin_x + (lanes[key] - min_lane) * (width + horizontal_gap),
            origin_y + depths[key] * (height + vertical_gap),
        )
        for key in by_id
    }


def vertical_dag_edge_ports(
    nodes: Iterable[object],
    positions: dict[str, tuple[float, float]],
    *,
    width: float,
    height: float,
    port_margin: float = 20.0,
) -> dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]:
    """Return compact, ordered ports for the transitive-reduced DAG edges.

    Ports are clustered around the centre of each card instead of being spread
    across its entire width. This keeps small branch fans visually restrained
    while retaining deterministic left-to-right ordering for larger graphs.
    Coordinates are absolute scene-space points on the card's bottom/top edge.
    """
    by_id, parents, _topological, order = _reduced_graph(nodes)
    missing_positions = sorted(set(by_id) - set(positions))
    if missing_positions:
        raise ValueError(f"Для узлов отсутствуют координаты: {missing_positions}")

    children: dict[str, list[str]] = {key: [] for key in by_id}
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[parent].append(child)

    def offsets(neighbours: list[str]) -> dict[str, float]:
        ordered = sorted(
            neighbours,
            key=lambda key: (
                float(positions[key][0]) + width / 2.0,
                order[key],
            ),
        )
        if not ordered:
            return {}
        if len(ordered) == 1:
            values = [width / 2.0]
        else:
            available = max(0.0, width - 2.0 * min(max(0.0, port_margin), width / 2.0))
            spacing = min(16.0, available / (len(ordered) - 1))
            half_span = spacing * (len(ordered) - 1) / 2.0
            centre = width / 2.0
            values = [centre - half_span + index * spacing for index in range(len(ordered))]
        return dict(zip(ordered, values))

    outgoing_offsets = {key: offsets(children[key]) for key in by_id}
    incoming_offsets = {key: offsets(list(parents[key])) for key in by_id}

    result: dict[
        tuple[str, str], tuple[tuple[float, float], tuple[float, float]]
    ] = {}
    for child, parent_ids in parents.items():
        child_x, child_y = positions[child]
        for parent in parent_ids:
            parent_x, parent_y = positions[parent]
            start = (
                float(parent_x + outgoing_offsets[parent][child]),
                float(parent_y + height),
            )
            end = (
                float(child_x + incoming_offsets[child][parent]),
                float(child_y),
            )
            result[(parent, child)] = (start, end)
    return result
