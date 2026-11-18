"""Чистая компоновка DAG для вертикального представления Workbench."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def vertical_dag_positions(nodes: Iterable[object], *, width: float, height: float,
                           horizontal_gap: float = 34.0, vertical_gap: float = 36.0,
                           origin_x: float = 20.0, origin_y: float = 20.0) -> dict[str, tuple[float, float]]:
    """Возвращает позиции узлов; цепочки идут вниз, альтернативные ветви — в стороны."""
    items = list(nodes)
    by_id = {str(node.id): node for node in items}
    if len(by_id) != len(items):
        raise ValueError("Workflow содержит повторяющиеся ID узлов.")
    order = {key: index for index, key in enumerate(by_id)}
    parents = {key: tuple(str(p) for p in node.parent_ids if str(p) in by_id)
               for key, node in by_id.items()}
    depths: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(key: str) -> int:
        if key in depths:
            return depths[key]
        if key in visiting:
            raise ValueError("Workflow содержит цикл; DAG не может быть отображён.")
        visiting.add(key)
        value = max((depth(parent) + 1 for parent in parents[key]), default=0)
        visiting.remove(key)
        depths[key] = value
        return value

    for key in by_id:
        depth(key)

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
            key=lambda entry: (sum(lanes[p] for p in entry[0]) / len(entry[0]) if entry[0] else 0.0,
                               min(order[key] for key in entry[1])),
        )
        previous_right = float('-inf')
        for parent_ids, child_ids in ordered_groups:
            anchor = (sum(lanes[p] for p in parent_ids) / len(parent_ids)) if parent_ids else 0.0
            count = len(child_ids)
            offsets = [index - (count - 1) / 2.0 for index in range(count)]
            desired = [anchor + offset for offset in offsets]
            shift = max(0.0, previous_right + 1.0 - desired[0]) if previous_right != float('-inf') else 0.0
            for key, lane in zip(child_ids, desired):
                lanes[key] = lane + shift
            previous_right = max(lanes[key] for key in child_ids)

    min_lane = min(lanes.values(), default=0.0)
    return {
        key: (origin_x + (lanes[key] - min_lane) * (width + horizontal_gap),
              origin_y + depths[key] * (height + vertical_gap))
        for key in by_id
    }
