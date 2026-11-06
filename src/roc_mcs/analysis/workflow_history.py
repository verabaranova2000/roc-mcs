"""Immutable run history for branchable scientific data processing.

The visible UI may render this graph as a tree, but the model is a DAG:
one analysis result may depend on several independently produced artifacts.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = 2


class NodeStatus(str):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def values(cls) -> set[str]:
        return {cls.PENDING, cls.RUNNING, cls.SUCCEEDED, cls.FAILED, cls.CANCELLED}


_TERMINAL = {NodeStatus.SUCCEEDED, NodeStatus.FAILED, NodeStatus.CANCELLED}
_ALLOWED = {
    NodeStatus.PENDING: {NodeStatus.RUNNING, NodeStatus.CANCELLED},
    NodeStatus.RUNNING: {NodeStatus.SUCCEEDED, NodeStatus.FAILED, NodeStatus.CANCELLED},
    NodeStatus.SUCCEEDED: set(),
    NodeStatus.FAILED: set(),
    NodeStatus.CANCELLED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Значение нельзя сохранить в JSON: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ProcessingNode:
    """Один конкретный запуск операции; параметры после создания неизменяемы."""

    id: str
    operation_id: str
    label: str
    parent_ids: tuple[str, ...]
    parameters_json: str
    status: str = NodeStatus.PENDING
    input_refs_json: str = "{}"
    output_refs_json: str = "{}"
    metadata_json: str = "{}"
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    software_version: str | None = None
    parameter_schema_version: int = 1
    error: str | None = None
    summary: str | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self.parameters_json)

    @property
    def input_refs(self) -> dict[str, dict[str, str]]:
        return json.loads(self.input_refs_json)

    @property
    def output_refs(self) -> dict[str, str]:
        return json.loads(self.output_refs_json)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation_id": self.operation_id,
            "label": self.label,
            "parent_ids": list(self.parent_ids),
            "parameters": self.parameters,
            "status": self.status,
            "inputs": self.input_refs,
            "output_refs": self.output_refs,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "software_version": self.software_version,
            "parameter_schema_version": self.parameter_schema_version,
            "error": self.error,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessingNode":
        status = str(data.get("status", NodeStatus.PENDING))
        if status not in NodeStatus.values():
            raise ValueError(f"Неизвестный статус узла: {status!r}")
        return cls(
            id=str(data["id"]),
            operation_id=str(data["operation_id"]),
            label=str(data["label"]),
            parent_ids=tuple(str(x) for x in data.get("parent_ids", ())),
            parameters_json=_json(data.get("parameters", {})),
            status=status,
            input_refs_json=_json(data.get("inputs", {})),
            output_refs_json=_json(data.get("output_refs", {})),
            metadata_json=_json(data.get("metadata", {})),
            created_at=str(data.get("created_at") or _now()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            software_version=data.get("software_version"),
            parameter_schema_version=int(data.get("parameter_schema_version", 1)),
            error=data.get("error"),
            summary=data.get("summary"),
        )


class WorkflowHistory:
    """Append-only граф запусков с явными ссылками на входы и выходы."""

    def __init__(self, nodes: list[ProcessingNode] | None = None):
        self._nodes: dict[str, ProcessingNode] = {}
        for node in nodes or []:
            if node.id in self._nodes:
                raise ValueError(f"Повторяется id узла: {node.id}")
            self._nodes[node.id] = node
        self._validate_graph()

    @property
    def nodes(self) -> tuple[ProcessingNode, ...]:
        return tuple(self._nodes.values())

    def get(self, node_id: str) -> ProcessingNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Неизвестный узел истории: {node_id}") from exc

    def add_node(
        self,
        operation_id: str,
        label: str,
        *,
        parent_ids: tuple[str, ...] | list[str] = (),
        input_refs: Mapping[str, Mapping[str, str]] | None = None,
        parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        output_refs: Mapping[str, str] | None = None,
        status: str = NodeStatus.PENDING,
        software_version: str | None = None,
        parameter_schema_version: int = 1,
        summary: str | None = None,
    ) -> ProcessingNode:
        if not operation_id.strip() or not label.strip():
            raise ValueError("operation_id и label не должны быть пустыми")
        if parameter_schema_version < 1:
            raise ValueError("parameter_schema_version должен быть положительным")
        status = str(status)
        if status not in NodeStatus.values():
            raise ValueError(f"Неизвестный статус узла: {status!r}")
        if status == NodeStatus.RUNNING:
            raise ValueError("Новый узел нельзя создавать сразу в статусе running")

        normalized_inputs: dict[str, dict[str, str]] = {}
        for port, ref in (input_refs or {}).items():
            port_name = str(port).strip()
            node_ref = str(ref.get("node_id", "")).strip()
            output_ref = str(ref.get("output", "")).strip()
            if not port_name or not node_ref or not output_ref:
                raise ValueError("Каждый input_ref должен содержать node_id и output")
            normalized_inputs[port_name] = {
                "node_id": node_ref,
                "output": output_ref,
            }

        parents = tuple(dict.fromkeys([
            *(str(x) for x in parent_ids),
            *(ref["node_id"] for ref in normalized_inputs.values()),
        ]))
        missing = [parent for parent in parents if parent not in self._nodes]
        if missing:
            raise ValueError(f"Родительские узлы ещё не существуют: {missing}")

        now = _now()
        node = ProcessingNode(
            id=uuid4().hex,
            operation_id=operation_id.strip(),
            label=label.strip(),
            parent_ids=parents,
            parameters_json=_json(dict(parameters or {})),
            status=status,
            input_refs_json=_json(normalized_inputs),
            output_refs_json=_json(dict(output_refs or {})),
            metadata_json=_json(dict(metadata or {})),
            created_at=now,
            finished_at=now if status in _TERMINAL else None,
            software_version=software_version,
            parameter_schema_version=parameter_schema_version,
            summary=summary,
        )
        self._nodes[node.id] = node
        return node

    def transition(
        self,
        node_id: str,
        status: str,
        *,
        output_refs: Mapping[str, str] | None = None,
        error: str | None = None,
        summary: str | None = None,
    ) -> ProcessingNode:
        """Меняет состояние выполнения, но не конфигурацию и не родителей."""
        node = self.get(node_id)
        status = str(status)
        if status not in _ALLOWED[node.status]:
            raise ValueError(f"Недопустимый переход состояния: {node.status} → {status}")
        now = _now()
        updated = replace(
            node,
            status=status,
            started_at=node.started_at or (now if status == NodeStatus.RUNNING else None),
            finished_at=now if status in _TERMINAL else None,
            output_refs_json=_json(dict(output_refs)) if output_refs is not None else node.output_refs_json,
            error=error,
            summary=summary if summary is not None else node.summary,
        )
        self._nodes[node_id] = updated
        return updated

    def children(self, node_id: str) -> tuple[ProcessingNode, ...]:
        self.get(node_id)
        return tuple(node for node in self._nodes.values() if node_id in node.parent_ids)

    def ancestors(self, node_id: str, *, include_self: bool = True) -> tuple[ProcessingNode, ...]:
        """Возвращает узел и его предков в порядке ближайший → дальний."""
        self.get(node_id)
        result: list[ProcessingNode] = []
        seen: set[str] = set()
        stack = [node_id] if include_self else list(self.get(node_id).parent_ids)
        while stack:
            current = stack.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = self.get(current)
            result.append(node)
            stack.extend(node.parent_ids)
        return tuple(result)

    def find_ancestor_output(self, node_id: str, output_name: str) -> ProcessingNode | None:
        for node in self.ancestors(node_id):
            if output_name in node.output_refs:
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_with": "roc_mcs.analysis",
            "nodes": [node.to_dict() for node in self._nodes.values()],
        }

    def save(self, path: str | Path) -> None:
        """Атомарная запись JSON-манифеста; сами научные массивы здесь не хранятся."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowHistory":
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        version = int(payload.get("schema_version", 1))
        if version not in {1, SCHEMA_VERSION}:
            raise ValueError(f"Неподдерживаемая версия истории: {version}")
        return cls([ProcessingNode.from_dict(item) for item in payload.get("nodes", [])])

    def _validate_graph(self) -> None:
        for node in self._nodes.values():
            missing = [parent for parent in node.parent_ids if parent not in self._nodes]
            if missing:
                raise ValueError(f"У узла {node.id} отсутствуют родители: {missing}")
            input_parents = {ref["node_id"] for ref in node.input_refs.values()}
            if not input_parents.issubset(set(node.parent_ids)):
                raise ValueError(f"Входы узла {node.id} ссылаются на неуказанных родителей")

        colors: dict[str, int] = {}

        def visit(node_id: str) -> None:
            color = colors.get(node_id, 0)
            if color == 1:
                raise ValueError("История содержит цикл")
            if color == 2:
                return
            colors[node_id] = 1
            for parent_id in self._nodes[node_id].parent_ids:
                visit(parent_id)
            colors[node_id] = 2

        for node_id in self._nodes:
            visit(node_id)
