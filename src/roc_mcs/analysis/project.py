"""Local analysis-project layout and persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .workflow_history import NodeStatus, WorkflowHistory


class WorkflowProject:
    """A portable folder: workflow.json + immutable HDF5 artifacts."""

    MANIFEST_NAME = "workflow.json"

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / self.MANIFEST_NAME
        self.artifact_store = ArtifactStore(self.directory / "artifacts")
        self.history = WorkflowHistory()

    @classmethod
    def create(
        cls,
        directory: str | Path,
        h5_path: str | Path,
        scan_key: str,
    ) -> "WorkflowProject":
        from .engine import create_source_node

        project = cls(directory)
        if project.manifest_path.exists():
            raise FileExistsError(
                f"Проект уже существует: {project.manifest_path}. "
                "Откройте его через WorkflowProject.open(), а не перезаписывайте."
            )
        create_source_node(project.history, h5_path, scan_key)
        project.save()
        return project

    @classmethod
    def open(cls, manifest_path: str | Path) -> "WorkflowProject":
        path = Path(manifest_path).expanduser().resolve()
        if path.is_dir():
            path = path / cls.MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(path)
        project = cls(path.parent)
        project.history = WorkflowHistory.load(path)
        project._recover_interrupted_runs()
        project.save()
        return project

    def _recover_interrupted_runs(self) -> None:
        """Mark nodes that were running when the process terminated as failed."""
        for node in self.history.nodes:
            if node.status == NodeStatus.RUNNING:
                self.history.transition(
                    node.id,
                    NodeStatus.FAILED,
                    error="Приложение завершилось до фиксации результата операции.",
                    summary="Незавершённый запуск; повторите операцию новой ветвью.",
                )

    def save(self) -> None:
        self.history.save(self.manifest_path)

    @property
    def source_node_id(self) -> str | None:
        for node in self.history.nodes:
            if node.operation_id == "source.scan":
                return node.id
        return None

    @property
    def source_metadata(self) -> dict[str, Any] | None:
        node_id = self.source_node_id
        return None if node_id is None else self.history.get(node_id).metadata
