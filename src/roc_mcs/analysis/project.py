"""Local analysis-project layout and persistence."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .workflow_history import NodeStatus, WorkflowHistory


_ARTIFACT_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def project_directory_for_scan(
    h5_path: str | Path, scan_key: str, root: str | Path,
) -> Path:
    """Return the canonical per-scan ROI project directory.

    The HDF5 stem and scan key form a readable, deterministic folder name;
    unsafe characters are collapsed to underscores so the path is portable.
    """
    source = Path(h5_path).expanduser().resolve()
    safe_scan = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scan_key)).strip("._") or "scan"
    return Path(root).expanduser().resolve() / f"{source.stem}_{safe_scan}.roi-analysis"


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

    def remove_subgraph(self, node_id: str) -> dict[str, Any]:
        """Persistently remove a node and its full downstream dependency closure.

        The source scan is the immutable root of a project and cannot be
        removed.  A running operation also blocks deletion, avoiding races
        between a worker writing an artifact and its parent branch disappearing.
        The new manifest is committed atomically *before* unreachable artifacts
        are unlinked.  Cleanup failures leave harmless orphan files and are
        returned to the caller; the manifest is never rolled back to reference
        partially deleted output.
        """
        selected = self.history.get(str(node_id))
        if selected.operation_id == "source.scan":
            raise ValueError("Исходный скан проекта нельзя удалить из пайплайна.")

        candidate, removed_nodes = self.history.without_subgraph(str(node_id))
        running = [node.label for node in removed_nodes if node.status == NodeStatus.RUNNING]
        if running:
            raise RuntimeError(
                "Нельзя удалить ветвь, пока выполняется этап: " + ", ".join(running)
            )

        remaining_artifacts: set[str] = set()
        for node in candidate.nodes:
            for artifact_id in node.output_refs.values():
                value = str(artifact_id)
                if _ARTIFACT_ID_RE.fullmatch(value):
                    remaining_artifacts.add(value)

        removed_artifacts: set[str] = set()
        for node in removed_nodes:
            for artifact_id in node.output_refs.values():
                value = str(artifact_id)
                if _ARTIFACT_ID_RE.fullmatch(value):
                    removed_artifacts.add(value)

        # First commit the dependency graph.  If that fails, neither the live
        # project nor any artifact file is changed.
        candidate.save(self.manifest_path)
        self.history = candidate

        deleted: list[str] = []
        cleanup_errors: dict[str, str] = {}
        for artifact_id in sorted(removed_artifacts - remaining_artifacts):
            try:
                if self.artifact_store.delete(artifact_id, missing_ok=True):
                    deleted.append(artifact_id)
            except OSError as exc:
                cleanup_errors[artifact_id] = str(exc)

        return {
            "removed_node_ids": tuple(node.id for node in removed_nodes),
            "removed_labels": tuple(node.label for node in removed_nodes),
            "deleted_artifact_ids": tuple(deleted),
            "cleanup_errors": cleanup_errors,
        }

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
