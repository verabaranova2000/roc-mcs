from __future__ import annotations

import numpy as np
import pytest

from roc_mcs.analysis.artifacts import ArtifactBundle
from roc_mcs.analysis.project import WorkflowProject, project_directory_for_scan
from roc_mcs.analysis.workflow_history import NodeStatus, WorkflowHistory


def _add_node(history, operation, label, parents=(), outputs=None, status=NodeStatus.SUCCEEDED):
    return history.add_node(
        operation,
        label,
        parent_ids=list(parents),
        output_refs=outputs or {},
        status=status,
    )


def test_history_subgraph_removal_covers_shared_dag_descendants_once():
    history = WorkflowHistory()
    root = _add_node(history, "source.scan", "Source")
    left = _add_node(history, "preprocess.left", "Left", (root.id,))
    right = _add_node(history, "preprocess.right", "Right", (root.id,))
    merge = _add_node(history, "analysis.merge", "Merge", (left.id, right.id))
    tail = _add_node(history, "analysis.tail", "Tail", (merge.id,))

    candidate, removed = history.without_subgraph(left.id)

    assert {node.id for node in removed} == {left.id, merge.id, tail.id}
    assert len({node.id for node in removed}) == len(removed)
    assert {node.id for node in candidate.nodes} == {root.id, right.id}
    # The operation is transactional at the model level: the original stays intact.
    assert {node.id for node in history.nodes} == {
        root.id, left.id, right.id, merge.id, tail.id,
    }


def test_project_deletes_only_artifacts_without_remaining_references(tmp_path):
    project = WorkflowProject(tmp_path / "sample.roi-analysis")
    history = WorkflowHistory()
    source = _add_node(history, "source.scan", "Source")
    shared_artifact = project.artifact_store.write(
        ArtifactBundle(arrays={"values": np.arange(4, dtype=np.int16)})
    )
    left = _add_node(
        history, "analysis.left", "Left", (source.id,), {"result": shared_artifact}
    )
    right = _add_node(
        history, "analysis.right", "Right", (source.id,), {"result": shared_artifact}
    )
    tail = _add_node(history, "analysis.tail", "Tail", (left.id,))
    project.history = history
    project.save()

    outcome = project.remove_subgraph(left.id)
    assert set(outcome["removed_node_ids"]) == {left.id, tail.id}
    assert shared_artifact not in outcome["deleted_artifact_ids"]
    assert project.artifact_store.path_for(shared_artifact).is_file()
    assert {node.id for node in WorkflowHistory.load(project.manifest_path).nodes} == {
        source.id, right.id,
    }

    # Once the last surviving node reference is removed, the orphaned HDF5
    # artifact is cleaned up as part of the same project operation.
    outcome = project.remove_subgraph(right.id)
    assert shared_artifact in outcome["deleted_artifact_ids"]
    assert not project.artifact_store.path_for(shared_artifact).exists()


def test_source_and_running_subgraphs_are_protected(tmp_path):
    project = WorkflowProject(tmp_path / "guarded.roi-analysis")
    source = _add_node(project.history, "source.scan", "Source")
    branch = _add_node(project.history, "analysis.branch", "Branch", (source.id,))
    running = project.history.add_node(
        "analysis.running", "Running", parent_ids=[branch.id], status=NodeStatus.PENDING
    )
    project.history.transition(running.id, NodeStatus.RUNNING)
    project.save()

    with pytest.raises(ValueError, match="Исходный скан"):
        project.remove_subgraph(source.id)
    with pytest.raises(RuntimeError, match="выполняется этап"):
        project.remove_subgraph(branch.id)
    assert {node.id for node in project.history.nodes} == {source.id, branch.id, running.id}
    assert project.manifest_path.is_file()


def test_scan_project_directory_is_stable_and_sanitizes_scan_key(tmp_path):
    source = tmp_path / "My experiment.h5"
    root = tmp_path / "Analyses"
    first = project_directory_for_scan(source, "scan/one:raw", root)
    second = project_directory_for_scan(source, "scan/one:raw", root)
    assert first == second
    assert first.parent == root.resolve()
    assert first.name == "My experiment_scan_one_raw.roi-analysis"
