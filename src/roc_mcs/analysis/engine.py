"""Workflow execution and explicit input resolution."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import ArtifactBundle, ArtifactStore
from .operations import OperationRegistry, OperationResult
from .workflow_history import NodeStatus, ProcessingNode, WorkflowHistory


SOURCE_REF = "@external-hdf5-scan"


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    artifact_id: str
    output_ports: tuple[str, ...]
    summary: str


def application_version() -> str:
    """Best-effort version stamp for reproducible run history."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("roc-mcs")
        except PackageNotFoundError:
            return "source-tree"
    except Exception:
        return "source-tree"


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def create_source_node(
    history: WorkflowHistory,
    h5_path: str | Path,
    scan_key: str,
) -> ProcessingNode:
    """Registers an external immutable source reference; raw HDF5 is not copied."""
    from roc_mcs.roi.h5_scan_repository import H5ScanRepository

    repository = H5ScanRepository(h5_path)
    info = repository.scans[scan_key]
    fingerprint = source_fingerprint(repository.h5_path)

    # Inspect only HDF5 metadata and theta. Do not read the full 3D volume
    # merely to create a source node or open the workbench.
    import h5py
    with h5py.File(repository.h5_path, "r") as handle:
        raw_dataset = handle[info.raw_path]
        raw_shape = tuple(map(int, raw_dataset.shape))
        raw_dtype = np.dtype(raw_dataset.dtype)
        theta = repository._theta_1d(np.asarray(handle[info.th_path]))
    if not np.isfinite(theta).all():
        raise ValueError("Ось θ содержит NaN или бесконечные значения")
    if len(raw_shape) != 3:
        raise ValueError(f"Ожидался трёхмерный HDF5 dataset, получено {raw_shape}")
    candidates = [axis for axis in (0, 2) if raw_shape[axis] == theta.size]
    if len(candidates) != 1:
        raise ValueError(
            f"Не удалось однозначно определить theta-axis: raw={raw_shape}, "
            f"theta={theta.size}, candidates={candidates}"
        )
    theta_axis = int(candidates[0])
    source_shape_yx = raw_shape[1:] if theta_axis == 0 else raw_shape[:2]
    theta_hash = hashlib.sha256(np.asarray(theta).tobytes()).hexdigest()[:16]
    signature_text = (
        f"{fingerprint['path']}|{fingerprint['size_bytes']}|{fingerprint['mtime_ns']}|"
        f"{scan_key}|{raw_shape}|{raw_dtype.str}|{theta_hash}"
    )
    signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:24]
    metadata = {
        "source_file": fingerprint["path"],
        "source_size_bytes": fingerprint["size_bytes"],
        "source_mtime_ns": fingerprint["mtime_ns"],
        "scan_key": scan_key,
        "raw_dataset": info.raw_path,
        "theta_dataset": info.th_path,
        "theta_axis": theta_axis,
        "source_shape": list(raw_shape),
        "source_shape_yx": list(map(int, source_shape_yx)),
        "theta_count": int(theta.size),
        "theta_min": float(np.nanmin(theta)),
        "theta_max": float(np.nanmax(theta)),
        "grid_id": signature,
        "dtype": str(raw_dtype),
    }
    return history.add_node(
        "source.scan",
        f"Скан · {scan_key}",
        metadata=metadata,
        output_refs={"scan": SOURCE_REF},
        status=NodeStatus.SUCCEEDED,
        software_version=application_version(),
        summary=(
            f"{scan_key}: {source_shape_yx[1]}×{source_shape_yx[0]} "
            f"pixels × {theta.size} θ"
        ),
    )


class WorkflowExecutor:
    """Runs one already-created node and writes its result as a new artifact."""

    def __init__(
        self,
        history: WorkflowHistory,
        artifact_store: ArtifactStore,
        registry: OperationRegistry,
    ):
        self.history = history
        self.artifact_store = artifact_store
        self.registry = registry

    def resolve_node_inputs(self, node: ProcessingNode) -> dict[str, ArtifactBundle]:
        """Resolve the exact parent outputs declared by the node."""
        inputs: dict[str, ArtifactBundle] = {}
        for port, reference in node.input_refs.items():
            parent = self.history.get(reference["node_id"])
            output_name = reference["output"]
            output_ref = parent.output_refs.get(output_name)
            if output_ref is None:
                raise ValueError(
                    f"У родительского узла {parent.label!r} нет выхода {output_name!r}"
                )
            if output_ref == SOURCE_REF:
                inputs[port] = self._load_external_scan(parent)
            else:
                inputs[port] = self.artifact_store.read(output_ref)
        return inputs

    def load_output(self, node: ProcessingNode, output_name: str) -> ArtifactBundle:
        """Load a specific output bundle from a node."""
        try:
            artifact_ref = node.output_refs[output_name]
        except KeyError as exc:
            raise KeyError(f"Узел {node.label!r} не содержит выход {output_name!r}") from exc
        if artifact_ref == SOURCE_REF:
            return self._load_external_scan(node)
        return self.artifact_store.read(artifact_ref)

    def execute_node(self, node_id: str) -> ExecutionOutcome:
        node = self.history.get(node_id)
        if node.status != NodeStatus.RUNNING:
            raise ValueError(f"Узел должен быть running, сейчас {node.status}")
        inputs = self.resolve_node_inputs(node)
        result: OperationResult = self.registry.execute(
            node.operation_id, inputs, node.parameters
        )
        artifact_id = self.artifact_store.write(result.bundle)
        return ExecutionOutcome(
            artifact_id=artifact_id,
            output_ports=result.output_ports,
            summary=result.summary,
        )

    @staticmethod
    def _load_external_scan(source_node: ProcessingNode) -> ArtifactBundle:
        from roc_mcs.roi.h5_scan_repository import H5ScanRepository

        metadata = source_node.metadata
        path = Path(metadata["source_file"])
        current = source_fingerprint(path)
        expected = {
            "path": metadata["source_file"],
            "size_bytes": int(metadata["source_size_bytes"]),
            "mtime_ns": int(metadata["source_mtime_ns"]),
        }
        if current != expected:
            raise RuntimeError(
                "Исходный HDF5 изменился после создания узла источника. "
                "Создайте новую Analysis Session для актуального файла."
            )

        repository = H5ScanRepository(path)
        scan_key = str(metadata["scan_key"])
        scan = repository.load_source(scan_key)
        x_coords, y_coords = repository.centers_xy(scan_key)
        data_yxt = np.asarray(scan.yxt)
        arrays = {
            "data_yxt": data_yxt,
            "theta": np.asarray(scan.th, dtype=float),
            "x_coords": np.asarray(x_coords, dtype=float),
            "y_coords": np.asarray(y_coords, dtype=float),
        }
        source_meta = dict(metadata)
        source_meta.update({
            "grid_id": metadata["grid_id"],
            "shape_yx": list(map(int, data_yxt.shape[:2])),
            "theta_axis": int(scan.axis),
            "pool": {"size": 1, "stride": 1},
            "pool_signature": "raw",
        })
        return ArtifactBundle(arrays=arrays, metadata=source_meta)


def build_input_refs(
    history: WorkflowHistory,
    selected_node_id: str,
    operation_id: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Builds explicit inputs from exact ancestor outputs for the selected branch."""
    history.get(selected_node_id)

    def required(port: str) -> ProcessingNode:
        node = history.find_ancestor_output(selected_node_id, port)
        if node is None:
            raise ValueError(
                f"Для операции {operation_id!r} не найден предшествующий выход {port!r}. "
                "Выберите подходящий узел-родитель."
            )
        if node.status != NodeStatus.SUCCEEDED:
            raise ValueError(f"Входной узел {node.label!r} ещё не завершён успешно")
        return node

    selected = history.get(selected_node_id)
    refs: dict[str, dict[str, str]] = {}
    if operation_id == "preprocess.pool":
        if "scan" not in selected.output_refs:
            raise ValueError(
                "Pooling создаётся от узла исходного или уже pooled Scan. "
                "Выберите соответствующий узел Scan, а не spike/peak/fit результат."
            )
        refs["scan"] = {"node_id": selected.id, "output": "scan"}
    elif operation_id == "preprocess.spikes":
        if "scan" not in selected.output_refs:
            raise ValueError(
                "Spike detection создаётся от узла Scan. "
                "Выберите исходный или pooled Scan."
            )
        refs["scan"] = {"node_id": selected.id, "output": "scan"}
    elif operation_id == "peaks.detect":
        if not ({"scan", "spikes"} & set(selected.output_refs)):
            raise ValueError(
                "Детектор пиков следует запускать от Scan или Spike detection. "
                "Выберите один из этих узлов."
            )
        refs["scan"] = {"node_id": required("scan").id, "output": "scan"}
        use_interpolated = bool((parameters or {}).get("use_interpolated_data", False))
        spikes = history.find_ancestor_output(selected_node_id, "spikes")
        if use_interpolated and spikes is None:
            raise ValueError(
                "Для использования интерполированных данных сначала добавьте и выполните "
                "этап обнаружения спайков либо отключите соответствующий параметр."
            )
        if spikes is not None:
            if spikes.status != NodeStatus.SUCCEEDED:
                raise ValueError("Spike detection должен быть успешно завершён")
            # Keep upstream spike diagnostics available to the preview even if
            # peak detection itself is explicitly run on raw counts.
            refs["spikes"] = {"node_id": spikes.id, "output": "spikes"}
    elif operation_id == "fit.scan":
        if "peaks" not in selected.output_refs:
            raise ValueError(
                "Подгонку следует запускать от завершённого каталога пиков. "
                "Выберите узел детектора пиков, параметры fit будут новой ветвью."
            )
        refs["scan"] = {"node_id": required("scan").id, "output": "scan"}
        refs["peaks"] = {"node_id": selected.id, "output": "peaks"}
        if bool((parameters or {}).get("use_spike_mask", True)):
            spikes = history.find_ancestor_output(selected_node_id, "spikes")
            if spikes is not None:
                if spikes.status != NodeStatus.SUCCEEDED:
                    raise ValueError("Spike detection должен быть успешно завершён")
                refs["spikes"] = {"node_id": spikes.id, "output": "spikes"}
    else:
        raise KeyError(f"Для операции {operation_id!r} не определены входы")
    return refs


def make_operation_label(operation_id: str, parameters: Mapping[str, Any]) -> str:
    if operation_id == "preprocess.pool":
        return f"Pooling · {parameters['size']} / {parameters['stride']}"
    if operation_id == "preprocess.spikes":
        return "Спайки · κ auto"
    if operation_id == "peaks.detect":
        return (
            f"Пики · SG {parameters['smooth_window']} · "
            f"d={parameters['min_distance']}"
        )
    if operation_id == "fit.scan":
        return f"Fit · {parameters['model']} · {parameters['backend']}"
    return operation_id
