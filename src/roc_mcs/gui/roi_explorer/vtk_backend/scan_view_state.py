"""Per-scan, presentation-only VTK view-state persistence.

This JSON sidecar is deliberately separate from ``workflow.json`` and HDF5
artifacts: camera pose and display controls describe how a scan is viewed, not
how its scientific outputs were computed.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _finite_number(value: Any, *, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        return None
    return number


def _vector(value: Any, *, minimum: float = -1.0e15, maximum: float = 1.0e15):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result = [_finite_number(item, minimum=minimum, maximum=maximum) for item in value]
    return result if all(item is not None for item in result) else None


def normalize_view_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return only the known, bounded schema fields."""
    if not isinstance(value, Mapping):
        raise ValueError("Состояние вида должно быть JSON-объектом.")
    version = int(value.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Неподдерживаемая версия view_state: {version}")

    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    axis = value.get("axis_scales")
    if isinstance(axis, (list, tuple)) and len(axis) == 3:
        scales = [_finite_number(item, minimum=0.1, maximum=10.0) for item in axis]
        if all(item is not None for item in scales):
            result["axis_scales"] = scales

    transfer = value.get("transfer")
    if isinstance(transfer, Mapping):
        validated: dict[str, Any] = {}
        bounds = {
            "low_percent": (0.0, 99.0),
            "high_percent": (1.0, 100.0),
            "opacity": (0.0, 1.0),
            "gamma": (0.05, 8.0),
            "sample_factor": (0.25, 8.0),
            "unit_distance": (0.05, 20.0),
            "projection_opacity": (0.0, 1.0),
        }
        for key, (low, high) in bounds.items():
            number = _finite_number(transfer.get(key), minimum=low, maximum=high)
            if number is not None:
                validated[key] = number
        if (
            "low_percent" in validated and "high_percent" in validated
            and validated["low_percent"] >= validated["high_percent"]
        ):
            validated.pop("low_percent", None)
            validated.pop("high_percent", None)
        for key in ("palette", "projection_palette"):
            item = transfer.get(key)
            if item is None and key == "projection_palette":
                validated[key] = None
            elif isinstance(item, str) and item and len(item) <= 64:
                validated[key] = item
        if isinstance(transfer.get("projection_enabled"), bool):
            validated["projection_enabled"] = transfer["projection_enabled"]
        if validated:
            result["transfer"] = validated

    camera = value.get("camera")
    if isinstance(camera, Mapping):
        validated_camera: dict[str, Any] = {}
        for key in ("position", "focal_point", "view_up"):
            vector = _vector(camera.get(key))
            if vector is not None:
                validated_camera[key] = vector
        parallel_scale = _finite_number(
            camera.get("parallel_scale"), minimum=1.0e-12, maximum=1.0e15
        )
        view_angle = _finite_number(camera.get("view_angle"), minimum=1.0, maximum=179.0)
        if parallel_scale is not None:
            validated_camera["parallel_scale"] = parallel_scale
        if view_angle is not None:
            validated_camera["view_angle"] = view_angle
        if isinstance(camera.get("parallel_projection"), bool):
            validated_camera["parallel_projection"] = camera["parallel_projection"]
        if (
            all(key in validated_camera for key in ("position", "focal_point", "view_up"))
            and (
                sum((a - b) ** 2 for a, b in zip(
                    validated_camera["position"], validated_camera["focal_point"]
                )) > 1.0e-20
            )
            and sum(component * component for component in validated_camera["view_up"]) > 1.0e-20
        ):
            result["camera"] = validated_camera

    scene = value.get("scene")
    if isinstance(scene, Mapping):
        validated_scene = {
            key: scene[key]
            for key in ("axes_visible", "bounding_box_visible", "axis_ticks_visible", "grid_visible")
            if isinstance(scene.get(key), bool)
        }
        if validated_scene:
            result["scene"] = validated_scene

    return result


def load_view_state(path: str | Path) -> dict[str, Any] | None:
    """Read a scan sidecar; missing/invalid files do not interrupt data viewing."""
    target = Path(path)
    if not target.is_file():
        return None
    with target.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return normalize_view_state(payload)


def save_view_state(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomically replace a strict JSON sidecar, leaving no partial write."""
    target = Path(path)
    payload = json.dumps(
        normalize_view_state(state), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
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
    return target
