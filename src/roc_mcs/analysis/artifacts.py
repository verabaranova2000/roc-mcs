"""Immutable HDF5 artifact storage for workflow outputs (no pickle)."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import h5py
import numpy as np


_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _normalize_json(value: Any) -> Any:
    """Convert scientific metadata to strict JSON without NaN/Inf tokens."""
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        # JSON has no NaN/Infinity. None means "not defined for this iteration".
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Неподдерживаемое значение метаданных: {type(value).__name__}")


def _encode_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            _normalize_json(dict(value)), ensure_ascii=False, sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Метаданные артефакта не сериализуются: {exc}") from exc


def _write_array(group: h5py.Group, name: str, value: Any) -> None:
    if not _KEY_RE.fullmatch(name):
        raise ValueError(f"Некорректное имя массива: {name!r}")
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise TypeError(f"Object-массив {name!r} запрещён; используйте числовой или строковый dtype")
    if array.dtype.kind == "U":
        dtype = h5py.string_dtype("utf-8")
        data = array.astype(object)
        group.create_dataset(name, data=data, dtype=dtype)
        return
    options = {}
    if array.ndim and array.size >= 256:
        options = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    group.create_dataset(name, data=array, **options)


@dataclass(slots=True)
class ArtifactBundle:
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    """Хранит каждый результат в отдельном неизменяемом HDF5-файле."""

    # v2 adds nullable numeric/boolean table columns. The reader remains able
    # to open v1 projects from the first Workbench MVP, but old readers reject
    # v2 rather than silently interpreting masked values as zeros.
    FORMAT_VERSION = 2
    READABLE_FORMAT_VERSIONS = {1, 2}

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        if not _ID_RE.fullmatch(str(artifact_id)):
            raise ValueError("Некорректный artifact_id")
        return self.directory / f"{artifact_id}.h5"

    def write(self, bundle: ArtifactBundle) -> str:
        artifact_id = uuid4().hex
        target = self._path(artifact_id)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{artifact_id}.", suffix=".tmp", dir=self.directory
        )
        os.close(fd)
        try:
            with h5py.File(temporary, "w") as handle:
                handle.attrs["format_version"] = self.FORMAT_VERSION
                handle.attrs["metadata_json"] = _encode_json(bundle.metadata)
                arrays_group = handle.create_group("arrays")
                for name, array in bundle.arrays.items():
                    _write_array(arrays_group, name, array)
                tables_group = handle.create_group("tables")
                for table_name, table in bundle.tables.items():
                    if not _KEY_RE.fullmatch(table_name):
                        raise ValueError(f"Некорректное имя таблицы: {table_name!r}")
                    table_group = tables_group.create_group(table_name)
                    self._write_table(table_group, table)
                handle.flush()
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return artifact_id

    @staticmethod
    def _write_table(group: h5py.Group, table: Any) -> None:
        """Поддерживает pandas.DataFrame и последовательность словарей.

        Строковые столбцы записываются как UTF-8 плюс явная null-mask. Это
        принципиально важно для pandas 3, где строки могут автоматически
        получать StringDtype, а None при обратном создании DataFrame может
        превращаться в NaN. Nullable numeric/boolean extension-типы также
        сохраняются с отдельной маской пропусков.
        """
        try:
            import pandas as pd
        except ImportError:
            pd = None

        if pd is not None and isinstance(table, pd.DataFrame):
            frame = table
        else:
            if pd is None:
                raise RuntimeError("Для сохранения таблиц требуется pandas")
            frame = pd.DataFrame.from_records(table)

        group.attrs["columns_json"] = json.dumps(list(map(str, frame.columns)))
        group.attrs["row_count"] = int(len(frame))

        for index in range(frame.shape[1]):
            name = f"c{index:04d}"
            series = frame.iloc[:, index]
            dtype = series.dtype

            is_extension = pd.api.types.is_extension_array_dtype(dtype)
            is_nullable_numeric = is_extension and (
                pd.api.types.is_integer_dtype(dtype)
                or pd.api.types.is_float_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
            )

            # Detect nullable numeric/boolean extension dtypes before asking
            # pandas whether a dtype is string-like. This keeps behavior stable
            # across pandas 2.x and 3.x dtype inference changes.
            if is_nullable_numeric:
                nulls = np.asarray(series.isna(), dtype=np.bool_)
                numpy_dtype = np.dtype(dtype.numpy_dtype)
                values = series.to_numpy(dtype=numpy_dtype, na_value=0)
                _write_array(group, name, values)
                group.create_dataset(f"null_{name}", data=nulls)
                group[name].attrs["logical_type"] = "nullable"
                group[name].attrs["pandas_dtype"] = str(dtype)
                continue

            is_string = pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype)
            if is_string:
                normalized: list[str] = []
                nulls: list[bool] = []
                # Iteration through the Series preserves the actual missing
                # sentinel used by object, StringDtype and pandas 3 string dtypes.
                for value in series.array:
                    try:
                        missing = bool(pd.isna(value))
                    except (TypeError, ValueError):
                        missing = value is None
                    nulls.append(missing)
                    if missing:
                        normalized.append("")
                    elif isinstance(value, (str, np.str_)):
                        normalized.append(str(value))
                    else:
                        # Preserve the legacy behavior for scalar object values
                        # while refusing containers whose repr is not a stable
                        # scientific table representation.
                        if isinstance(value, (list, tuple, dict, set, np.ndarray)):
                            raise TypeError(
                                f"Столбец {str(frame.columns[index])!r} содержит "
                                f"неподдерживаемое object-значение {type(value).__name__}"
                            )
                        normalized.append(str(value))

                dtype_h5 = h5py.string_dtype("utf-8")
                group.create_dataset(
                    name, data=np.asarray(normalized, dtype=object), dtype=dtype_h5
                )
                group.create_dataset(
                    f"null_{name}", data=np.asarray(nulls, dtype=np.bool_)
                )
                group[name].attrs["logical_type"] = "string"
                continue

            values = series.to_numpy()
            _write_array(group, name, values)
            group[name].attrs["logical_type"] = "native"


    def read(self, artifact_id: str) -> ArtifactBundle:
        path = self._path(artifact_id)
        if not path.is_file():
            raise FileNotFoundError(f"Артефакт не найден: {artifact_id}")
        arrays: dict[str, np.ndarray] = {}
        tables: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:
            version = int(handle.attrs.get("format_version", -1))
            if version not in self.READABLE_FORMAT_VERSIONS:
                raise ValueError(f"Неподдерживаемый формат артефакта: {version}")
            metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
            for name, dataset in handle["arrays"].items():
                arrays[name] = self._read_dataset(dataset)
            for table_name, group in handle["tables"].items():
                tables[table_name] = self._read_table(group)
        return ArtifactBundle(arrays=arrays, tables=tables, metadata=metadata)

    @staticmethod
    def _read_dataset(dataset: h5py.Dataset) -> np.ndarray:
        value = dataset[()]
        array = np.asarray(value)
        if array.dtype.kind in {"S", "O"}:
            if array.ndim == 0:
                item = array.item()
                if isinstance(item, bytes):
                    return np.asarray(item.decode("utf-8"), dtype=object)
                return array
            flat = [
                item.decode("utf-8") if isinstance(item, bytes) else item
                for item in array.ravel()
            ]
            array = np.asarray(flat, dtype=object).reshape(array.shape)
        return array

    @classmethod
    def _read_table(cls, group: h5py.Group):
        import pandas as pd

        columns = json.loads(group.attrs["columns_json"])
        row_count = int(group.attrs.get("row_count", -1))
        restored_columns: list[Any] = []
        for index, name in enumerate(columns):
            dataset = group[f"c{index:04d}"]
            values = cls._read_dataset(dataset)
            if values.ndim != 1 or (row_count >= 0 and len(values) != row_count):
                raise ValueError(
                    f"Повреждённая таблица {group.name!r}: столбец {name!r} "
                    f"имеет shape={values.shape}, row_count={row_count}"
                )
            logical_type = dataset.attrs.get("logical_type", "")
            if isinstance(logical_type, bytes):
                logical_type = logical_type.decode("utf-8")

            if logical_type == "string":
                null_name = f"null_{dataset.name.rsplit('/', 1)[-1]}"
                if null_name not in group:
                    raise ValueError(
                        f"Повреждённая таблица {group.name!r}: отсутствует null-mask "
                        f"для столбца {name!r}"
                    )
                null_mask = np.asarray(group[null_name], dtype=bool)
                if null_mask.shape != values.shape:
                    raise ValueError(
                        f"Повреждённая таблица {group.name!r}: null-mask столбца "
                        f"{name!r} имеет несовместимую форму"
                    )
                # Explicit object dtype prevents pandas 3 string inference from
                # converting Python None back to NaN during DataFrame creation.
                restored = [
                    None if is_null else str(value)
                    for value, is_null in zip(values, null_mask)
                ]
                series = pd.Series(restored, dtype=object, name=name)
            elif logical_type == "nullable":
                null_name = f"null_{dataset.name.rsplit('/', 1)[-1]}"
                if null_name not in group:
                    raise ValueError(
                        f"Повреждённая таблица {group.name!r}: отсутствует null-mask "
                        f"для nullable-столбца {name!r}"
                    )
                null_mask = np.asarray(group[null_name], dtype=bool)
                if null_mask.shape != values.shape:
                    raise ValueError(
                        f"Повреждённая таблица {group.name!r}: null-mask столбца "
                        f"{name!r} имеет несовместимую форму"
                    )
                pandas_dtype = dataset.attrs.get("pandas_dtype", "")
                if isinstance(pandas_dtype, bytes):
                    pandas_dtype = pandas_dtype.decode("utf-8")
                restored = [
                    pd.NA if is_null else value
                    for value, is_null in zip(values, null_mask)
                ]
                series = pd.Series(
                    pd.array(restored, dtype=str(pandas_dtype)), name=name
                )
            else:
                # Legacy/native NumPy columns retain their original dtype and
                # NaN semantics. Series are concatenated positionally below so
                # duplicate column labels are not silently discarded.
                series = pd.Series(values, copy=False, name=name)
            restored_columns.append(series)

        if not restored_columns:
            return pd.DataFrame(index=range(max(0, row_count)))
        frame = pd.concat(restored_columns, axis=1, copy=False)
        frame.columns = columns
        return frame

    def contains(self, artifact_id: str) -> bool:
        try:
            return self._path(artifact_id).is_file()
        except ValueError:
            return False

    def path_for(self, artifact_id: str) -> Path:
        return self._path(artifact_id)
