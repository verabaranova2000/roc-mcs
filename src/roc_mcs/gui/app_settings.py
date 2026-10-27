"""Единые пользовательские настройки ROC-MCS."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS: dict[str, Any] = {
    "schema_version": 1,
    "hdf5_databases": [],
    "last_hdf5_database": "",
    "experiments_dir": "",
    "mcs_temp_dir": "",
    "mcs32_exe": "",
}


def settings_path() -> Path:
    """Путь к настройкам пользователя; ROC_MCS_SETTINGS позволяет задать свой."""
    override = os.environ.get("ROC_MCS_SETTINGS", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser().resolve()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home() / "AppData" / "Local")
        return root / "ROC-MCS" / "settings.json"
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "roc_mcs" / "settings.json"


def load_settings() -> dict[str, Any]:
    """Читает JSON, сохраняя неизвестные ключи для совместимости/расширения."""
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError, TypeError):
        data = {}
    return {**DEFAULT_SETTINGS, **data}


def save_settings(settings: dict[str, Any]) -> Path:
    """Атомарно сохраняет настройки, не оставляя частично записанный JSON."""
    path = settings_path(); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**DEFAULT_SETTINGS, **settings, "schema_version": 1}
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False); stream.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        try: Path(temp_name).unlink(missing_ok=True)
        except OSError: pass
        raise
    return path


def _configured_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(os.path.expandvars(value.strip())).expanduser()
    return raw.resolve() if raw.is_absolute() else (settings_path().parent / raw).resolve()


def resolve_setting_path(key: str, *, require_exists: bool = False) -> Path | None:
    """Разрешает путь из пользовательских настроек; относительные пути считаются от settings.json."""
    path = _configured_path(load_settings().get(key))
    if path is None or require_exists and not path.exists():
        return None
    return path


def resolve_initial_database() -> Path | None:
    """Берёт последний использованный файл, затем первый существующий из списка."""
    env_path = _configured_path(os.environ.get("ROC_MCS_HDF5", ""))
    candidates = [env_path] if env_path is not None else []
    settings = load_settings()
    candidates.append(_configured_path(settings.get("last_hdf5_database")))
    configured = settings.get("hdf5_databases", [])
    if isinstance(configured, list):
        candidates.extend(_configured_path(item) for item in configured)
    seen: set[str] = set()
    for path in candidates:
        if path is None:
            continue
        marker = os.path.normcase(str(path))
        if marker not in seen and path.is_file():
            return path
        seen.add(marker)
    return None


def remember_hdf5_database(path: str | Path) -> Path:
    settings = load_settings(); settings["last_hdf5_database"] = str(Path(path).expanduser().resolve())
    return save_settings(settings)
