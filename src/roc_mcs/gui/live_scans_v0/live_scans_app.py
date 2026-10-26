"""Точка входа Live ROC Dashboard; поддерживает запуск двойным щелчком в Windows."""
from __future__ import annotations
import ctypes, os, sys, traceback
from datetime import datetime
from pathlib import Path

def _project_root() -> Path: return Path(__file__).resolve().parents[4]
def _same_environment(current: Path, expected: Path) -> bool:
    try: return current.resolve().parent == expected.resolve().parent
    except OSError: return os.path.normcase(str(current.parent)) == os.path.normcase(str(expected.parent))
def _log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) if os.name == "nt" else Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    folder = base / "roc_mcs" / "logs"; folder.mkdir(parents=True, exist_ok=True)
    return folder / "live_scans_startup.log"
def _report_error() -> None:
    details = traceback.format_exc()
    try:
        with _log_path().open("a", encoding="utf-8") as f: f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}]\n{details}\n")
    except OSError: pass
    message = "Live ROC Dashboard не удалось запустить. Подробности записаны в:\n" + str(_log_path()) + "\n\n" + details[-1800:]
    if os.name == "nt":
        try: ctypes.windll.user32.MessageBoxW(None, message, "Live ROC Dashboard — ошибка запуска", 0x10)
        except Exception: pass
    else: sys.stderr.write(message + "\n")
def _bootstrap() -> None:
    root = _project_root(); os.chdir(root)
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path: sys.path.insert(0, str(src))
    for name in (".venv", "venv"):
        envdir = root / name; expected = envdir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not expected.is_file() or _same_environment(Path(sys.executable), expected): continue
        executable = expected.with_name("pythonw.exe") if os.name == "nt" and expected.with_name("pythonw.exe").is_file() else expected
        os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])

def main() -> int:
    try:
        _bootstrap()
        from roc_mcs.gui.live_scans.roc_live_dashboard_v4 import main as launch_dashboard
        launch_dashboard(); return 0
    except SystemExit: raise
    except Exception:
        _report_error(); return 1

if __name__ == "__main__": raise SystemExit(main())
