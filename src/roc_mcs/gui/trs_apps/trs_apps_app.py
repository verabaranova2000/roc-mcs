"""Точка входа TRS Apps с поиском окружения проекта."""
from __future__ import annotations
import ctypes, os, sys, traceback
from datetime import datetime
from pathlib import Path

def project_root() -> Path: return Path(__file__).resolve().parents[4]
def startup_log() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) if os.name == "nt" else Path(os.environ.get("XDG_STATE_HOME", Path.home()/".local"/"state"))
    folder = base / "roc_mcs" / "logs"; folder.mkdir(parents=True, exist_ok=True); return folder / "trs_apps_startup.log"
def report_error() -> None:
    details=traceback.format_exc()
    try:
        with startup_log().open("a",encoding="utf-8") as f: f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}]\n{details}\n")
    except OSError: pass
    message="TRS Apps не удалось запустить. Подробности: "+str(startup_log())+"\n\n"+details[-1600:]
    if os.name=="nt":
        try: ctypes.windll.user32.MessageBoxW(None,message,"TRS Apps — ошибка запуска",0x10)
        except Exception: pass
    else: sys.stderr.write(message+"\n")
def _configured_environment_is_active() -> bool:
    """Не переключаться в .venv, если ярлык явно указал другое окружение."""
    for name in ("pythonw_path.txt", "python_path.txt"):
        cfg = project_root() / name
        try: lines = [line.strip() for line in cfg.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        except OSError: continue
        if not lines: continue
        try: configured = Path(lines[0]).resolve()
        except (OSError, ValueError): continue
        current = Path(sys.executable).resolve()
        if current == configured or current.parent == configured.parent and current.stem.lower().startswith("python") and configured.stem.lower().startswith("python"):
            return True
    return False

def _project_environment_is_active() -> bool:
    root = project_root(); current = Path(sys.executable).resolve()
    for name in (".venv314", ".venv", "venv"):
        env = root / name; expected = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        candidates = [expected, expected.with_name("pythonw.exe")] if os.name == "nt" else [expected]
        for candidate in candidates:
            try:
                if candidate.is_file() and current == candidate.resolve(): return True
            except OSError: pass
    return False

def bootstrap() -> None:
    root=project_root(); os.chdir(root); src=root/"src"
    if src.is_dir() and str(src) not in sys.path: sys.path.insert(0,str(src))
    if _configured_environment_is_active() or _project_environment_is_active(): return
    for name in (".venv314", ".venv", "venv"):
        envdir=root/name; expected=envdir/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        if expected.is_file() and Path(sys.executable).resolve().parent != expected.resolve().parent:
            executable=expected.with_name("pythonw.exe") if os.name=="nt" and expected.with_name("pythonw.exe").is_file() else expected
            os.execv(str(executable),[str(executable),str(Path(__file__).resolve()),*sys.argv[1:]])
def main() -> int:
    try:
        bootstrap()
        from PyQt6 import QtWidgets
        from roc_mcs.gui.components.icon_provider import set_icon_theme
        from roc_mcs.gui.trs_apps.app_host import TRSAppsWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv); app.setApplicationName("TRS Apps"); app.setOrganizationName("ROC-MCS"); set_icon_theme(app,"#d8e0ec")
        window=TRSAppsWindow(); window.show(); return app.exec()
    except SystemExit: raise
    except Exception: report_error(); return 1
if __name__=="__main__": raise SystemExit(main())
