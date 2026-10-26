"""Автономная точка входа ROI Explorer; HDF5 можно передать первым аргументом."""
from __future__ import annotations
import ctypes, os, sys, traceback
from datetime import datetime
from pathlib import Path

def project_root() -> Path: return Path(__file__).resolve().parents[4]
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

def bootstrap() -> None:
    root=project_root(); os.chdir(root); src=root/"src"
    if src.is_dir() and str(src) not in sys.path: sys.path.insert(0,str(src))
    if _configured_environment_is_active(): return
    for name in (".venv","venv"):
        expected=root/name/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        if expected.is_file() and Path(sys.executable).resolve().parent!=expected.resolve().parent:
            executable=expected.with_name("pythonw.exe") if os.name=="nt" and expected.with_name("pythonw.exe").is_file() else expected
            os.execv(str(executable),[str(executable),str(Path(__file__).resolve()),*sys.argv[1:]])
def _error_log() -> Path:
    base=Path(os.environ.get("LOCALAPPDATA",Path.home())) if os.name=="nt" else Path(os.environ.get("XDG_STATE_HOME",Path.home()/".local"/"state"))
    folder=base/"roc_mcs"/"logs"; folder.mkdir(parents=True,exist_ok=True); return folder/"roi_explorer_startup.log"
def main() -> int:
    launch_cwd = Path.cwd()
    try:
        bootstrap()
        from PyQt6 import QtWidgets
        from roc_mcs.gui.components.icon_provider import set_icon_theme
        from roc_mcs.gui.roi_explorer.explorer_window import RockingCurveExplorerWindow
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv); app.setApplicationName("ROI Explorer"); app.setOrganizationName("ROC-MCS"); set_icon_theme(app,"#d8e0ec")
        path=Path(sys.argv[1]).expanduser() if len(sys.argv)>1 else None
        if path is not None and not path.is_absolute(): path=(launch_cwd/path).resolve()
        if path is None:
            start=Path.home()
            last=Path(os.environ.get("USERPROFILE",str(Path.home())))/"Desktop"/"For_Sardana"
            if last.is_dir(): start=last
            selected,_=QtWidgets.QFileDialog.getOpenFileName(None,"Открыть HDF5 со сканами 2D-детектора",str(start),"HDF5 (*.h5 *.hdf5 *.hdf);;Все файлы (*)")
            if not selected: return 0
            path=Path(selected)
        if not path.is_file(): raise FileNotFoundError(f"Файл не найден: {path}")
        window=RockingCurveExplorerWindow(path); window.show(); return app.exec()
    except SystemExit: raise
    except Exception:
        details=traceback.format_exc()
        try: _error_log().write_text(details,encoding="utf-8")
        except OSError: pass
        message="ROI Explorer не удалось запустить. Подробности: "+str(_error_log())+"\n\n"+details[-1600:]
        if os.name=="nt":
            try: ctypes.windll.user32.MessageBoxW(None,message,"ROI Explorer — ошибка запуска",0x10)
            except Exception: pass
        else: sys.stderr.write(message+"\n")
        return 1
if __name__=="__main__": raise SystemExit(main())
