"""Double-clickable launcher for the standalone VTK/HDF5 viewer."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_project() -> Path | None:
    here = Path(__file__).resolve().parent
    for root in (here, *here.parents):
        src = root / "src"
        marker = src / "roc_mcs" / "roi" / "h5_scan_repository.py"
        if marker.is_file():
            source = str(src)
            if source not in sys.path:
                sys.path.insert(0, source)
            return root
    return None


def main() -> int:
    # This launcher intentionally supports one Qt binding. Do not let an
    # unrelated environment value make VTK select PySide while the UI imports PyQt6.
    os.environ["QT_API"] = "pyqt6"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    root = _bootstrap_project()
    if root is None:
        print(
            "Не найден корень ROC-MCS (ожидается src/roc_mcs/roi/h5_scan_repository.py).\n"
            "Положите launch_roi_vtk_h5_viewer.py в корень checkout или используйте:\n"
            "python -m roc_mcs.gui.roi_explorer.vtk_backend.desktop_viewer --h5 FILE",
            file=sys.stderr,
        )
        return 2
    from roc_mcs.gui.roi_explorer.vtk_backend.desktop_viewer import main as viewer_main
    return viewer_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
