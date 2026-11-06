from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PyQt6.QtWidgets import QApplication, QMessageBox

from roc_mcs.analysis.project import WorkflowProject
from roc_mcs.gui.processing_workbench.window import ProcessingWorkbenchWindow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ROC-MCS Processing Workbench")
    parser.add_argument("--h5", help="Исходный HDF5-файл")
    parser.add_argument("--scan", help="Ключ scan, например entry472")
    parser.add_argument("--project", help="Папка проекта или workflow.json для открытия")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        if args.project and (Path(args.project).is_file() or (Path(args.project) / "workflow.json").is_file()):
            project = WorkflowProject.open(args.project)
        else:
            if not args.h5 or not args.scan:
                raise ValueError("Для нового проекта обязательны --h5 и --scan.")
            h5_path = Path(args.h5).expanduser().resolve()
            project_dir = Path(args.project).expanduser() if args.project else (
                Path.home() / "Documents" / "ROC-MCS Analyses" /
                f"{h5_path.stem}_{args.scan}.roi-analysis"
            )
            project = WorkflowProject.create(project_dir, h5_path, args.scan)
        window = ProcessingWorkbenchWindow(project)
        window.show()
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, "Не удалось открыть Processing Workbench", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
