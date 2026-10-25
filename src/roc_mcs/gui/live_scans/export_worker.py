"""Запускает pipeline/build.py в отдельном процессе, не блокируя Qt и matplotlib."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
from PyQt6 import QtCore

class PipelineExportWorker(QtCore.QThread):
    succeeded = QtCore.pyqtSignal(str, list)
    failed = QtCore.pyqtSignal(str)
    def __init__(self, request: dict, parent=None):
        super().__init__(parent); self.request = dict(request)
    def run(self):
        request_path = None
        try:
            project_root = Path(__file__).resolve().parents[4]
            cli = Path(__file__).with_name("export_cli.py")
            python = Path(sys.executable)
            if python.name.lower() == "pythonw.exe" and python.with_name("python.exe").is_file(): python = python.with_name("python.exe")
            fd, request_path = tempfile.mkstemp(prefix="roc_mcs_export_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream: json.dump(self.request, stream, ensure_ascii=False, allow_nan=False)
            env = os.environ.copy(); src = str(project_root / "src")
            env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["MPLBACKEND"] = "Agg"
            result = subprocess.run([str(python), str(cli), request_path], cwd=str(project_root), env=env,
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            if result.returncode != 0: raise RuntimeError((result.stderr or result.stdout or f"Процесс экспорта завершился с кодом {result.returncode}")[-12000:])
            data = json.loads(result.stdout.strip().splitlines()[-1])
            self.succeeded.emit(data["output_folder"], list(data.get("files", [])))
        except Exception as exc: self.failed.emit(str(exc))
        finally:
            if request_path:
                try: os.unlink(request_path)
                except OSError: pass
