#!/usr/bin/env python3
"""Native Qt/VTK render isolation for ROI Explorer HDF5 viewer.

Run from the roc-mcs checkout. Each case runs in a fresh child process so a
Windows native access violation (0xC0000005) is captured without killing the
whole test suite. The script does not modify HDF5, NPZ, or application files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

DEFAULT_H5 = r"C:\Users\Baranova Vera\Desktop\For_Sardana\TPC_data.h5"
CASES = ("empty", "volume-only", "volume-plus-projection", "app-layout")


def _log(message: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    print(f"[VTK-ISOLATION {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}" +
          (f" | {details}" if details else ""), flush=True)


def _load_payload(args):
    root = Path(args.repo).expanduser().resolve()
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Не найдена папка src: {src}")
    sys.path.insert(0, str(src))

    from roc_mcs.gui.roi_explorer.vtk_backend.h5_volume_source import H5VolumeLoader
    from roc_mcs.roi.roi_types import ROI

    loader = H5VolumeLoader(args.h5)
    if args.scan not in loader.scan_keys:
        raise KeyError(f"Scan {args.scan!r} отсутствует. Доступно: {loader.scan_keys}")
    roi = ROI(
        cx=args.cx, cy=args.cy, width=args.width, height=args.height,
        angle=args.angle, label="Native render isolation", type="rect",
    )
    payload = loader.load(
        args.scan, mode="roi", roi=roi,
        remove_spikes=False, smoothing_sigma=0.0,
    )
    _log(
        "PAYLOAD_READY", scan=payload.scan_key, shape=payload.shape,
        dtype=payload.scalar_dtype, finite=payload.finite_count,
        invalid=payload.invalid_count, coord=payload.coord_system,
    )
    return payload


def _child(args) -> int:
    root = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(root / "src"))

    # Load the same payload as the desktop viewer, before constructing Qt/VTK.
    payload = None if args.case == "empty" else _load_payload(args)

    from PyQt6 import QtCore, QtWidgets
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.vtkCommonCore import vtkVersion
    import vtk
    from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (
        VTKVolumeScene, TransferSettings,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["vtk-h5-render-isolation"])
    app.setQuitOnLastWindowClosed(False)

    class TracedQVTK(QVTKRenderWindowInteractor):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.paint_count = 0
            self.paint_stage = "pre-init"

        def paintEvent(self, event):
            next_count = self.paint_count + 1
            stage = self.paint_stage
            _log("PAINT_BEGIN", case=args.case, stage=stage, count=next_count,
                 size=(self.width(), self.height()))
            # A native access violation cannot be caught by Python; the parent
            # process records its Windows exit status and retains this log.
            super().paintEvent(event)
            self.paint_count = next_count
            _log("PAINT_END", case=args.case, stage=stage, count=next_count)

    main = QtWidgets.QMainWindow()
    main.setWindowTitle(f"VTK render isolation: {args.case}")
    if args.case == "app-layout":
        main.resize(1500, 920)
    else:
        main.resize(1040, 820)

    central = QtWidgets.QWidget(main)
    root_layout = QtWidgets.QHBoxLayout(central)
    root_layout.setContentsMargins(0, 0, 0, 0)
    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
    panel = QtWidgets.QWidget(splitter)
    panel.setMinimumWidth(310)
    panel_layout = QtWidgets.QVBoxLayout(panel)
    panel_layout.addWidget(QtWidgets.QLabel("Native render isolation"))
    panel_layout.addStretch(1)
    viewer = QtWidgets.QWidget(splitter)
    viewer_layout = QtWidgets.QVBoxLayout(viewer)
    viewer_layout.setContentsMargins(0, 0, 0, 0)
    widget = TracedQVTK(viewer)
    viewer_layout.addWidget(widget, 1)
    splitter.addWidget(panel)
    splitter.addWidget(viewer)
    splitter.setStretchFactor(1, 1)
    root_layout.addWidget(splitter)
    main.setCentralWidget(central)

    state: dict[str, Any] = {"scene": None, "error": None}

    def initialize_and_render():
        try:
            rw = widget.GetRenderWindow()
            rw.SetMultiSamples(0)
            created = widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_WState_Created)
            _log(
                "WIDGET_STATE", case=args.case, visible=widget.isVisible(),
                created=created, size=(widget.width(), widget.height()),
                render_window=rw.GetClassName(), multisamples=rw.GetMultiSamples(),
                vtk_version=vtkVersion.GetVTKVersion(), qt_version=QtCore.qVersion(),
                qt_base=__import__("vtkmodules.qt", fromlist=["QVTKRWIBase"]).QVTKRWIBase,
            )
            if not widget.isVisible() or not created:
                raise RuntimeError("QVTK не имеет видимого созданного native widget resource.")

            # Match the earlier passing diagnostic: two completed pre-init
            # paints, then Initialize(). Unlike the old app logic, if Qt has
            # delivered only one paint we explicitly force the second repaint.
            if widget.paint_count < 2:
                _log("FORCE_SECOND_PREINIT_PAINT", current=widget.paint_count)
                widget.repaint()
            if widget.paint_count < 2:
                raise RuntimeError(
                    f"Не удалось получить второй pre-init paintEvent; получено {widget.paint_count}."
                )

            interactor = rw.GetInteractor()
            if interactor is None:
                raise RuntimeError("У QVTK отсутствует vtkGenericRenderWindowInteractor.")
            if not bool(interactor.GetInitialized()):
                _log("INITIALIZE_BEGIN", interactor=interactor.GetClassName())
                widget.Initialize()
            if not bool(interactor.GetInitialized()):
                raise RuntimeError("GetInitialized() остался False после Initialize().")
            _log("INITIALIZE_DONE", enabled=interactor.GetEnabled(),
                 initialized=interactor.GetInitialized())

            scene = VTKVolumeScene(rw, interactor, render_callback=widget.Render)
            state["scene"] = scene
            _log("SCENE_CREATED")

            if payload is not None:
                show_projection = args.case != "volume-only"
                settings = TransferSettings(show_projection=show_projection)
                info = scene.set_payload(payload, settings)
                widget.paint_stage = "payload-set"
                _log(
                    "SET_PAYLOAD_DONE", case=args.case,
                    points=info.get("point_count"),
                    projection_visible=show_projection,
                    projection_actor=(scene.projection_actor is not None),
                    viewport=(widget.width(), widget.height()),
                )
            else:
                widget.paint_stage = "empty-scene"
                widget.Render()
                _log("EMPTY_RENDER_REQUESTED")

            def request_second_render():
                _log("SECOND_RENDER_REQUEST", case=args.case)
                if state["scene"] is not None:
                    state["scene"]._request_render()
                else:
                    widget.Render()

            QtCore.QTimer.singleShot(900, request_second_render)
        except BaseException as exc:
            state["error"] = exc
            traceback.print_exc()
            _log("CHILD_PYTHON_ERROR", error=f"{type(exc).__name__}: {exc}")
            app.exit(23)

    main.show()
    QtCore.QTimer.singleShot(150, initialize_and_render)
    QtCore.QTimer.singleShot(args.duration_ms, app.quit)
    _log("EVENT_LOOP_ENTER", case=args.case)
    rc = int(app.exec())
    _log("EVENT_LOOP_EXIT", case=args.case, rc=rc,
         paint_count=widget.paint_count, scene_created=state["scene"] is not None)
    if state["scene"] is not None:
        try:
            state["scene"].close()
        except Exception:
            traceback.print_exc()
    main.close()
    return 23 if state["error"] is not None else rc


def _status_from_returncode(rc: int) -> str:
    unsigned = rc & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        return "ACCESS_VIOLATION_0xC0000005"
    if rc == 0:
        return "PASS"
    return f"EXIT_0x{unsigned:08X}"


def _suite(args) -> int:
    root = Path(args.repo).expanduser().resolve()
    script = Path(__file__).resolve()
    log_dir = root / "vtk_h5_render_isolation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["QT_API"] = "pyqt6"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"

    results = []
    for case in CASES:
        log_path = log_dir / f"{case}.log"
        cmd = [
            sys.executable, "-X", "faulthandler", "-u", str(script),
            "--case", case, "--repo", str(root), "--h5", str(args.h5),
            "--scan", args.scan, "--cx", str(args.cx), "--cy", str(args.cy),
            "--width", str(args.width), "--height", str(args.height),
            "--angle", str(args.angle), "--duration-ms", str(args.duration_ms),
        ]
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(root), env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                rc = proc.wait(timeout=args.case_timeout)
                status = _status_from_returncode(rc)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                status = "TIMEOUT"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if status == "PASS" and "PAINT_END" not in text:
            status = "NO_COMPLETED_PAINT"
        results.append((case, status, log_path))
        print(f"{case:24s} {status:32s} log={log_path}", flush=True)
        if status != "PASS":
            print(text[-10000:], flush=True)

    print("SUMMARY=" + ", ".join(f"{case}:{status}" for case, status, _ in results), flush=True)
    print("LOG_DIR=" + str(log_dir), flush=True)
    return 0 if all(status == "PASS" for _, status, _ in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--scan", default="entry472")
    parser.add_argument("--cx", type=float, default=127.5)
    parser.add_argument("--cy", type=float, default=127.5)
    parser.add_argument("--width", type=float, default=89.6)
    parser.add_argument("--height", type=float, default=89.6)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--duration-ms", type=int, default=3500)
    parser.add_argument("--case-timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.case:
        try:
            return _child(args)
        except BaseException:
            traceback.print_exc()
            return 2
    return _suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
