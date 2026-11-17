"""Низкочастотный монитор RAM/CPU и фоновый опрос NVIDIA GPU."""
from __future__ import annotations
import ctypes
import os
import shutil
import subprocess
import time
from datetime import datetime
from PyQt6 import QtCore
from .telemetry import GPUSnapshot, ResourceSnapshot, classify_memory, event_loop_delay_ms, parse_nvidia_smi


def _windows_commit_status() -> tuple[int | None, int | None]:
    """Системный commit headroom: (CommitLimit - CommitTotal) * PageSize."""
    if os.name != "nt": return None, None
    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]
    info = PERFORMANCE_INFORMATION(); info.cb = ctypes.sizeof(info)
    try:
        api = ctypes.WinDLL("psapi", use_last_error=True); query = api.GetPerformanceInfo
        query.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), ctypes.c_ulong]; query.restype = ctypes.c_int
        if not query(ctypes.byref(info), info.cb) or not info.PageSize: return None, None
        return int(info.CommitLimit * info.PageSize), max(0, int((info.CommitLimit - info.CommitTotal) * info.PageSize))
    except Exception: return None, None


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)


class _NvidiaQuery(QtCore.QRunnable):
    def __init__(self):
        super().__init__(); self.signals = _WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        now = datetime.now().strftime("%H:%M:%S")
        executable = shutil.which("nvidia-smi")
        if not executable:
            self.signals.finished.emit(GPUSnapshot(now, (), "nvidia-smi не найден; GPU не измеряется")); return
        command = [executable, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1.5, check=False, creationflags=flags)
            if completed.returncode != 0:
                message = (completed.stderr or f"nvidia-smi завершился с кодом {completed.returncode}").strip()[:180]
                snapshot = GPUSnapshot(now, (), message)
            else:
                snapshot = GPUSnapshot(now, parse_nvidia_smi(completed.stdout), "" if completed.stdout.strip() else "GPU не обнаружен")
        except subprocess.TimeoutExpired:
            snapshot = GPUSnapshot(now, (), "Опрос GPU превысил 1,5 с; показание пропущено")
        except Exception as exc:
            snapshot = GPUSnapshot(now, (), f"Опрос GPU недоступен: {type(exc).__name__}: {exc}"[:180])
        self.signals.finished.emit(snapshot)


class ResourceMonitorService(QtCore.QObject):
    snapshotChanged = QtCore.pyqtSignal(object)
    gpuChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, interval_ms: int = 2000):
        super().__init__(parent); self._psutil = None; self._process = None; self._peak_rss = 0; self._gpu_pending = False; self._gpu_worker = None; self._closing = False
        self._interval_s = max(1.0, int(interval_ms) / 1000.0); self._expected_tick = None; self._gui_delay_peak_ms = 0.0
        self._pool = QtCore.QThreadPool(self); self._pool.setMaxThreadCount(1); self._pool.setExpiryTimeout(10000)
        try:
            import psutil
            self._psutil = psutil; self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(None); psutil.cpu_percent(None)
        except Exception:
            self._psutil = None; self._process = None
        self._timer = QtCore.QTimer(self); self._timer.setInterval(max(1000, int(interval_ms))); self._timer.timeout.connect(self.sample_now); self._timer.start()

    @QtCore.pyqtSlot()
    def sample_now(self):
        now = datetime.now().strftime("%H:%M:%S")
        monotonic_now = time.monotonic(); gui_delay = event_loop_delay_ms(monotonic_now, self._expected_tick)
        self._expected_tick = monotonic_now + self._interval_s; self._gui_delay_peak_ms = max(self._gui_delay_peak_ms, gui_delay)
        if self._psutil is None or self._process is None:
            self.snapshotChanged.emit(ResourceSnapshot(sampled_at=now, total_ram=None, available_ram=None, commit_limit=None, commit_available=None,
                process_rss=None, process_private_commit=None, process_threads=None, process_handles=None, process_peak_rss=None,
                system_cpu_pct=None, process_cpu_pct=None, gui_event_delay_ms=gui_delay, gui_event_delay_peak_ms=self._gui_delay_peak_ms,
                state="unknown", error="psutil не установлен или недоступен")); return
        try:
            vm = self._psutil.virtual_memory(); info = self._process.memory_info()
            rss = int(info.rss); self._peak_rss = max(self._peak_rss, rss, int(getattr(info, "peak_wset", 0) or 0))
            private = getattr(info, "private", None); commit_limit, commit_available = _windows_commit_status()
            threads = int(self._process.num_threads()); handles = int(self._process.num_handles()) if hasattr(self._process, "num_handles") else None
            ram_state = classify_memory(int(vm.available), int(vm.total)); commit_state = classify_memory(commit_available, commit_limit)
            severity = {"unknown": 0, "normal": 1, "warning": 2, "critical": 3}
            state = max((ram_state, commit_state), key=severity.get)
            snapshot = ResourceSnapshot(sampled_at=now, total_ram=int(vm.total), available_ram=int(vm.available), commit_limit=commit_limit,
                commit_available=commit_available, process_rss=rss, process_private_commit=int(private) if private is not None else None,
                process_threads=threads, process_handles=handles, process_peak_rss=self._peak_rss,
                system_cpu_pct=float(self._psutil.cpu_percent(None)), process_cpu_pct=float(self._process.cpu_percent(None)),
                gui_event_delay_ms=gui_delay, gui_event_delay_peak_ms=self._gui_delay_peak_ms, state=state)
        except Exception as exc:
            snapshot = ResourceSnapshot(sampled_at=now, total_ram=None, available_ram=None, commit_limit=None, commit_available=None,
                process_rss=None, process_private_commit=None, process_threads=None, process_handles=None, process_peak_rss=self._peak_rss or None,
                system_cpu_pct=None, process_cpu_pct=None, gui_event_delay_ms=gui_delay, gui_event_delay_peak_ms=self._gui_delay_peak_ms,
                state="unknown", error=f"Сбор RAM/CPU: {type(exc).__name__}: {exc}"[:180])
        self.snapshotChanged.emit(snapshot)

    @QtCore.pyqtSlot()
    def request_gpu_sample(self):
        if self._closing or self._gpu_pending: return
        self._gpu_pending = True; worker = _NvidiaQuery(); self._gpu_worker = worker; worker.signals.finished.connect(self._on_gpu_finished); self._pool.start(worker)

    @QtCore.pyqtSlot(object)
    def _on_gpu_finished(self, snapshot):
        self._gpu_pending = False; self._gpu_worker = None
        if not self._closing: self.gpuChanged.emit(snapshot)

    def stop(self):
        self._closing = True; self._timer.stop(); self._pool.clear()
