from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

GIB = 1024 ** 3


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError as exc:
        raise SystemExit("Нужен psutil: python -m pip install psutil") from exc


def _scan_meta(h5_path: Path, scan_key: str) -> dict[str, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Нужны h5py и numpy в выбранном Python.") from exc
    with h5py.File(h5_path, "r") as h5:
        if scan_key not in h5:
            raise ValueError(f"В HDF5 нет группы {scan_key!r}.")
        group = h5[scan_key]
        candidates = ((f"{scan_key}/measurement/Frame", f"{scan_key}/measurement/th"),
                      (f"{scan_key}/raw_data_2D", f"{scan_key}/raw_data_th"))
        paths = next(((d, t) for d, t in candidates if d in h5 and t in h5), None)
        if paths is None:
            raise ValueError(f"Для {scan_key!r} не найден поддерживаемый набор Frame/th или raw_data_2D/raw_data_th.")
        dataset = h5[paths[0]]
        theta = np.asarray(h5[paths[1]]).squeeze()
        if dataset.ndim != 3 or theta.ndim != 1 or theta.size < 2:
            raise ValueError(f"Неподдерживаемая форма: data={dataset.shape}, theta={theta.shape}.")
        angle_axes = [i for i in (0, 2) if dataset.shape[i] == theta.size]
        if len(angle_axes) != 1:
            raise ValueError(f"Ось θ неоднозначна или отсутствует: shape={dataset.shape}, len(theta)={theta.size}.")
        if dataset.dtype.kind not in "iuf" or dataset.dtype.kind == "c":
            raise TypeError(f"Нужен вещественный числовой HDF5 dtype, получен {dataset.dtype}.")
        n = int(np.prod(dataset.shape, dtype=np.int64))
        itemsize = int(dataset.dtype.itemsize)
        if dataset.dtype == np.dtype("float32") or (dataset.dtype.kind in "iu" and itemsize <= 2):
            display_itemsize = 4
        elif dataset.dtype == np.dtype("float64") or itemsize > 2:
            display_itemsize = 8
        else:
            display_itemsize = 8
        source_bytes = n * itemsize
        return {"h5_path": str(h5_path.resolve()), "scan": scan_key, "dataset": paths[0],
                "theta_dataset": paths[1], "shape": tuple(int(x) for x in dataset.shape),
                "dtype": str(dataset.dtype), "dtype_bytes": itemsize, "chunks": dataset.chunks,
                "compression": dataset.compression, "theta_count": int(theta.size),
                "theta_axis": int(angle_axes[0]), "voxels": n, "source_bytes": source_bytes,
                "display_dtype_bytes": display_itemsize, "display_volume_bytes": n * display_itemsize,
                "source_gib": source_bytes / GIB, "display_volume_gib": n * display_itemsize / GIB,
                "file_size_bytes": h5_path.stat().st_size}


def _gpu_rows() -> list[dict[str, Any]]:
    exe = "nvidia-smi.exe" if os.name == "nt" else "nvidia-smi"
    try:
        result = subprocess.run([exe, "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            continue
        try:
            rows.append({"index": int(row[0]), "name": row[1], "memory_total_mib": int(row[2]),
                         "memory_used_mib": int(row[3]), "memory_free_mib": int(row[4]),
                         "utilization_gpu_pct": int(row[5])})
        except ValueError:
            continue
    return rows


def _suggest_budgets(total_bytes: int, available_bytes: int) -> tuple[float, float]:
    total_gib, available_gib = total_bytes / GIB, available_bytes / GIB
    reserve_gib = max(6.0, 0.35 * total_gib)
    process_cap_gib = max(0.0, min(0.25 * total_gib, available_gib - reserve_gib))
    return reserve_gib, process_cap_gib


def _viewer_estimate_bytes(meta: dict[str, Any], remove_spikes: bool = False) -> int:
    return int(meta["source_bytes"] + meta["voxels"] * (32 + (64 if remove_spikes else 0)))


def _system_info() -> dict[str, Any]:
    psutil = _psutil()
    vm = psutil.virtual_memory()
    total_gib, available_gib = vm.total / GIB, vm.available / GIB
    reserve_gib, process_cap_gib = _suggest_budgets(vm.total, vm.available)
    return {"time": datetime.now().astimezone().isoformat(), "os": platform.platform(),
            "python": sys.version, "python_executable": sys.executable,
            "cpu_logical": psutil.cpu_count(), "ram_total_gib": round(total_gib, 2),
            "ram_available_gib": round(available_gib, 2), "ram_percent_used": float(vm.percent),
            "suggested_system_reserve_gib": round(reserve_gib, 2),
            "suggested_process_cap_gib": round(process_cap_gib, 2),
            "budget_note": "Осторожный стартовый бюджет приёмочного испытания, а не универсальная норма и не аппаратная гарантия. Увеличьте резерв, если на ПК работают приборы/основная программа.",
            "gpu_nvidia": _gpu_rows()}


def _window_probe(pid: int) -> tuple[float | None, bool]:
    if os.name != "nt":
        return None, False
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.SendMessageTimeoutW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                           wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t))
    user32.SendMessageTimeoutW.restype = ctypes.c_size_t
    matches: list[tuple[int, int]] = []
    enum_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == pid and user32.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
                matches.append((area, int(hwnd)))
        return True

    user32.EnumWindows(enum_type(visit), 0)
    if not matches:
        return None, False
    hwnd = wintypes.HWND(max(matches)[1])
    result = ctypes.c_size_t()
    started = time.perf_counter()
    ok = user32.SendMessageTimeoutW(hwnd, 0, 0, 0, 0x0002, 500, ctypes.byref(result))
    elapsed = (time.perf_counter() - started) * 1000.0
    return (elapsed if ok else None), True


def _process_tree(root):
    try:
        return [root, *root.children(recursive=True)]
    except Exception:
        return [root]


def _tree_metrics(root) -> dict[str, Any]:
    members = _process_tree(root)
    rss = private = vms = cpu = threads = handles = 0
    private_known = True
    for process in members:
        try:
            mem = process.memory_info()
            rss += int(mem.rss)
            vms += int(mem.vms)
            if hasattr(mem, "private"):
                private += int(mem.private)
            else:
                private_known = False
            cpu += float(process.cpu_percent(interval=None))
            threads += int(process.num_threads())
            handles += int(process.num_handles()) if hasattr(process, "num_handles") else 0
        except Exception:
            continue
    return {"process_count": len(members), "rss_gib": rss / GIB,
            "private_gib": (private / GIB if private_known else None),
            "budget_memory_gib": (private / GIB if private_known else rss / GIB),
            "vms_gib": vms / GIB, "cpu_pct_total": cpu,
            "threads": threads, "handles": handles}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cmd_system(_args) -> int:
    info = _system_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def cmd_preflight(args) -> int:
    import h5py
    import numpy as np
    meta = _scan_meta(args.h5, args.scan)
    n = meta["voxels"]
    source_bytes = meta["source_bytes"]
    estimate = _viewer_estimate_bytes(meta, args.remove_spikes)
    meta["current_viewer_preflight_gib"] = estimate / GIB
    meta["remove_spikes_requested"] = bool(args.remove_spikes)
    meta["mode"] = args.mode
    meta["roi_note"] = "ROI не уменьшает исходное чтение: текущий loader сначала вызывает load_scan() для полного скана." if args.mode == "roi" else "Полный объём загружается целиком."
    meta["estimate_note"] = "Это формула текущего viewer, а не измеренный пик и не гарантия RAM/VRAM. Она не является жёстким ограничителем ОС."
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    return 0


def _terminate_tree(proc: subprocess.Popen, root, log) -> None:
    log({"event": "SAFE_STOP_REQUESTED", "reason": "resource/watchdog threshold"})
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=12)
    except Exception:
        pass
    for process in reversed(_process_tree(root)):
        try:
            process.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    for process in reversed(_process_tree(root)):
        try:
            process.kill()
        except Exception:
            pass


def cmd_monitor(args) -> int:
    if os.name != "nt":
        raise SystemExit("Этот приёмочный монитор рассчитан на Windows.")
    psutil = _psutil()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("После -- укажите команду запуска viewer.")
    h5_meta = _scan_meta(args.h5, args.scan)
    n = h5_meta["voxels"]
    estimate = _viewer_estimate_bytes(h5_meta, args.remove_spikes)
    system = _system_info()
    if system["ram_available_gib"] < args.min_system_available_gib + 1.0:
        raise SystemExit(f"Запуск отменён: доступно {system['ram_available_gib']:.2f} GiB RAM; "
                         f"для запуска нужно не менее {args.min_system_available_gib + 1.0:.2f} GiB "
                         "(заданный резерв плюс 1 GiB стартового запаса).")
    if estimate / GIB > args.max_process_gib:
        raise SystemExit(f"Запуск отменён: расчётная оценка viewer {estimate / GIB:.2f} GiB "
                         f"уже превышает установленный предел процесса {args.max_process_gib:.2f} GiB. "
                         "Оценка не заменяет измерения и не является точным пиком.")
    if args.min_gpu_free_mib is not None:
        rows = _gpu_rows()
        if not rows:
            raise SystemExit("Задан лимит VRAM, но nvidia-smi не смог получить данные. Для Intel/AMD/WDDM используйте системный монитор GPU вручную.")
        gpu = next((x for x in rows if x["index"] == args.gpu_index), None)
        if gpu is None or gpu["memory_free_mib"] < args.min_gpu_free_mib:
            raise SystemExit(f"Запуск отменён: GPU {args.gpu_index} не имеет требуемых {args.min_gpu_free_mib} MiB свободной памяти.")
    run_dir = args.log_dir.resolve() if args.log_dir else Path("logs/vtk_acceptance") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    console_path, csv_path = run_dir / "viewer_console.log", run_dir / "metrics.csv"
    events_path, summary_path = run_dir / "events.jsonl", run_dir / "summary.json"
    h5_before = args.h5.stat()
    started = datetime.now().astimezone()
    event_file = events_path.open("w", encoding="utf-8")

    def event(record):
        event_file.write(json.dumps({"time": datetime.now().astimezone().isoformat(), **record}, ensure_ascii=False, default=str) + "\n")
        event_file.flush()

    with console_path.open("w", encoding="utf-8", errors="replace", buffering=1) as console, csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fields = ["time", "elapsed_s", "rss_gib", "private_gib", "budget_memory_gib", "vms_gib", "system_available_gib",
                  "system_ram_pct", "cpu_pct_total", "threads", "handles", "process_count", "gpu_name", "gpu_total_mib",
                  "gpu_used_mib", "gpu_free_mib", "gpu_pct", "gui_probe_ms", "gui_probe_failed"]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
        proc = subprocess.Popen(command, cwd=args.cwd, stdout=console, stderr=subprocess.STDOUT,
                                creationflags=flags, close_fds=True)
        root = psutil.Process(proc.pid)
        root.cpu_percent(None)
        event({"event": "START", "pid": proc.pid, "command": command, "cwd": str(args.cwd),
               "h5": str(args.h5.resolve()), "scan": args.scan, "h5_meta": h5_meta,
               "current_viewer_preflight_gib": estimate / GIB, "limits": vars(args)})
        print(f"PID={proc.pid}; logs: {run_dir}")
        print("Приложение откроется отдельно. Проверяйте интерфейс, не запускайте другие тяжёлые задачи.")
        start_mono = time.monotonic()
        gpu_cache, last_gpu = [], 0.0
        low_ram_streak = high_mem_streak = low_gpu_streak = missing_gpu_streak = 0
        gui_fail_since = None
        stop_reason = None
        try:
            while proc.poll() is None:
                now = time.monotonic()
                elapsed = now - start_mono
                if elapsed > args.max_duration_min * 60:
                    stop_reason = "MAX_DURATION_EXCEEDED"
                try:
                    tree = _tree_metrics(root)
                except psutil.NoSuchProcess:
                    break
                vm = psutil.virtual_memory()
                available_gib = vm.available / GIB
                gui_probe_ms = None
                if now - last_gpu >= max(2.0, args.sample_seconds * 2):
                    gpu_cache, last_gpu = _gpu_rows(), now
                gpu = next((g for g in gpu_cache if g["index"] == args.gpu_index), None)
                gui_failed = False
                if args.gui_watchdog_seconds > 0:
                    gui_probe_ms, has_window = _window_probe(proc.pid)
                    gui_failed = has_window and gui_probe_ms is None
                    if gui_failed:
                        gui_fail_since = gui_fail_since or now
                    else:
                        gui_fail_since = None
                low_ram_streak = low_ram_streak + 1 if available_gib < args.min_system_available_gib else 0
                high_mem_streak = high_mem_streak + 1 if tree["budget_memory_gib"] > args.max_process_gib else 0
                low_gpu_streak = low_gpu_streak + 1 if (args.min_gpu_free_mib is not None and gpu and gpu["memory_free_mib"] < args.min_gpu_free_mib) else 0
                missing_gpu_streak = missing_gpu_streak + 1 if args.min_gpu_free_mib is not None and not gpu else 0
                row = {"time": datetime.now().astimezone().isoformat(), "elapsed_s": round(elapsed, 2), **tree,
                       "system_available_gib": round(available_gib, 3), "system_ram_pct": float(vm.percent),
                       "gpu_name": gpu["name"] if gpu else None, "gpu_total_mib": gpu["memory_total_mib"] if gpu else None,
                       "gpu_used_mib": gpu["memory_used_mib"] if gpu else None, "gpu_free_mib": gpu["memory_free_mib"] if gpu else None,
                       "gpu_pct": gpu["utilization_gpu_pct"] if gpu else None, "gui_probe_ms": round(gpu_probe_ms, 1) if gpu_probe_ms is not None else None,
                       "gui_probe_failed": gui_failed}
                writer.writerow({key: row.get(key) for key in fields})
                csv_file.flush()
                if low_ram_streak >= args.breach_samples:
                    stop_reason = "SYSTEM_AVAILABLE_RAM_BELOW_FLOOR"
                elif high_mem_streak >= args.breach_samples:
                    stop_reason = "PROCESS_MEMORY_LIMIT_EXCEEDED"
                elif low_gpu_streak >= args.breach_samples:
                    stop_reason = "GPU_FREE_MEMORY_BELOW_FLOOR"
                elif missing_gpu_streak >= args.breach_samples:
                    stop_reason = "GPU_METRICS_LOST"
                elif gui_fail_since and now - gui_fail_since >= args.gui_watchdog_seconds:
                    stop_reason = "GUI_UNRESPONSIVE"
                if stop_reason:
                    event({"event": "LIMIT_BREACH", "reason": stop_reason, "last_metrics": row})
                    _terminate_tree(proc, root, event)
                    break
                time.sleep(max(0.25, args.sample_seconds))
        except KeyboardInterrupt:
            stop_reason = "USER_INTERRUPTED"
            _terminate_tree(proc, root, event)
        if proc.poll() is None:
            _terminate_tree(proc, root, event)
        return_code = proc.poll()
    h5_after = args.h5.stat()
    source_unchanged = h5_before.st_size == h5_after.st_size and h5_before.st_mtime_ns == h5_after.st_mtime_ns
    summary = {"started": started.isoformat(), "finished": datetime.now().astimezone().isoformat(),
               "duration_s": round(time.monotonic() - start_mono, 2), "pid": proc.pid,
               "return_code": return_code, "status": "LIMIT_REACHED" if stop_reason else "PROCESS_EXITED",
               "stop_reason": stop_reason, "h5_stat_unchanged": source_unchanged,
               "h5_before": {"size": h5_before.st_size, "mtime_ns": h5_before.st_mtime_ns},
               "h5_after": {"size": h5_after.st_size, "mtime_ns": h5_after.st_mtime_ns},
               "max_process_gib": args.max_process_gib, "min_system_available_gib": args.min_system_available_gib,
               "max_estimated_gib_current_viewer_formula": estimate / GIB,
               "limitations": ["Пороговый монитор работает с периодом измерения; это не жёсткий лимит памяти ОС.",
                               "VRAM надёжно измеряется только если nvidia-smi отдаёт данные; WDDM/интегрированная GPU могут не показывать память процесса.",
                               "Успешное закрытие процесса не заменяет отдельную проверку научной калибровки."]}
    _write_json(summary_path, summary)
    event_file.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if stop_reason or not source_unchanged else (return_code or 0)


def build_parser():
    parser = argparse.ArgumentParser(description="Приёмочная диагностика VTK HDF5 viewer на Windows")
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("system", help="Снимок RAM/CPU/GPU до запуска")
    p.set_defaults(func=cmd_system)
    p = subs.add_parser("preflight", help="Чтение только HDF5 metadata; без загрузки полного массива")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--mode", choices=("roi", "full"), default="roi")
    p.add_argument("--remove-spikes", action="store_true")
    p.set_defaults(func=cmd_preflight)
    p = subs.add_parser("monitor", help="Запуск и мониторинг реального viewer")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--max-process-gib", type=float, required=True, help="Порог памяти процесса/дерева процессов")
    p.add_argument("--min-system-available-gib", type=float, required=True, help="Минимальный остаток доступной RAM всей системы")
    p.add_argument("--min-gpu-free-mib", type=int)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--remove-spikes", action="store_true")
    p.add_argument("--sample-seconds", type=float, default=1.0)
    p.add_argument("--breach-samples", type=int, default=3)
    p.add_argument("--max-duration-min", type=float, default=60.0)
    p.add_argument("--gui-watchdog-seconds", type=float, default=30.0, help="Остановить viewer при отсутствии ответа окна дольше порога; 0 отключает")
    p.add_argument("--log-dir", type=Path)
    p.add_argument("--cwd", type=Path, default=Path.cwd())
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_monitor)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if hasattr(parsed, "h5") and not parsed.h5.is_file():
        raise SystemExit(f"HDF5 не найден: {parsed.h5}")
    if parsed.action == "monitor":
        if parsed.max_process_gib <= 0 or parsed.min_system_available_gib < 0:
            raise SystemExit("Лимит процесса должен быть > 0 GiB, системный резерв — >= 0 GiB.")
        if parsed.sample_seconds <= 0 or parsed.breach_samples < 1 or parsed.max_duration_min <= 0:
            raise SystemExit("Интервал, число подтверждающих выборок и длительность должны быть положительными.")
        if parsed.min_gpu_free_mib is not None and parsed.min_gpu_free_mib < 0:
            raise SystemExit("Порог свободной VRAM должен быть >= 0 MiB.")
    raise SystemExit(parsed.func(parsed))
