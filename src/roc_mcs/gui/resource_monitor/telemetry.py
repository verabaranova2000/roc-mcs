"""Чистые модели и разбор показаний монитора ресурсов."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from io import StringIO

GIB = 1024 ** 3

@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    sampled_at: str
    total_ram: int | None
    available_ram: int | None
    commit_limit: int | None
    commit_available: int | None
    process_rss: int | None
    process_private_commit: int | None
    process_threads: int | None
    process_handles: int | None
    process_peak_rss: int | None
    system_cpu_pct: float | None
    process_cpu_pct: float | None
    gui_event_delay_ms: float | None
    gui_event_delay_peak_ms: float | None
    state: str
    error: str | None = None

@dataclass(frozen=True, slots=True)
class GPUMetric:
    index: int
    name: str
    utilization_pct: float | None
    memory_used: int | None
    memory_total: int | None

@dataclass(frozen=True, slots=True)
class GPUSnapshot:
    sampled_at: str
    devices: tuple[GPUMetric, ...]
    message: str = ""


def classify_memory(available: int | None, total: int | None) -> str:
    """Эвристическая индикация запаса RAM; не является допуском к загрузке."""
    if available is None or total is None or total <= 0 or available < 0 or available > total:
        return "unknown"
    warning = max(2 * GIB, int(total * 0.15))
    critical = max(1 * GIB, int(total * 0.08))
    if available <= critical: return "critical"
    if available <= warning: return "warning"
    return "normal"


def parse_nvidia_smi(output: str) -> tuple[GPUMetric, ...]:
    """Разбирает CSV nvidia-smi; N/A остаётся None, а не превращается в ноль."""
    result = []
    for row in csv.reader(StringIO(output.strip()), skipinitialspace=True):
        if not row: continue
        if len(row) != 5: raise ValueError(f"Ожидалось 5 полей nvidia-smi, получено {len(row)}.")
        index, name, utilization, used, total = (value.strip() for value in row)
        if int(index) < 0 or not name: raise ValueError("Некорректный индекс или имя GPU.")
        def number(value: str) -> float | None:
            if value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}: return None
            parsed = float(value)
            if not 0 <= parsed <= 100: raise ValueError("Некорректная загрузка GPU.")
            return parsed
        util = number(utilization)
        unavailable = {"N/A", "[N/A]", "NOT SUPPORTED"}
        mem_used = None if used.upper() in unavailable else int(float(used) * 1024 ** 2)
        mem_total = None if total.upper() in unavailable else int(float(total) * 1024 ** 2)
        if mem_used is not None and mem_total is not None and (mem_used < 0 or mem_total <= 0 or mem_used > mem_total):
            raise ValueError("Некорректный объём памяти GPU.")
        result.append(GPUMetric(int(index), name, util, mem_used, mem_total))
    return tuple(result)


def event_loop_delay_ms(now: float, expected: float | None) -> float:
    """Приблизительная задержка цикла событий Qt относительно срока таймера."""
    return 0.0 if expected is None else max(0.0, (now - expected) * 1000.0)


def format_gib(value: int | None) -> str:
    return "н/д" if value is None else f"{value / GIB:.2f} GiB"


def format_pct(value: float | None) -> str:
    return "н/д" if value is None else f"{value:.0f}%"
