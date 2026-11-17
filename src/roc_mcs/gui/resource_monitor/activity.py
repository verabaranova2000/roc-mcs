"""Низкозатратный учёт длительных операций приложения."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from itertools import count
from threading import RLock, current_thread
from time import monotonic
from typing import Callable, Iterator

@dataclass(frozen=True, slots=True)
class ActivityItem:
    name: str
    elapsed_seconds: float
    thread_name: str

@dataclass(frozen=True, slots=True)
class ActivityStatus:
    active: tuple[ActivityItem, ...] = ()
    last_name: str | None = None
    last_duration_seconds: float | None = None
    last_thread_name: str | None = None

_lock = RLock()
_ids = count(1)
_active: dict[int, tuple[str, float, str]] = {}
_last: tuple[str, float, str] | None = None

@contextmanager
def activity(name: str) -> Iterator[None]:
    """Регистрирует одну операцию; запись удаляется даже при исключении."""
    global _last
    label = str(name).strip()
    if not label: raise ValueError("Имя операции не должно быть пустым.")
    started = monotonic(); thread_name = current_thread().name
    with _lock:
        token = next(_ids); _active[token] = (label, started, thread_name)
    try:
        yield
    finally:
        duration = max(0.0, monotonic() - started)
        with _lock:
            _active.pop(token, None); _last = (label, duration, thread_name)

def activity_status() -> ActivityStatus:
    now = monotonic()
    with _lock:
        active = tuple(ActivityItem(name, max(0.0, now - started), thread_name)
                       for name, started, thread_name in sorted(_active.values(), key=lambda item: item[1]))
        last = _last
    return ActivityStatus(active, *(last if last is not None else (None, None, None)))

def tracked_activity(label: str | Callable[..., str]):
    """Декоратор для публичной синхронной операции; не меняет результат функции."""
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            name = label(*args, **kwargs) if callable(label) else label
            with activity(name): return func(*args, **kwargs)
        return wrapped
    return decorate
