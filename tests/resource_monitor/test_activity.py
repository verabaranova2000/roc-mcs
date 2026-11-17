from __future__ import annotations
from threading import Event, Thread
import pytest
from roc_mcs.gui.resource_monitor.activity import activity, activity_status, tracked_activity


def test_activity_is_visible_while_running_and_cleared_afterward():
    with activity("JAX · подгонка"):
        item, = [item for item in activity_status().active if item.name == "JAX · подгонка"]
        assert item.elapsed_seconds >= 0
        assert item.thread_name
    status = activity_status()
    assert all(item.name != "JAX · подгонка" for item in status.active)
    assert status.last_name == "JAX · подгонка"
    assert status.last_duration_seconds >= 0


def test_activity_is_removed_and_duration_recorded_when_operation_fails():
    with pytest.raises(RuntimeError, match="test failure"):
        with activity("HDF5 · чтение"):
            raise RuntimeError("test failure")
    status = activity_status()
    assert all(item.name != "HDF5 · чтение" for item in status.active)
    assert status.last_name == "HDF5 · чтение"
    assert status.last_duration_seconds >= 0


def test_nested_activities_are_reported_independently():
    with activity("outer"):
        with activity("inner"):
            names = {item.name for item in activity_status().active}
            assert {"outer", "inner"}.issubset(names)


def test_activity_can_be_observed_from_another_thread():
    entered, release = Event(), Event()
    def worker():
        with activity("worker operation"):
            entered.set(); release.wait(2)
    thread = Thread(target=worker, name="acceptance-worker")
    thread.start()
    try:
        assert entered.wait(2)
        item, = [item for item in activity_status().active if item.name == "worker operation"]
        assert item.thread_name == "acceptance-worker"
    finally:
        release.set(); thread.join(2)
    assert not thread.is_alive()
    assert all(item.name != "worker operation" for item in activity_status().active)


def test_tracked_activity_preserves_result_signature_and_exception():
    @tracked_activity(lambda value: f"calculation:{value}")
    def calculate(value: int) -> int:
        return value * 3
    assert calculate(7) == 21
    assert calculate.__name__ == "calculate"
    assert calculate.__annotations__["return"] == "int"
    assert activity_status().last_name == "calculation:7"


def test_tracked_activity_rethrows_original_exception():
    @tracked_activity("failing operation")
    def fail():
        raise ValueError("original")
    with pytest.raises(ValueError, match="original"):
        fail()
    status = activity_status()
    assert status.last_name == "failing operation"
    assert all(item.name != "failing operation" for item in status.active)


def test_empty_activity_name_is_rejected_before_registering():
    with pytest.raises(ValueError, match="не должно быть пустым"):
        with activity("  "):
            pass
