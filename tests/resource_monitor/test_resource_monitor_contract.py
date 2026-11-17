from __future__ import annotations
import ast
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUI = ROOT / "src" / "roc_mcs" / "gui"


def test_titlebar_places_monitor_next_to_global_settings():
    source = (GUI / "trs_apps" / "window_chrome.py").read_text(encoding="utf-8")
    assert 'self.btn_resources = ResourceMonitorButton(self)' in source
    assert source.index('row.addWidget(self.btn_resources)') < source.index('row.addWidget(self.btn_settings)')


def test_gpu_subprocess_is_isolated_in_worker_and_has_timeout():
    source = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
    assert len(calls) == 1
    assert any(keyword.arg == "timeout" for keyword in calls[0].keywords)
    assert "self._pool.setMaxThreadCount(1)" in source


def test_monitor_uses_low_frequency_nonblocking_timer_and_stops_on_close():
    service = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    host = (GUI / "trs_apps" / "app_host.py").read_text(encoding="utf-8")
    assert "max(1000, int(interval_ms))" in service
    assert "self._process.cpu_percent(None)" in service
    assert "self._resource_monitor.stop()" in host


def test_gpu_panel_does_not_claim_per_process_vram():
    panel = (GUI / "resource_monitor" / "panel.py").read_text(encoding="utf-8")
    assert "всего адаптера" in panel
    assert "WDDM" in panel


def test_three_icon_variants_are_valid_svg():
    icons = GUI / "resources" / "icons"
    for name in ("resource_monitor_gauge.svg", "resource_monitor_activity.svg", "resource_monitor_chip.svg"):
        root = ET.parse(icons / name).getroot()
        assert root.tag.endswith("svg")


def test_windows_private_commit_uses_private_usage_not_pagefile_alias():
    source = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert 'getattr(info, "private", None)' in source
    assert 'getattr(info, "pagefile", None)' not in source
    assert 'getattr(info, "peak_wset", 0)' in source


def test_gui_event_loop_delay_is_reported_and_labeled_as_estimate():
    source = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    panel = (GUI / "resource_monitor" / "panel.py").read_text(encoding="utf-8")
    assert "event_loop_delay_ms" in source
    assert "Задержка Qt event loop" in panel


def test_system_commit_is_read_from_windows_performance_info():
    source = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert "GetPerformanceInfo" in source
    assert "CommitLimit - CommitTotal" in source
    assert "GlobalMemoryStatusEx" not in source


def test_gpu_polling_is_active_only_while_popup_is_visible():
    panel = (GUI / "resource_monitor" / "panel.py").read_text(encoding="utf-8")
    assert "self._service.request_gpu_sample(); self._gpu_timer.start()" in panel
    assert "self._gpu_timer.stop()" in panel


def test_gpu_worker_is_retained_until_completion_and_not_spawned_concurrently():
    service = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert "self._gpu_worker = worker" in service
    assert "if self._closing or self._gpu_pending: return" in service
    assert "self._gpu_worker = None" in service


def test_monitor_uses_lightweight_process_memory_metrics():
    service = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert ".memory_info()" in service
    assert ".memory_full_info()" not in service


def test_panel_is_persistent_modeless_tool_window_with_manual_close():
    panel = (GUI / "resource_monitor" / "panel.py").read_text(encoding="utf-8")
    assert "WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint" in panel
    assert "WindowType.Popup" not in panel
    assert "setModal(False)" in panel
    assert "WindowModality.NonModal" in panel
    assert "WA_DeleteOnClose, False" in panel
    assert "WA_ShowWithoutActivating, True" in panel
    assert 'self.close_button.clicked.connect(self.close)' in panel
    assert "self._panel._user_moved = True" in panel


def test_monitor_button_never_toggles_panel_closed():
    host = (GUI / "trs_apps" / "app_host.py").read_text(encoding="utf-8")
    tree = ast.parse(host)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "open_resource_monitor")
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    names = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    assert "show" in names and "raise_" in names
    assert "hide" not in names and "close" not in names
    assert "panel.user_moved" in host


def test_scientific_entry_points_are_instrumented_without_manual_call_sites():
    files = {
        "hdf5": GUI / "roi_explorer" / "vtk_backend" / "h5_volume_source.py",
        "vtk": GUI / "roi_explorer" / "vtk_backend" / "vtk_scene.py",
        "fit": ROOT / "src" / "roc_mcs" / "fitting" / "pipeline.py",
    }
    sources = {name: path.read_text(encoding="utf-8") for name, path in files.items()}
    assert '@tracked_activity("HDF5 · чтение и подготовка объёма")' in sources["hdf5"]
    assert '@tracked_activity("VTK · построение/обновление объёмной сцены")' in sources["vtk"]
    assert "@tracked_activity(lambda *args, **kwargs:" in sources["fit"]
    assert "JAX" in sources["fit"] and "lmfit" in sources["fit"]
    jax = (ROOT / "src" / "roc_mcs" / "fitting" / "backends" / "jax_backend.py").read_text(encoding="utf-8")
    assert '@tracked_activity("JAX · пакетная подгонка")' in jax


def test_activity_updates_are_emitted_with_regular_resource_samples():
    service = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert "activityChanged = QtCore.pyqtSignal(object)" in service
    assert "self.activityChanged.emit(activity_status())" in service


def test_private_commit_and_peak_rss_use_current_psutil_fields_first():
    service = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert 'getattr(info, "private", None) if os.name == "nt" else None' in service
    assert 'getattr(info, "vms", None)' not in service
    assert 'getattr(info, "peak_rss", None)' in service
    assert 'getattr(info, "peak_wset", 0)' in service  # совместимость со старым psutil


def test_gpu_unavailable_is_not_reported_as_gpu_absence():
    panel = (GUI / "resource_monitor" / "panel.py").read_text(encoding="utf-8")
    telemetry = (GUI / "resource_monitor" / "service.py").read_text(encoding="utf-8")
    assert "не означает отсутствие GPU" in panel
    assert "nvidia-smi.exe не найден" in telemetry
    assert "GPU может присутствовать" in telemetry
    assert "find_nvidia_smi()" in telemetry
