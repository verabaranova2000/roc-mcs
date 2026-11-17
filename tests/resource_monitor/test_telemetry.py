from __future__ import annotations
import pytest
from roc_mcs.gui.resource_monitor.telemetry import GIB, classify_memory, format_gib, format_pct, parse_nvidia_smi


def test_classify_memory_uses_headroom_not_used_percent():
    assert classify_memory(8 * GIB, 16 * GIB) == "normal"
    assert classify_memory(2 * GIB, 16 * GIB) == "warning"
    assert classify_memory(GIB, 16 * GIB) == "critical"


@pytest.mark.parametrize("available,total", [(None, 8 * GIB), (1, None), (1, 0), (-1, 8 * GIB), (9 * GIB, 8 * GIB)])
def test_classify_memory_returns_unknown_for_invalid_input(available, total):
    assert classify_memory(available, total) == "unknown"


def test_parse_nvidia_smi_reads_multiple_adapters_and_exact_mib_units():
    output = '0, "NVIDIA RTX A4000", 47, 4096, 16384\n1, "NVIDIA RTX A2000", 12, 1024, 8192'
    devices = parse_nvidia_smi(output)
    assert len(devices) == 2
    assert (devices[0].index, devices[0].name, devices[0].utilization_pct) == (0, "NVIDIA RTX A4000", 47.0)
    assert devices[0].memory_used == 4096 * 1024**2
    assert devices[0].memory_total == 16384 * 1024**2
    assert devices[1].index == 1


def test_parse_nvidia_smi_keeps_unsupported_metrics_unknown():
    device, = parse_nvidia_smi('0, "GPU", N/A, N/A, N/A')
    assert device.utilization_pct is None
    assert device.memory_used is None and device.memory_total is None


@pytest.mark.parametrize("output", ["0,GPU,101,20,100", "0,GPU,10,200,100", "0,GPU,10,20", "not,csv"])
def test_parse_nvidia_smi_rejects_invalid_rows(output):
    with pytest.raises((ValueError, TypeError)):
        parse_nvidia_smi(output)


def test_formatters_do_not_fabricate_missing_values():
    assert format_gib(None) == "н/д"
    assert format_pct(None) == "н/д"
    assert format_gib(GIB) == "1.00 GiB"
    assert format_pct(99.6) == "100%"


def test_event_loop_delay_is_zero_before_first_sample_and_never_negative():
    from roc_mcs.gui.resource_monitor.telemetry import event_loop_delay_ms
    assert event_loop_delay_ms(10.0, None) == 0.0
    assert event_loop_delay_ms(9.0, 10.0) == 0.0
    assert event_loop_delay_ms(10.125, 10.0) == pytest.approx(125.0)
