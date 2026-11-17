"""Наблюдение за системными ресурсами TRS Apps."""
from .telemetry import GPUMetric, GPUSnapshot, ResourceSnapshot, classify_memory, parse_nvidia_smi

__all__ = ["GPUMetric", "GPUSnapshot", "ResourceSnapshot", "classify_memory", "parse_nvidia_smi"]
