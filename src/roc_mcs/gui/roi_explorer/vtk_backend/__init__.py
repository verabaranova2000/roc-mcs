"""Direct-HDF5 VTK volume viewer backend for ROI Explorer.

This package is opt-in and does not replace or import the Plotly widgets.
"""
from .h5_volume_source import H5VolumeLoader, VolumePayload, SpikeReport
from .vtk_scene import VTKVolumeScene, TransferSettings, build_rectilinear_grid, roi_local_to_detector_matrix

__all__ = [
    "H5VolumeLoader", "VolumePayload", "SpikeReport",
    "VTKVolumeScene", "TransferSettings", "build_rectilinear_grid", "roi_local_to_detector_matrix",
]
