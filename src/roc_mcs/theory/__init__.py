"""Reusable crystallographic and diffraction-theory models.

Pure theory lives here; workflow/GUI adapters are intentionally elsewhere.
"""
from .crystals import (
    CA_F2_FLUORITE, CrystalReflectionModel, CubicFccCrystal, XraylibCrystalModel,
    available_crystal_models, get_crystal_model, register_crystal_model, xraylib_available,
)
from .bragg import BraggDoubletConfig, calculate_doublet_reflections

__all__ = [
    "CA_F2_FLUORITE", "CrystalReflectionModel", "CubicFccCrystal", "XraylibCrystalModel",
    "available_crystal_models", "get_crystal_model", "register_crystal_model", "xraylib_available",
    "BraggDoubletConfig", "calculate_doublet_reflections",
]
