"""VTK GPU volume scene for the ROI Explorer HDF5 adapter."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import logging

import numpy as np

from roc_mcs.gui.resource_monitor.activity import tracked_activity
from .display_palette import resolve_projection_palette
from .roi_preview_geometry import canonicalize_roi_preview


logger = logging.getLogger(__name__)

# vtkOpenGLVolumeRGBTable chooses a power-of-two 1D texture.  The user's
# current OpenGL driver reports a 16384-texel maximum; keep the transfer
# function's estimated requirement within that real hardware limit instead
# of allowing VTK to silently downsample a 32768-texel request.
_MAX_VOLUME_COLOR_TEXTURE_WIDTH = 16384
_MAX_VOLUME_OPACITY_TEXTURE_WIDTH = 16384
_MAX_VOLUME_PALETTE_CONTROL_POINTS = 64
_MIN_VOLUME_PALETTE_CONTROL_POINTS = 2
_INITIAL_VOLUME_OPACITY_CONTROL_POINTS = 17


def _rebase_vtk_mouse_move(interactor, previous_position: tuple[int, int] | None):
    """Set VTK's previous pointer sample before TrackballCamera handles a move.

    This must be called by the Qt QVTK adapter *after* SetEventInformation and
    immediately before MouseMoveEvent. While TrackballCamera has grabbed focus,
    VTK intentionally routes events exclusively to the style and bypasses
    ordinary interactor observers; doing this from an interactor MouseMove
    observer is therefore too late / not delivered at all.
    """
    if interactor is None or previous_position is None:
        return None
    try:
        style = interactor.GetInteractorStyle()
        if style is None or int(style.GetState()) == 0:
            return None
        current = tuple(map(int, interactor.GetEventPosition()))
        previous = tuple(map(int, previous_position))
        interactor.SetLastEventPosition(*previous)
        return current
    except Exception:
        logger.debug("Could not rebase VTK pointer sample before camera motion", exc_info=True)
        return None


def _stable_axis_corner_bits(
    camera_position: Any,
    ranges: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    previous: tuple[int, int, int] | None,
    *,
    hysteresis_fraction: float = 0.025,
) -> tuple[int, int, int]:
    """Choose camera-facing axis faces with a small center-plane hysteresis.

    Without hysteresis, a camera aligned exactly with an axis lies on the
    centre planes of the other two dimensions. Floating-point noise or the
    first sub-degree mouse movement can then flip the grid/labels between
    opposite faces, which looks like a camera snap although the camera pose is
    continuous. Keep the prior face until the camera crosses a scale-relative
    dead band; the physical volume and camera transform are not modified.
    """
    position = np.asarray(camera_position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all() or len(ranges) != 3:
        raise ValueError("Для выбора грани нужны конечные координаты камеры и три диапазона")
    fraction = float(hysteresis_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise ValueError("hysteresis_fraction должен лежать в [0, 0.5)")
    bits: list[int] = []
    for index, (low, high) in enumerate(ranges):
        low, high = float(low), float(high)
        center = 0.5 * (low + high)
        span = max(high - low, np.finfo(np.float64).eps * max(abs(low), abs(high), 1.0))
        offset = float(position[index] - center)
        if previous is None:
            bit = 1 if offset >= 0.0 else 0
        else:
            old = int(previous[index])
            margin = span * fraction
            if old == 1:
                bit = 0 if offset < -margin else 1
            else:
                bit = 1 if offset > margin else 0
        bits.append(bit)
    return tuple(bits)  # type: ignore[return-value]


def _vtk():
    try:
        # Import the OpenGL2 volume-rendering implementation explicitly.
        # This registers the concrete GPU mapper factory instead of relying on
        # incidental imports performed by a particular VTK distribution.
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError as exc:
        raise ImportError("Для 3D-рендерера требуется пакет vtk.") from exc
    return vtk, numpy_to_vtk


def build_rectilinear_grid(payload: Any):
    """Create an exact-coordinate vtkRectilinearGrid from volume[x,y,theta].

    VTK point IDs advance fastest along X. For a NumPy (x,y,z) array this
    corresponds to Fortran-order flattening; changing to C order would permute
    the physical scalar locations and is therefore explicitly avoided.
    """
    vtk, numpy_to_vtk = _vtk()
    volume = np.asarray(payload.volume)
    if not np.issubdtype(volume.dtype, np.number) or np.issubdtype(volume.dtype, np.complexfloating):
        raise TypeError(f"VTK volume должен иметь вещественный числовой dtype, получен {volume.dtype}.")
    axes = [np.asarray(payload.x), np.asarray(payload.y), np.asarray(payload.theta)]
    if volume.ndim != 3 or tuple(volume.shape) != tuple(len(a) for a in axes):
        raise ValueError(
            f"Несогласованная геометрия: volume={volume.shape}, "
            f"axis lengths={tuple(len(a) for a in axes)}."
        )
    if any(int(n) < 2 for n in volume.shape):
        raise ValueError(
            "Для корректного 3D volume rendering каждая размерность должна "
            f"содержать минимум 2 узла; получено {volume.shape}. Увеличьте ROI."
        )
    if any(a.ndim != 1 or not np.all(np.isfinite(a)) for a in axes):
        raise ValueError("Координатные оси должны быть конечными одномерными массивами.")
    for name, axis in zip(("x", "y", "theta"), axes):
        if axis.size > 1 and not np.all(np.diff(axis) > 0):
            raise ValueError(f"Ось {name} должна быть строго возрастающей.")

    finite = np.isfinite(volume)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        raise ValueError("Для VTK нет конечных скалярных значений.")
    # where= avoids materializing a second full-size finite_values array.
    value_min = float(np.min(volume, where=finite, initial=np.inf))
    value_max = float(np.max(volume, where=finite, initial=-np.inf))
    scale = max(abs(value_min), abs(value_max), 1.0)
    sentinel = value_min - 0.01 * scale
    if not np.isfinite(sentinel) or sentinel >= value_min:
        sentinel = float(np.nextafter(value_min, -np.inf))
    if not np.isfinite(sentinel):
        # At the most-negative representable value there is no finite lower
        # sentinel. Equal-to-min is still transparent because its opacity is 0.
        sentinel = value_min

    # Build exactly one Fortran-ordered renderer copy. VTK point IDs advance
    # fastest along X, so this layout preserves volume[x,y,theta] without an
    # intermediate np.where() array plus a second ravel copy.
    vtk_dtype = np.float32 if np.asarray(volume).dtype == np.float32 else np.float64
    renderer_copy = np.array(volume, dtype=vtk_dtype, order="F", copy=True)
    flat = renderer_copy.ravel(order="F")
    flat[~np.isfinite(flat)] = sentinel
    scalars = numpy_to_vtk(flat, deep=True)
    del flat, renderer_copy
    scalars.SetName("Intensity")

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    for values, setter in zip(axes, (grid.SetXCoordinates, grid.SetYCoordinates, grid.SetZCoordinates)):
        arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
        arr.SetNumberOfComponents(1)
        setter(arr)
    grid.GetPointData().SetScalars(scalars)
    grid.Modified()
    info = {
        "dimensions": tuple(int(v) for v in volume.shape),
        "point_count": int(volume.size),
        "finite_count": finite_count,
        "invalid_count": int(volume.size - finite_count),
        "scalar_min": value_min,
        "scalar_max": value_max,
        "invalid_sentinel": float(sentinel),
        "scalar_dtype": str(np.dtype(vtk_dtype)),
        "coordinate_bounds": tuple((float(a[0]), float(a[-1])) for a in axes),
    }
    return grid, info


def build_projection_grid(payload: Any):
    """Build a low-memory rectilinear XY projection plane.

    ``payload.projection`` is (x, y). VTK point IDs advance fastest along X,
    so point scalars are flattened in Fortran order just like the 3D volume.
    The plane is coplanar with the lower theta face. Keeping the projection
    attached to the volume (rather than offsetting it by an arbitrary fraction
    of the theta range) makes the geometry unambiguous and keeps it attached
    when the display-only theta scale changes.
    """
    vtk, numpy_to_vtk = _vtk()
    raw_projection = np.asarray(payload.projection)
    if not np.issubdtype(raw_projection.dtype, np.number) or np.issubdtype(raw_projection.dtype, np.complexfloating):
        raise TypeError(
            f"VTK projection должна иметь вещественный числовой dtype, получен {raw_projection.dtype}."
        )
    projection = np.asarray(raw_projection, dtype=np.float64)
    x = np.asarray(payload.x, dtype=np.float64)
    y = np.asarray(payload.y, dtype=np.float64)
    theta = np.asarray(payload.theta, dtype=np.float64)
    if projection.shape != (x.size, y.size):
        raise ValueError(
            f"2D-проекция должна иметь shape={(x.size, y.size)}, "
            f"получено {projection.shape}."
        )
    if x.size < 2 or y.size < 2 or theta.size < 2:
        raise ValueError("Для XY-проекции и 3D-объёма нужны минимум 2 узла по каждой оси.")
    for name, axis in (("x", x), ("y", y), ("theta", theta)):
        if axis.ndim != 1 or not np.all(np.isfinite(axis)):
            raise ValueError(f"Ось {name} должна быть конечным 1D-массивом.")
        if axis.size > 1 and not np.all(np.diff(axis) > 0):
            raise ValueError(f"Ось {name} должна быть строго возрастающей.")

    finite = np.isfinite(projection)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        raise ValueError("2D-проекция не содержит конечных значений.")
    value_min = float(np.min(projection, where=finite, initial=np.inf))
    value_max = float(np.max(projection, where=finite, initial=-np.inf))
    # A single hot/dead detector pixel must not collapse the contrast of the
    # entire XY image. Keep the real scalar range for diagnostics, but map the
    # 0.1..99.9 percentile display interval. Values outside it are clipped only
    # by the lookup table; the underlying projection remains untouched.
    finite_values = projection[finite]
    if value_min >= 0.0:
        display_min = 0.0
        positive = finite_values[finite_values > 0.0]
        display_max = float(np.percentile(positive, 99.9)) if positive.size else value_max
    else:
        display_min, display_max = map(float, np.percentile(finite_values, [0.1, 99.9]))
    if not np.isfinite(display_min):
        display_min = value_min
    if not np.isfinite(display_max) or display_max <= display_min:
        display_max = display_min + max(abs(display_min) * 1e-6, 1.0)
    scale = max(abs(value_min), abs(value_max), 1.0)
    sentinel = value_min - 0.01 * scale
    if not np.isfinite(sentinel) or sentinel >= value_min:
        sentinel = float(np.nextafter(value_min, -np.inf))
    if not np.isfinite(sentinel):
        sentinel = value_min

    # The XY projection is the detector image associated with the volume. It
    # must sit directly on the lower theta face, with no visual gap.
    z_floor = float(theta[0])

    scalars_np = np.array(projection, dtype=np.float64, order="F", copy=True)
    scalars_flat = scalars_np.ravel(order="F")
    scalars_flat[~finite.ravel(order="F")] = sentinel
    scalars = numpy_to_vtk(scalars_flat, deep=True)
    scalars.SetName("XY projection")

    # A rectilinear 2D grid stores only the coordinate axes. A StructuredGrid
    # would allocate an (Nx, Ny, 3) point array for a plane whose X/Y geometry
    # is already rectilinear; that is needlessly expensive for large detectors.
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(int(x.size), int(y.size), 1)
    for values, setter in (
        (x, grid.SetXCoordinates),
        (y, grid.SetYCoordinates),
        (np.asarray([z_floor], dtype=np.float64), grid.SetZCoordinates),
    ):
        arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
        arr.SetNumberOfComponents(1)
        setter(arr)
    grid.GetPointData().SetScalars(scalars)
    grid.Modified()
    return grid, {
        "dimensions": (int(x.size), int(y.size), 1),
        "point_count": int(projection.size),
        "finite_count": finite_count,
        "invalid_count": int(projection.size - finite_count),
        "scalar_min": value_min,
        "scalar_max": value_max,
        "display_scalar_min": display_min,
        "display_scalar_max": display_max,
        "invalid_sentinel": float(sentinel),
        "z_plane": z_floor,
    }


def roi_local_to_detector_matrix(roi: Any):
    """Affine VTK matrix for ROI extractor's local (u,v) grid.

    ROI.to_detector implements:
        x = cx + cos(a)*u - sin(a)*v
        y = cy + sin(a)*u + cos(a)*v
    Theta remains the world Z coordinate.
    """
    if roi is None:
        return None
    vtk, _ = _vtk()
    angle = np.deg2rad(float(roi.angle))
    c, s = float(np.cos(angle)), float(np.sin(angle))
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(0, 0, c)
    matrix.SetElement(0, 1, -s)
    matrix.SetElement(0, 3, float(roi.cx))
    matrix.SetElement(1, 0, s)
    matrix.SetElement(1, 1, c)
    matrix.SetElement(1, 3, float(roi.cy))
    return matrix


def _palette_samples(name: str, n: int = 48) -> np.ndarray:
    try:
        # Reuse the same palette catalogue/LUT provider as the existing 2D GUI.
        from roc_mcs.gui.roi_explorer.colormap_catalog import mpl_colormap
        cmap = mpl_colormap(name)
        return np.asarray(cmap(np.linspace(0.0, 1.0, n))[:, :3], dtype=float)
    except Exception:
        try:
            from matplotlib import colormaps
            cmap = colormaps.get_cmap(name)
            return np.asarray(cmap(np.linspace(0.0, 1.0, n))[:, :3], dtype=float)
        except Exception:
            pass
        # Small built-in fallback; normal app environment includes Matplotlib.
        presets = {
            "gray": ((0, 0, 0), (1, 1, 1)),
            "viridis": ((0.267, 0.005, 0.329), (0.128, 0.567, 0.551), (0.993, 0.906, 0.144)),
            "magma": ((0.001, 0.000, 0.014), (0.716, 0.215, 0.475), (0.987, 0.991, 0.750)),
            "inferno": ((0.001, 0.000, 0.014), (0.735, 0.216, 0.330), (0.988, 0.998, 0.645)),
            "plasma": ((0.050, 0.030, 0.528), (0.798, 0.280, 0.470), (0.940, 0.975, 0.131)),
            "cividis": ((0.000, 0.126, 0.302), (0.489, 0.485, 0.471), (0.996, 0.909, 0.218)),
            "turbo": ((0.190, 0.072, 0.232), (0.160, 0.735, 0.925), (0.980, 0.980, 0.130)),
            "gist_earth": ((0.000, 0.000, 0.000), (0.200, 0.350, 0.250), (0.650, 0.550, 0.400), (1.0, 1.0, 1.0)),
        }
        anchors = np.asarray(presets.get(name, presets["viridis"]), dtype=float)
        old = np.linspace(0, 1, len(anchors))
        new = np.linspace(0, 1, n)
        return np.column_stack([np.interp(new, old, anchors[:, i]) for i in range(3)])


def _set_adaptive_volume_palette(
    color_function: Any,
    palette: str,
    color_low: float,
    color_high: float,
    scalar_min: float,
    scalar_max: float,
    vtk_math: Any,
    *,
    max_texture_width: int = _MAX_VOLUME_COLOR_TEXTURE_WIDTH,
) -> dict[str, int]:
    """Populate a linear scalar palette without requesting an oversized GL LUT.

    The transfer function is intentionally defined over the robust display
    interval [color_low, color_high], while VTK estimates its 1D RGB texture
    over the complete data scalar range [scalar_min, scalar_max].  When rare
    outliers make that interval very small, a fixed 64-point palette can make
    VTK request 32768 texels even on a driver limited to 16384.  Reduce only
    the number of palette control points until VTK's own estimator fits the
    hardware-safe texture width.  Scalar values, thresholds, opacity and the
    linear mapping interval are not altered.
    """
    sample_count = int(_MAX_VOLUME_PALETTE_CONTROL_POINTS)
    estimated_samples = 1
    estimated_width = 1
    scalar_min = float(scalar_min)
    scalar_max = float(scalar_max)
    if not (np.isfinite(scalar_min) and np.isfinite(scalar_max)):
        raise ValueError("Границы скалярного диапазона для цветовой функции должны быть конечными.")

    while True:
        colors = _palette_samples(palette, sample_count)
        color_function.RemoveAllPoints()
        for i, color in enumerate(colors):
            value = float(color_low + (color_high - color_low) * (i / (sample_count - 1)))
            color_function.AddRGBPoint(value, *map(float, color))

        if scalar_max > scalar_min:
            estimated_samples = max(
                1,
                int(color_function.EstimateMinNumberOfSamples(scalar_min, scalar_max)),
            )
        else:
            estimated_samples = 1
        estimated_width = max(1, int(vtk_math.NearestPowerOfTwo(estimated_samples)))
        if (
            estimated_width <= int(max_texture_width)
            or sample_count <= _MIN_VOLUME_PALETTE_CONTROL_POINTS
        ):
            break
        sample_count = max(
            _MIN_VOLUME_PALETTE_CONTROL_POINTS,
            sample_count // 2,
        )

    if estimated_width > int(max_texture_width):
        # Do not hide an exceptional data/driver mismatch.  This is only
        # reachable when even a two-knot linear palette cannot fit the
        # renderer's scalar range into the known OpenGL texture limit.
        logger.warning(
            "VTK volume color transfer still estimates a %d-texel table "
            "(limit %d) for scalar range [%g, %g] and palette interval [%g, %g].",
            estimated_width,
            int(max_texture_width),
            scalar_min,
            scalar_max,
            float(color_low),
            float(color_high),
        )
    return {
        "control_points": int(sample_count),
        "estimated_samples": int(estimated_samples),
        "estimated_texture_width": int(estimated_width),
    }


def _set_adaptive_volume_opacity(
    opacity_function: Any,
    sentinel: float,
    scalar_min: float,
    scalar_max: float,
    opacity_low: float,
    opacity_high: float,
    max_opacity: float,
    gamma: float,
    vtk_math: Any,
    *,
    max_texture_width: int = _MAX_VOLUME_OPACITY_TEXTURE_WIDTH,
) -> dict[str, int]:
    """Keep the original gamma transfer curve and bake only the GPU-safe LUT.

    The normal 17-knot function is retained unchanged whenever it fits. If
    VTK estimates a larger power-of-two texture than the graphics driver
    supports, sample that exact function once at the driver's maximum 1D LUT
    resolution and rebuild it from those samples. This matches the fallback
    resolution VTK would otherwise use internally, but makes the resolution
    explicit and prevents a second, silent downsampling pass.
    """
    data_min = float(scalar_min)
    table_min = float(min(sentinel, data_min))
    scalar_max = float(scalar_max)
    opacity_low = float(opacity_low)
    opacity_high = float(opacity_high)
    max_opacity = float(max_opacity)
    gamma = float(gamma)
    if not all(np.isfinite(v) for v in (
        sentinel, data_min, scalar_max, opacity_low, opacity_high, max_opacity, gamma
    )):
        raise ValueError("Границы и параметры opacity transfer function должны быть конечными.")
    if scalar_max < data_min:
        raise ValueError("Некорректный скалярный диапазон opacity transfer function.")
    if opacity_high <= opacity_low:
        raise ValueError("Верхний порог opacity должен быть выше нижнего.")
    if not (0.0 <= max_opacity <= 1.0) or gamma <= 0.0:
        raise ValueError("max_opacity должен быть в [0,1], gamma — строго положительной.")

    opacity_function.RemoveAllPoints()
    opacity_function.AddPoint(float(sentinel), 0.0)
    opacity_function.AddPoint(float(data_min), 0.0)
    opacity_function.AddPoint(float(opacity_low), 0.0)
    for t in np.linspace(0.0, 1.0, _INITIAL_VOLUME_OPACITY_CONTROL_POINTS):
        value = opacity_low + (opacity_high - opacity_low) * float(t)
        alpha = max_opacity * (float(t) ** gamma)
        opacity_function.AddPoint(float(value), float(alpha))
    if scalar_max > opacity_high:
        opacity_function.AddPoint(float(scalar_max), max_opacity)

    if scalar_max > table_min:
        estimated_samples = max(
            1,
            int(opacity_function.EstimateMinNumberOfSamples(table_min, scalar_max)),
        )
    else:
        estimated_samples = 1
    estimated_width = max(1, int(vtk_math.NearestPowerOfTwo(estimated_samples)))
    original_estimated_width = int(estimated_width)
    original_control_points = int(opacity_function.GetSize())
    resampled = False

    if estimated_width > int(max_texture_width) and scalar_max > table_min:
        # vtkPiecewiseFunction.GetTable uses the same scalar interval and
        # uniform sampling basis required by the GPU mapper. Rebuilding from
        # these samples reproduces the driver's maximum-resolution fallback.
        table_size = int(max_texture_width)
        if table_size < 2:
            raise ValueError("Лимит текстуры opacity LUT должен быть не меньше 2.")
        table_values = [0.0] * table_size
        opacity_function.GetTable(table_min, scalar_max, table_size, table_values)
        opacity_function.BuildFunctionFromTable(
            table_min, scalar_max, table_size, table_values
        )
        resampled = True
        estimated_samples = max(
            1,
            int(opacity_function.EstimateMinNumberOfSamples(table_min, scalar_max)),
        )
        estimated_width = max(1, int(vtk_math.NearestPowerOfTwo(estimated_samples)))

    if estimated_width > int(max_texture_width):
        logger.warning(
            "VTK volume opacity transfer still estimates a %d-texel table "
            "(limit %d) for scalar range [%g, %g] and opacity interval [%g, %g].",
            estimated_width,
            int(max_texture_width),
            table_min,
            scalar_max,
            opacity_low,
            opacity_high,
        )
    elif resampled:
        logger.info(
            "VTK volume opacity transfer sampled at the GPU limit (%d texels; "
            "original estimate %d; resulting estimate %d; scalar range [%g, %g]).",
            int(max_texture_width),
            int(original_estimated_width),
            int(estimated_width),
            table_min,
            scalar_max,
        )

    return {
        "control_points": int(opacity_function.GetSize()),
        "original_control_points": int(original_control_points),
        "estimated_samples": int(estimated_samples),
        "estimated_texture_width": int(estimated_width),
        "resampled": int(resampled),
    }

@dataclass(slots=True)
class TransferSettings:
    # These are fractions of the robust p99.9 positive-intensity reference,
    # matching ROI_Explorer_3D_VTK_LAB and the Plotly explorer controls.
    lower_fraction: float = 0.16
    upper_fraction: float = 0.95
    max_opacity: float = 0.18
    gamma: float = 2.0
    palette: str = "viridis"
    sample_distance_factor: float = 1.0
    opacity_unit_distance_factor: float = 1.0
    show_projection: bool = True
    projection_opacity: float = 0.75
    # Backward-compatible inputs for the older diagnostic launcher. When both
    # are supplied, they explicitly request NumPy percentiles (0..100). The
    # desktop GUI does not use these legacy fields.
    lower_percentile: float | None = None
    upper_percentile: float | None = None

    # None keeps the historical theme-aware grayscale projection palette.
    # A named map is an explicit display-only override for the XY plane.
    # Kept last to preserve positional construction compatibility.
    projection_palette: str | None = None


def compute_opacity_unit_distance(reference_voxel_distance: float, factor: float) -> float:
    """Return VTK scalar-opacity unit distance in the payload's world units.

    The reference distance is the smallest positive physical grid spacing,
    matching the VTK lab notebook. It is deliberately independent from
    sample_distance_factor: the former controls opacity accumulation, while
    the latter controls ray sampling density.
    """
    spacing = float(reference_voxel_distance)
    multiplier = float(factor)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("reference voxel distance должен быть конечным и > 0.")
    if not np.isfinite(multiplier) or not (0.05 <= multiplier <= 20.0):
        raise ValueError("opacity_unit_distance_factor должен лежать в [0.05, 20].")
    return max(spacing * multiplier, 1e-6)


def _line_polydata(vtk: Any, segments: list[tuple[np.ndarray, np.ndarray]]):
    """Create a compact polydata object containing independent 3D line segments."""
    data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    lines = vtk.vtkCellArray()
    for start, end in segments:
        p0 = points.InsertNextPoint(float(start[0]), float(start[1]), float(start[2]))
        p1 = points.InsertNextPoint(float(end[0]), float(end[1]), float(end[2]))
        lines.InsertNextCell(2)
        lines.InsertCellPoint(p0)
        lines.InsertCellPoint(p1)
    data.SetPoints(points)
    data.SetLines(lines)
    data.Modified()
    return data


class VTKVolumeScene:
    """Owns VTK objects and mutates only transfer functions between UI updates."""

    def __init__(self, render_window: Any, interactor: Any, render_callback=None):
        vtk, _ = _vtk()
        self.vtk = vtk
        self.render_window = render_window
        self.interactor = interactor
        # QVTKRenderWindowInteractor.Render() is delegated to the VTK
        # interactor and renders immediately.  The owning Qt widget's
        # paintEvent also invokes that VTK method.  GUI slots therefore pass
        # QWidget.update as render_callback so Qt coalesces requests and the
        # actual render runs through QVTK's normal paint lifecycle.
        self._render_callback = render_callback
        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)
        # Match the familiar PyVista/Jupyter camera interaction. A generic
        # render-window interactor defaults to vtkInteractorStyleSwitch, whose
        # active mode can be vtkInteractorStyleJoystickCamera. Use the stable
        # trackball mapping explicitly on every platform.
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor_style.SetMotionFactor(10.0)
        self.interactor_style.SetMouseWheelMotionFactor(1.0)
        self.interactor_style.SetDefaultRenderer(self.renderer)
        self.interactor.SetInteractorStyle(self.interactor_style)
        self.grid = None
        self.mapper = None
        # Original screen-space sampling and automatic-quality mode are saved
        # while a linked theta-range slider is actively dragged.
        self._theta_range_interaction_render_state: tuple[float, bool] | None = None
        self.volume = None
        self.volume_property = None
        self.color_function = vtk.vtkColorTransferFunction()
        self.opacity_function = vtk.vtkPiecewiseFunction()
        self.outline_filter = None
        self.outline_actor = None
        self.axis_tick_polydata = None
        self.axis_tick_actor = None
        self.grid_polydata = None
        self.grid_actor = None
        self._axis_label_actors: list[Any] = []
        self._axis_label_title_flags: list[bool] = []
        self._scene_base_matrix = vtk.vtkMatrix4x4()
        self._scene_base_matrix.Identity()
        # Stable matrix object shared by all scene props.  Its contents are
        # updated in-place so already-attached outline/grid actors never retain
        # a stale identity matrix after payload replacement.
        self._scene_user_matrix = vtk.vtkMatrix4x4()
        self._scene_user_matrix.Identity()
        self._axis_scales = (1.0, 1.0, 1.0)
        self._axis_corner = None
        # Immutable physical axes are kept separate from the display-only axes
        # installed in vtkRectilinearGrid. Each Apply is calculated from these
        # source arrays, never from previously scaled coordinates.
        self._physical_x = None
        self._physical_y = None
        self._physical_theta = None
        self._display_x = None
        self._display_y = None
        self._display_theta = None
        # GPU-only theta preview crop. None means the full volume is visible;
        # otherwise the tuple stores inclusive indices into the immutable
        # physical theta axis. It never slices payload.volume or projection.
        self._theta_display_range_indices: tuple[int, int] | None = None
        # Optional visual caps for the active theta preview interval. These
        # are ordinary lightweight polygon props; they never participate in
        # volume cropping, scalar transfer functions, or the payload itself.
        self._theta_boundary_planes_visible = False
        self._theta_boundary_plane_opacity = 0.12
        self._theta_boundary_plane_sources: list[Any] = []
        self._theta_boundary_plane_append = None
        self._theta_boundary_plane_mapper = None
        self._theta_boundary_plane_actor = None
        # Live editable selector preview. These persistent actors are created
        # once and only their tiny polydata is replaced while the user edits.
        self._roi_preview_roi = None
        self._roi_preview_enabled = False
        self._roi_preview_options = {
            "selector_type": "rect",
            "geometry_mode": "surface",
            "show_fill": True,
            "show_outline": True,
            "show_points": False,
            "opacity": 0.10,
            "highlight_mode": "inside",
            "highlight_strength": 0.92,
            "highlight_color": "#FFB547",
            "thickness_voxels": 1.5,
        }
        self._roi_preview_surface_polydata = None
        self._roi_preview_outline_polydata = None
        self._roi_preview_points_polydata = None
        self._roi_preview_surface_actor = None
        self._roi_preview_outline_actor = None
        self._roi_preview_points_actor = None
        self._camera_observer_id = None
        self.projection_grid = None
        self.projection_mapper = None
        self.projection_actor = None
        self.projection_lut = None
        self.projection_info = None
        # Selected-theta emphasis is injected into the existing GPU volume
        # ray-casting shader. With the optional marker disabled (the default),
        # this path creates no extra geometry and preserves the original slice
        # appearance exactly. The marker is a separate abstract geometric plane,
        # never a scalar-coloured copy of the source data.
        self._theta_slice_index: int | None = None
        self._theta_slice_enabled = False
        self._theta_slice_palette = "turbo"
        self._theta_slice_strength = 0.90
        self._theta_slice_shader_property = None
        self._theta_slice_uniforms = None
        self._theta_slice_shader_installed = False
        self._theta_slice_bounds: tuple[float, float] | None = None
        self._theta_slice_transfer_range: tuple[float, float] | None = None
        # Optional abstract marker at the selected physical theta. It mirrors
        # the translucent geometric caps in the theta-range tool: no source
        # scalars, lookup table, or change to the volume's slice shader.
        self._theta_slice_marker_enabled = False
        self._theta_slice_marker_opacity = 0.12
        self._theta_slice_marker_source = vtk.vtkPlaneSource()
        self._theta_slice_marker_source.SetXResolution(1)
        self._theta_slice_marker_source.SetYResolution(1)
        self._theta_slice_marker_source.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
        self._theta_slice_marker_mapper = vtk.vtkPolyDataMapper()
        self._theta_slice_marker_mapper.SetInputConnection(
            self._theta_slice_marker_source.GetOutputPort()
        )
        self._theta_slice_marker_actor = vtk.vtkActor()
        self._theta_slice_marker_actor.SetMapper(self._theta_slice_marker_mapper)
        self._theta_slice_marker_actor.SetUserMatrix(self._scene_user_matrix)
        self._theta_slice_marker_actor.PickableOff()
        marker_property = self._theta_slice_marker_actor.GetProperty()
        marker_property.SetLighting(False)
        marker_property.SetAmbient(1.0)
        marker_property.SetDiffuse(0.0)
        marker_property.SetOpacity(float(self._theta_slice_marker_opacity))
        marker_property.SetEdgeVisibility(True)
        marker_property.SetEdgeOpacity(0.92)
        marker_property.SetLineWidth(1.5)
        self._theta_slice_marker_actor.SetVisibility(0)
        self.renderer.AddActor(self._theta_slice_marker_actor)

        # Peak centres are a separate, lightweight point layer. They never
        # alter the volume scalar array or its transfer function. The actor is
        # persistent across payload refreshes; only its point data is replaced.
        self._peak_marker_polydata = vtk.vtkPolyData()
        self._peak_marker_points = vtk.vtkPoints()
        self._peak_marker_points.SetDataTypeToDouble()
        self._peak_marker_polydata.SetPoints(self._peak_marker_points)
        # Use instanced sphere geometry rather than GL point-sprites.  Point
        # sprites depend on driver-specific point-size limits and can disappear
        # in embedded Qt/VTK contexts even while line overlays render correctly.
        # vtkGlyph3DMapper renders each event as a real, depth-tested 3D sphere.
        self._peak_marker_glyph_source = vtk.vtkSphereSource()
        self._peak_marker_glyph_source.SetRadius(1.0)
        self._peak_marker_glyph_source.SetThetaResolution(16)
        self._peak_marker_glyph_source.SetPhiResolution(12)
        self._peak_marker_glyph_source.Update()
        self._peak_marker_mapper = vtk.vtkGlyph3DMapper()
        self._peak_marker_mapper.SetInputData(self._peak_marker_polydata)
        self._peak_marker_mapper.SetSourceConnection(
            self._peak_marker_glyph_source.GetOutputPort()
        )
        # The per-point scale array is refreshed from the active camera so the
        # requested marker diameter is measured in actual screen pixels. A
        # single world-space radius would shrink/grow with zoom and would be
        # especially misleading across the angular (theta) axis.
        self._peak_marker_mapper.SetScaleModeToScaleByMagnitude()
        self._peak_marker_mapper.SetScaleArray("PeakMarkerGlyphScale")
        self._peak_marker_mapper.SetScaleFactor(0.1)
        self._peak_marker_mapper.SetScalarModeToUsePointData()
        self._peak_marker_mapper.SetColorModeToDirectScalars()
        self._peak_marker_mapper.ScalarVisibilityOn()
        self._peak_marker_actor = vtk.vtkActor()
        self._peak_marker_actor.SetMapper(self._peak_marker_mapper)
        self._peak_marker_actor.SetUserMatrix(self._scene_user_matrix)
        # Hover inspection uses a dedicated vtkPointPicker restricted to this
        # actor; the marker layer remains read-only and does not intercept the
        # trackball camera's mouse interaction.
        self._peak_marker_actor.PickableOn()
        peak_property = self._peak_marker_actor.GetProperty()
        peak_property.SetOpacity(0.95)
        peak_property.SetLighting(True)
        peak_property.SetAmbient(0.78)
        peak_property.SetDiffuse(0.22)
        peak_property.SetSpecular(0.12)
        peak_property.SetSpecularPower(18.0)
        self._peak_marker_actor.SetVisibility(0)
        # Register the persistent marker actor exactly once. The picker can
        # reference an unregistered actor, so hover diagnostics alone do not
        # reveal a missing renderer prop; without AddActor(), markers never draw.
        self.renderer.AddActor(self._peak_marker_actor)
        self._peak_marker_positions_physical = np.empty((0, 3), dtype=np.float64)
        self._peak_marker_display_positions = np.empty((0, 3), dtype=np.float64)
        self._peak_marker_colors = np.empty((0, 4), dtype=np.uint8)
        self._peak_marker_peak_ids = np.empty(0, dtype=np.int64)
        self._peak_marker_branch_ids = np.empty(0, dtype=np.int64)
        self._peak_marker_analysis_included = np.empty(0, dtype=bool)
        self._peak_marker_rendered_indices = np.empty(0, dtype=np.int64)
        self._show_excluded_branches = False
        self._peak_branch_event_counts: dict[int, int] = {}
        self._peak_excluded_branch_event_counts: dict[int, int] = {}
        # Shared branch-manager display state belongs to the scene from its
        # first construction, not only after set_payload() happens to run.
        self._branch_display_colors: dict[int, str] = {}
        self._branch_display_selected_ids = np.empty(0, dtype=np.int64)
        self._branch_display_focus_selected = False
        self._branch_display_previous_visibility: tuple[bool, bool] | None = None
        self._peak_point_picker = vtk.vtkPointPicker()
        self._peak_point_picker.SetTolerance(0.012)
        self._peak_point_picker.PickFromListOn()
        self._peak_point_picker.InitializePickList()
        self._peak_point_picker.AddPickList(self._peak_marker_actor)
        self._peak_marker_scale_values = np.empty(0, dtype=np.float64)
        self._peak_marker_size = 7.0
        self._peak_marker_opacity = 0.95
        self._peak_marker_world_radius = 0.1
        self._peak_marker_world_radius_range = (0.1, 0.1)
        self._peak_marker_reference_pitch = 0.0
        self._peak_marker_visible = False
        self._peak_marker_camera_observer_id = None
        self._peak_marker_resize_observer_id = None

        # Branch links are an independent, lightweight line layer.  They are
        # built from the immutable peak catalogue + branch_edges table and do
        # not modify the scientific volume or the catalogue itself.
        self._branch_edge_polydata = vtk.vtkPolyData()
        self._branch_edge_mapper = vtk.vtkPolyDataMapper()
        self._branch_edge_mapper.SetInputData(self._branch_edge_polydata)
        self._branch_edge_mapper.SetScalarModeToUseCellData()
        self._branch_edge_mapper.SetColorModeToDirectScalars()
        self._branch_edge_mapper.ScalarVisibilityOn()
        self._branch_edge_actor = vtk.vtkActor()
        self._branch_edge_actor.SetMapper(self._branch_edge_mapper)
        self._branch_edge_actor.SetUserMatrix(self._scene_user_matrix)
        self._branch_edge_actor.PickableOff()
        branch_property = self._branch_edge_actor.GetProperty()
        branch_property.SetLighting(False)
        branch_property.SetLineWidth(2.0)
        self._branch_edge_actor.SetVisibility(0)
        self.renderer.AddActor(self._branch_edge_actor)

        # Correlation correspondences are a separate display-only layer. Solid
        # links mean accepted source↔partner assignments; optional dashed amber
        # links mean supported alternatives which were not selected. Neither
        # layer changes branch_edges or any scientific artifact.
        self._branch_correlation_matched_polydata = vtk.vtkPolyData()
        self._branch_correlation_matched_mapper = vtk.vtkPolyDataMapper()
        self._branch_correlation_matched_mapper.SetInputData(
            self._branch_correlation_matched_polydata
        )
        self._branch_correlation_matched_mapper.SetScalarModeToUseCellData()
        self._branch_correlation_matched_mapper.SetColorModeToDirectScalars()
        self._branch_correlation_matched_mapper.ScalarVisibilityOn()
        self._branch_correlation_matched_actor = vtk.vtkActor()
        self._branch_correlation_matched_actor.SetMapper(
            self._branch_correlation_matched_mapper
        )
        self._branch_correlation_matched_actor.SetUserMatrix(self._scene_user_matrix)
        self._branch_correlation_matched_actor.PickableOff()
        matched_property = self._branch_correlation_matched_actor.GetProperty()
        matched_property.SetLighting(False)
        matched_property.SetLineWidth(2.0)
        matched_property.SetOpacity(0.82)
        self._branch_correlation_matched_actor.SetVisibility(0)
        self.renderer.AddActor(self._branch_correlation_matched_actor)

        self._branch_correlation_candidate_polydata = vtk.vtkPolyData()
        self._branch_correlation_candidate_mapper = vtk.vtkPolyDataMapper()
        self._branch_correlation_candidate_mapper.SetInputData(
            self._branch_correlation_candidate_polydata
        )
        self._branch_correlation_candidate_mapper.ScalarVisibilityOff()
        self._branch_correlation_candidate_actor = vtk.vtkActor()
        self._branch_correlation_candidate_actor.SetMapper(
            self._branch_correlation_candidate_mapper
        )
        self._branch_correlation_candidate_actor.SetUserMatrix(self._scene_user_matrix)
        self._branch_correlation_candidate_actor.PickableOff()
        candidate_property = self._branch_correlation_candidate_actor.GetProperty()
        candidate_property.SetLighting(False)
        candidate_property.SetColor(0.96, 0.61, 0.22)
        candidate_property.SetLineWidth(1.4)
        candidate_property.SetOpacity(0.82)
        self._branch_correlation_candidate_actor.SetVisibility(0)
        self.renderer.AddActor(self._branch_correlation_candidate_actor)

        self._branch_edge_positions_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_edge_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_edge_base_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_edge_branch_ids = np.empty(0, dtype=np.int64)
        self._branch_edge_analysis_included = np.empty(0, dtype=bool)
        self._branch_edge_visible = False
        self._branch_edge_width = 2.0
        self._branch_edge_opacity = 0.92
        self._branch_correlation_matched_segments_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_correlation_matched_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_correlation_candidate_segments_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_correlation_matched_peak_ids = np.empty(0, dtype=np.int64)
        self._branch_correlation_focus_branch_ids = np.empty(0, dtype=np.int64)
        self._branch_correlation_visible = False
        self._branch_correlation_show_candidates = False
        self._branch_correlation_emphasize_core = True
        self._branch_correlation_residual_opacity = 0.28
        self._branch_correlation_line_width = 2.0
        self._branch_correlation_opacity = 0.82
        self._branch_correlation_candidate_opacity = 0.82
        self._peak_marker_base_colors = np.empty((0, 4), dtype=np.uint8)
        self._volume_visible = True

        # The map cursor is an independent, narrow optical marker inside the
        # volume ray. Unlike the previous RGB tint, it contributes its own
        # alpha even when the scalar transfer function is fully transparent.
        # Front-to-back volume compositing then depth-orders cloud and marker.
        self._cursor_line_enabled = False
        self._cursor_line_position: tuple[float, float] | None = None
        self._cursor_line_radii: tuple[float, float] | None = None
        # Optical density of the independent cursor marker, measured per
        # volume-texture voxel. It contributes opacity inside the ray marcher
        # even where the source scalar has zero transfer-function opacity.
        self._cursor_line_density = 0.95
        self._cursor_line_color = "#FFBD55"
        self.axes_widget = None
        self.payload = None
        self.grid_info = None
        self.settings = TransferSettings()
        self.theme = "dark"
        self._finite_sample = None
        self._positive_reference_999: float | None = None
        self._positive_max: float | None = None
        self._percentile_sample_stride = 1
        self._min_spacing = 1.0
        self._physical_theta = None
        self._theta_scale = 1.0
        self._projection_base_z = None
        # Independent visibility switches: the corner orientation triad, the
        # volume bounds frame, and coordinate tick marks/labels are separate
        # scientific aids and must never be coupled to one overloaded toggle.
        self._axes_visible = True
        self._bounding_box_visible = True
        self._axis_ticks_visible = True
        self._grid_visible = self._default_grid_visibility()
        self._last_transfer_info: dict[str, float] = {}
        self._closed = False
        self._build_axes_marker()

    @staticmethod
    def _default_grid_visibility() -> bool:
        """Use the same initial grid preference as the Plotly volume view."""
        try:
            from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings
            return bool(load_plot_display_settings()["volume"].get("show_grid", True))
        except Exception:
            return True

    def _request_render(self) -> None:
        """Schedule a QVTK repaint through Qt's paint event."""
        callback = self._render_callback
        if not self._closed and callable(callback):
            callback()

    def _build_axes_marker(self) -> None:
        """Create the orientation widget and one geometry-locked frame/axis system.

        The rectangular frame is the authoritative geometry. Tick marks, labels
        and optional face gridlines are generated in the *same local coordinate
        system* and receive the exact same ROI transform as the volume. This avoids
        vtkCubeAxesActor's world-aligned AABB axes drifting away from a rotated ROI.
        """
        vtk = self.vtk
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        try:
            axes.GetZAxisCaptionActor2D().SetCaption("θ")
        except Exception:
            pass
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0.0, 0.0, 0.18, 0.22)
        widget.SetEnabled(1)
        widget.InteractiveOff()
        self.axes_widget = widget

        # Authoritative rectangular wireframe: vtkOutlineFilter uses the exact
        # rectilinear dataset bounds. The actor is transformed exactly like volume.
        self.outline_filter = vtk.vtkOutlineFilter()
        self.outline_mapper = vtk.vtkPolyDataMapper()
        self.outline_mapper.SetInputConnection(self.outline_filter.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(self.outline_mapper)
        self.outline_actor.PickableOff()
        self.renderer.AddActor(self.outline_actor)

        # Major tick marks are short segments starting ON the frame edges.
        self.axis_tick_polydata = vtk.vtkPolyData()
        self.axis_tick_mapper = vtk.vtkPolyDataMapper()
        self.axis_tick_mapper.SetInputData(self.axis_tick_polydata)
        self.axis_tick_actor = vtk.vtkActor()
        self.axis_tick_actor.SetMapper(self.axis_tick_mapper)
        self.axis_tick_actor.PickableOff()
        self.renderer.AddActor(self.axis_tick_actor)

        # Gridlines are drawn only on the three camera-facing box faces.
        self.grid_polydata = vtk.vtkPolyData()
        self.grid_mapper = vtk.vtkPolyDataMapper()
        self.grid_mapper.SetInputData(self.grid_polydata)
        self.grid_actor = vtk.vtkActor()
        self.grid_actor.SetMapper(self.grid_mapper)
        self.grid_actor.PickableOff()
        self.renderer.AddActor(self.grid_actor)

        # Two optional translucent caps show the selected theta interval.
        # A single actor/mapper is used for both planes to keep the overlay
        # cheap and make its lifetime independent of slider drag events.
        self._theta_boundary_plane_sources = [vtk.vtkPlaneSource(), vtk.vtkPlaneSource()]
        for source in self._theta_boundary_plane_sources:
            source.SetXResolution(1)
            source.SetYResolution(1)
            source.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
        self._theta_boundary_plane_append = vtk.vtkAppendPolyData()
        for source in self._theta_boundary_plane_sources:
            self._theta_boundary_plane_append.AddInputConnection(source.GetOutputPort())
        self._theta_boundary_plane_mapper = vtk.vtkPolyDataMapper()
        self._theta_boundary_plane_mapper.SetInputConnection(
            self._theta_boundary_plane_append.GetOutputPort()
        )
        self._theta_boundary_plane_actor = vtk.vtkActor()
        self._theta_boundary_plane_actor.SetMapper(self._theta_boundary_plane_mapper)
        self._theta_boundary_plane_actor.SetUserMatrix(self._scene_user_matrix)
        self._theta_boundary_plane_actor.PickableOff()
        plane_property = self._theta_boundary_plane_actor.GetProperty()
        plane_property.SetLighting(False)
        plane_property.SetAmbient(1.0)
        plane_property.SetDiffuse(0.0)
        plane_property.SetOpacity(float(self._theta_boundary_plane_opacity))
        plane_property.SetEdgeVisibility(True)
        plane_property.SetEdgeOpacity(0.92)
        plane_property.SetLineWidth(1.5)
        self._theta_boundary_plane_actor.SetVisibility(0)
        self.renderer.AddActor(self._theta_boundary_plane_actor)

        # ROI selector preview is a small, display-only overlay. Separate fill,
        # outline and sample-marker actors let users reduce visual density
        # without touching the rendered volume or its transfer function.
        self._roi_preview_surface_polydata = vtk.vtkPolyData()
        self._roi_preview_outline_polydata = vtk.vtkPolyData()
        self._roi_preview_points_polydata = vtk.vtkPolyData()
        self._roi_preview_surface_mapper = vtk.vtkPolyDataMapper()
        self._roi_preview_surface_mapper.SetInputData(self._roi_preview_surface_polydata)
        self._roi_preview_outline_mapper = vtk.vtkPolyDataMapper()
        self._roi_preview_outline_mapper.SetInputData(self._roi_preview_outline_polydata)
        self._roi_preview_points_mapper = vtk.vtkPolyDataMapper()
        self._roi_preview_points_mapper.SetInputData(self._roi_preview_points_polydata)

        self._roi_preview_surface_actor = vtk.vtkActor()
        self._roi_preview_surface_actor.SetMapper(self._roi_preview_surface_mapper)
        self._roi_preview_outline_actor = vtk.vtkActor()
        self._roi_preview_outline_actor.SetMapper(self._roi_preview_outline_mapper)
        self._roi_preview_points_actor = vtk.vtkActor()
        self._roi_preview_points_actor.SetMapper(self._roi_preview_points_mapper)
        for actor in (
            self._roi_preview_surface_actor,
            self._roi_preview_outline_actor,
            self._roi_preview_points_actor,
        ):
            actor.PickableOff()
            if hasattr(actor, "UseBoundsOff"):
                actor.UseBoundsOff()
            # Keep the same shared ROI transform as the volume/frame. For the
            # normal HDF5 path it is identity (detector-grid regridding); this
            # also keeps direct local-(u,v) scene payloads geometrically valid.
            actor.SetUserMatrix(self._scene_user_matrix)
            actor.SetVisibility(0)
            self.renderer.AddActor(actor)
        surface_property = self._roi_preview_surface_actor.GetProperty()
        surface_property.SetLighting(False)
        surface_property.SetAmbient(1.0)
        surface_property.SetDiffuse(0.0)
        outline_property = self._roi_preview_outline_actor.GetProperty()
        outline_property.SetLighting(False)
        outline_property.SetAmbient(1.0)
        outline_property.SetDiffuse(0.0)
        outline_property.SetLineWidth(1.8)
        points_property = self._roi_preview_points_actor.GetProperty()
        points_property.SetLighting(False)
        points_property.SetAmbient(1.0)
        points_property.SetDiffuse(0.0)
        points_property.SetPointSize(6.0)
        if hasattr(points_property, "SetRenderPointsAsSpheres"):
            points_property.SetRenderPointsAsSpheres(True)
        self._style_roi_preview()

        # Billboard text always faces the camera. Its anchors are transformed
        # explicitly in _make_axis_text because vtkAssembly does not reliably
        # propagate transforms to vtkBillboardTextActor3D in this pipeline.
        for prop in (self.outline_actor, self.axis_tick_actor, self.grid_actor):
            prop.SetUserMatrix(self._scene_user_matrix)
        self.outline_actor.SetVisibility(0)
        self.axis_tick_actor.SetVisibility(0)
        self.grid_actor.SetVisibility(0)
        self._style_geometry_axes()

        # A rotated ROI has a local u/v frame. Re-select the visible corner only
        # when the camera crosses into another octant; this is cheap during drag.
        try:
            self._camera_observer_id = self.interactor.AddObserver(
                "InteractionEvent", self._on_camera_interaction
            )
        except Exception:
            self._camera_observer_id = None
        # QVTK's Qt adapter rebases every drag sample before dispatching it to
        # TrackballCamera. Interactor observers cannot do this during a drag:
        # GrabFocus intentionally routes events exclusively to the style.

        # Keep the screen-pixel diameter invariant while the camera zooms,
        # rotates, switches projection, or the embedded viewport is resized.
        # Observers are owned and explicitly removed by close().
        try:
            active_camera = self.renderer.GetActiveCamera()
            self._peak_marker_camera_observer_id = active_camera.AddObserver(
                "ModifiedEvent", self._on_peak_marker_view_changed
            )
        except Exception:
            self._peak_marker_camera_observer_id = None
        try:
            self._peak_marker_resize_observer_id = self.render_window.AddObserver(
                "WindowResizeEvent", self._on_peak_marker_view_changed
            )
        except Exception:
            self._peak_marker_resize_observer_id = None

    @staticmethod
    def _rgb(hex_color: str) -> tuple[float, float, float]:
        value = str(hex_color).lstrip("#")
        if len(value) != 6:
            raise ValueError(f"Некорректный цвет: {hex_color!r}")
        return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _axis_style(self) -> dict[str, Any]:
        if self.theme == "dark":
            foreground, line, grid = "#e1e4e8", "#aeb5be", "#343941"
        else:
            foreground, line, grid = "#111111", "#222222", "#e0e0e0"
        try:
            from roc_mcs.gui.roi_explorer.plot_display_settings import load_plot_display_settings
            style = load_plot_display_settings()["volume"]
        except Exception:
            style = {"font_family": "Times New Roman", "tick_size": 14, "label_size": 16}
        return {
            "foreground": self._rgb(foreground),
            "line": self._rgb(line),
            "grid": self._rgb(grid),
            "font_family": str(style.get("font_family", "Times New Roman")),
            "tick_size": int(style.get("tick_size", 14)),
            "label_size": int(style.get("label_size", 16)),
        }

    def _style_geometry_axes(self) -> None:
        style = self._axis_style()
        if self.outline_actor is not None:
            prop = self.outline_actor.GetProperty()
            prop.SetColor(*style["line"])
            prop.SetLineWidth(1.35)
            prop.SetLighting(False)
        if self.axis_tick_actor is not None:
            prop = self.axis_tick_actor.GetProperty()
            prop.SetColor(*style["line"])
            prop.SetLineWidth(1.25)
            prop.SetLighting(False)
        if self.grid_actor is not None:
            prop = self.grid_actor.GetProperty()
            prop.SetColor(*style["grid"])
            prop.SetLineWidth(0.65)
            prop.SetOpacity(0.48)
            prop.SetLighting(False)
        for actor, is_title in zip(self._axis_label_actors, self._axis_label_title_flags):
            text = actor.GetTextProperty()
            text.SetFontFamilyAsString(style["font_family"])
            text.SetFontSize(style["label_size"] if is_title else style["tick_size"])
            text.SetColor(*style["foreground"])
            text.SetBold(False)
            text.SetItalic(False)
            text.SetShadow(False)
            text.SetJustificationToCentered()
            text.SetVerticalJustificationToCentered()
            text.SetBackgroundOpacity(0.0)
        self._style_theta_boundary_planes()
        self._style_theta_slice_marker()

    def _style_roi_preview(self) -> None:
        """Apply one restrained, theme-aware accent to all ROI preview layers."""
        if self.theme == "dark":
            accent = (0.20, 0.84, 0.77)
        else:
            accent = (0.00, 0.48, 0.44)
        if self._roi_preview_surface_actor is not None:
            prop = self._roi_preview_surface_actor.GetProperty()
            prop.SetColor(*accent)
            prop.SetOpacity(float(self._roi_preview_options.get("opacity", 0.10)))
        if self._roi_preview_outline_actor is not None:
            prop = self._roi_preview_outline_actor.GetProperty()
            prop.SetColor(*accent)
            prop.SetOpacity(0.96)
            prop.SetLineWidth(1.8)
        if self._roi_preview_points_actor is not None:
            prop = self._roi_preview_points_actor.GetProperty()
            prop.SetColor(1.0, 0.88, 0.48) if self.theme == "dark" else prop.SetColor(0.72, 0.31, 0.04)
            prop.SetOpacity(1.0)
            prop.SetPointSize(5.5)

    @staticmethod
    def _roi_preview_polydata(vtk: Any, coordinates, cells, cell_kind: str):
        """Create tiny double-precision polygon/line/vertex polydata."""
        data = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for xyz in coordinates:
            points.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        if cell_kind == "polys":
            cell_array = vtk.vtkCellArray()
            for cell in cells:
                ids = vtk.vtkIdList()
                for index in cell:
                    ids.InsertNextId(int(index))
                cell_array.InsertNextCell(ids)
            data.SetPolys(cell_array)
        elif cell_kind == "lines":
            cell_array = vtk.vtkCellArray()
            for cell in cells:
                ids = vtk.vtkIdList()
                for index in cell:
                    ids.InsertNextId(int(index))
                cell_array.InsertNextCell(ids)
            data.SetLines(cell_array)
        elif cell_kind == "verts":
            cell_array = vtk.vtkCellArray()
            for index in cells:
                cell_array.InsertNextCell(1)
                cell_array.InsertCellPoint(int(index))
            data.SetVerts(cell_array)
        else:
            raise ValueError(f"Неизвестный тип ROI preview cells: {cell_kind!r}.")
        data.SetPoints(points)
        data.Modified()
        return data

    @staticmethod
    def _roi_preview_map_axis(value: float, physical_axis, display_axis) -> float:
        if physical_axis is None or display_axis is None:
            return float(value)
        source = np.asarray(physical_axis, dtype=np.float64)
        display = np.asarray(display_axis, dtype=np.float64)
        if source.size < 2 or display.size != source.size:
            return float(value)
        source_center = 0.5 * (float(source[0]) + float(source[-1]))
        display_center = 0.5 * (float(display[0]) + float(display[-1]))
        source_span = float(source[-1] - source[0])
        display_span = float(display[-1] - display[0])
        if not np.isfinite(source_span) or source_span <= 0.0:
            return float(value)
        return display_center + (float(value) - source_center) * (display_span / source_span)

    def set_roi_preview(
        self,
        roi: Any,
        *,
        selector_type: str | None = None,
        show_fill: bool = True,
        show_outline: bool = True,
        show_points: bool = False,
        opacity: float = 0.10,
        geometry_mode: str | None = None,
        highlight_mode: str = "inside",
        highlight_strength: float = 0.92,
        highlight_color: str = "#FFB547",
        thickness_voxels: float = 1.5,
        render: bool = True,
    ) -> None:
        """Set live selector geometry and GPU cloud emphasis without reloading data.

        ``highlight_mode`` is one of ``off``, ``inside`` (color only the
        intersecting volume samples), or ``context`` (also dim non-selected
        samples). Line/point selections have zero mathematical area, so their
        display footprint is specified separately in voxel-width units.
        """
        canonical_roi, kind, _points = canonicalize_roi_preview(roi, selector_type)
        try:
            alpha = float(opacity)
            strength = float(highlight_strength)
            thickness = float(thickness_voxels)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Параметры отображения ROI preview должны быть числами.") from exc
        if not all(np.isfinite(value) for value in (alpha, strength, thickness)):
            raise ValueError("Параметры отображения ROI preview должны быть конечными числами.")
        if not 0.02 <= alpha <= 0.35:
            raise ValueError("Opacity ROI preview должна лежать в диапазоне 0.02–0.35.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Сила подсветки ROI должна лежать в диапазоне 0–1.")
        if not 0.5 <= thickness <= 6.0:
            raise ValueError("Ширина выделения линии/точки должна лежать в диапазоне 0.5–6 вокселей.")
        mode = str(highlight_mode).strip().lower()
        if mode not in ("off", "inside", "context"):
            raise ValueError("Режим подсветки ROI должен быть off, inside или context.")
        color_rgb = self._hex_rgb(highlight_color)
        normalized_color = "#" + "".join(f"{round(component * 255):02X}" for component in color_rgb)
        if geometry_mode is None:
            geometry_mode = "surface" if show_fill else ("outline" if show_outline else "hidden")
        geometry_mode = str(geometry_mode).strip().lower()
        if geometry_mode not in ("surface", "outline", "hidden"):
            raise ValueError("Режим геометрии ROI должен быть surface, outline или hidden.")

        # Copy only the scalar geometry. The object supplied by the UI can be
        # edited afterwards without mutating the committed preview state.
        self._roi_preview_roi = canonical_roi
        self._roi_preview_options = {
            "selector_type": kind,
            "geometry_mode": geometry_mode,
            "show_fill": geometry_mode == "surface",
            "show_outline": geometry_mode in ("surface", "outline"),
            "show_points": bool(show_points),
            "opacity": alpha,
            "highlight_mode": mode,
            "highlight_strength": strength,
            "highlight_color": normalized_color,
            "thickness_voxels": thickness,
        }
        self._style_roi_preview()
        self._update_roi_preview_geometry()
        if render:
            self._request_render()

    def set_roi_preview_visible(self, visible: bool, *, render: bool = True) -> None:
        """Toggle the geometric preview and the matching in-volume highlight."""
        self._roi_preview_enabled = bool(visible)
        if self._roi_preview_enabled:
            # Geometry may have been intentionally left untouched while the
            # preview was hidden and θ-range changes were being applied.
            self._update_roi_preview_geometry()
        else:
            self._apply_roi_preview_visibility()
            self._update_roi_preview_shader_uniforms()
        if render:
            self._request_render()

    def _apply_roi_preview_visibility(self) -> None:
        ready = bool(self._roi_preview_enabled and self.payload is not None and self._roi_preview_roi is not None)
        geometry_mode = str(self._roi_preview_options.get("geometry_mode", "surface"))
        if self._roi_preview_surface_actor is not None:
            self._roi_preview_surface_actor.SetVisibility(
                1 if ready and geometry_mode == "surface" else 0
            )
        if self._roi_preview_outline_actor is not None:
            self._roi_preview_outline_actor.SetVisibility(
                1 if ready and geometry_mode in ("surface", "outline") else 0
            )
        if self._roi_preview_points_actor is not None:
            self._roi_preview_points_actor.SetVisibility(
                1 if ready and self._roi_preview_options.get("show_points", False) else 0
            )

    def _update_roi_preview_shader_uniforms(
        self, kind: str | None = None, xy=None, *, coordinates_are_display: bool = False
    ) -> None:
        """Update ROI shader uniforms in the exact coordinate frame used by its actors.

        ``xy`` is detector-local unless ``coordinates_are_display`` is true.
        Geometry rebuilding passes its already-transformed vertices so that
        UV-to-local conversion and axis scaling are never applied twice.
        """
        uniforms = self._theta_slice_uniforms
        if uniforms is None:
            return
        options = self._roi_preview_options
        mode_name = str(options.get("highlight_mode", "off"))
        kind = str(kind or options.get("selector_type", "rect"))
        active = bool(
            self._roi_preview_enabled
            and self.payload is not None
            and self._roi_preview_roi is not None
            and mode_name != "off"
            and self._physical_x is not None
            and self._physical_y is not None
            and self._display_x is not None
            and self._display_y is not None
        )
        uniforms.SetUniformi("rocRoiSelectionEnabled", int(active))
        if not active:
            return

        if xy is None:
            roi = self._roi_preview_roi
            if kind == "point":
                raw_xy = np.asarray([[float(roi.cx), float(roi.cy)]], dtype=np.float64)
            else:
                raw_xy = np.asarray(roi.corners(), dtype=np.float64).reshape(-1, 2)
        else:
            raw_xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)

        if coordinates_are_display:
            # _update_roi_preview_geometry has already applied the optional
            # payload UV transform and the current display-axis mapping.
            display_xy = np.array(raw_xy, dtype=np.float64, copy=True)
        else:
            if str(getattr(self.payload, "coord_system", "xy")).lower() == "uv":
                base_roi = getattr(self.payload, "roi", None)
                if base_roi is None:
                    uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                    return
                local_x, local_y = base_roi.to_local(raw_xy[:, 0], raw_xy[:, 1])
                raw_xy = np.column_stack((local_x, local_y))
            display_xy = np.array(raw_xy, dtype=np.float64, copy=True)
            display_xy[:, 0] = [
                self._roi_preview_map_axis(v, self._physical_x, self._display_x)
                for v in display_xy[:, 0]
            ]
            display_xy[:, 1] = [
                self._roi_preview_map_axis(v, self._physical_y, self._display_y)
                for v in display_xy[:, 1]
            ]
        if not np.all(np.isfinite(display_xy)):
            uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
            return

        type_code = {"rect": 0, "line": 1, "point": 2}.get(kind)
        if type_code is None:
            uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
            return
        uniforms.SetUniformi("rocRoiSelectionType", type_code)
        if kind == "rect":
            if display_xy.shape != (4, 2):
                uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                return
            area2 = float(np.sum(
                display_xy[:, 0] * np.roll(display_xy[:, 1], -1)
                - display_xy[:, 1] * np.roll(display_xy[:, 0], -1)
            ))
            if not np.isfinite(area2) or abs(area2) <= 1e-18:
                uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                return
            for index, point in enumerate(display_xy):
                uniforms.SetUniform2f(f"rocRoiQuad{index}", (float(point[0]), float(point[1])))
            uniforms.SetUniformf("rocRoiQuadOrientation", 1.0 if area2 > 0 else -1.0)
        elif kind == "line":
            if display_xy.shape != (2, 2):
                uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                return
            uniforms.SetUniform2f("rocRoiLineStart", (float(display_xy[0, 0]), float(display_xy[0, 1])))
            uniforms.SetUniform2f("rocRoiLineEnd", (float(display_xy[1, 0]), float(display_xy[1, 1])))
        else:
            if display_xy.shape != (1, 2):
                uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                return
            uniforms.SetUniform2f("rocRoiPoint", (float(display_xy[0, 0]), float(display_xy[0, 1])))

        dx = float(np.median(np.diff(self._display_x)))
        dy = float(np.median(np.diff(self._display_y)))
        if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0.0 and dy > 0.0):
            uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
            return
        half_width = 0.5 * float(options.get("thickness_voxels", 1.5))
        uniforms.SetUniform2f(
            "rocRoiFootprintRadius",
            (max(dx * half_width, 1e-12), max(dy * half_width, 1e-12)),
        )
        uniforms.SetUniformi(
            "rocRoiHighlightMode",
            2 if mode_name == "context" else 1,
        )
        uniforms.SetUniformf(
            "rocRoiHighlightStrength", float(options.get("highlight_strength", 0.92))
        )
        uniforms.SetUniform3f(
            "rocRoiHighlightColor",
            self._hex_rgb(str(options.get("highlight_color", "#FFB547"))),
        )

    def _update_roi_preview_geometry(self) -> None:
        """Rebuild a handful of ROI preview primitives in the current display frame."""
        if (
            self.payload is None or not self._roi_preview_enabled
            or self._roi_preview_roi is None
            or self._roi_preview_surface_polydata is None
        ):
            self._apply_roi_preview_visibility()
            if self._theta_slice_uniforms is not None:
                self._theta_slice_uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
            return
        theta = self._display_theta
        if theta is None or len(theta) < 2:
            self._apply_roi_preview_visibility()
            return
        low_index, high_index = self._theta_display_range_indices or (0, len(theta) - 1)
        low_index = int(np.clip(low_index, 0, len(theta) - 1))
        high_index = int(np.clip(high_index, 0, len(theta) - 1))
        if high_index <= low_index:
            low_index, high_index = 0, len(theta) - 1
        z_low = float(theta[low_index])
        z_high = float(theta[high_index])
        # Separate the lower footprint very slightly from the XY projection;
        # this avoids depth fighting without moving the marker outside the ROI.
        z_step = float(np.median(np.diff(theta)))
        if np.isfinite(z_step) and z_step > 0:
            nudge = min(0.025 * z_step, 0.08 * (z_high - z_low))
            z_low += nudge
            z_high -= nudge
        if not (np.isfinite(z_low) and np.isfinite(z_high) and z_low < z_high):
            self._apply_roi_preview_visibility()
            return

        roi = self._roi_preview_roi
        kind = str(self._roi_preview_options.get("selector_type", "rect"))
        if kind == "rect":
            xy = np.asarray(roi.corners(), dtype=np.float64).reshape(-1, 2)
            if xy.shape != (4, 2):
                raise ValueError("Для прямоугольной ROI preview ожидаются четыре угла.")
        elif kind == "line":
            xy = np.asarray(roi.corners(), dtype=np.float64).reshape(-1, 2)
            if xy.shape != (2, 2):
                raise ValueError("Для линейной ROI preview ожидаются два конца отрезка.")
        else:
            xy = np.asarray([[float(roi.cx), float(roi.cy)]], dtype=np.float64)
        if not np.all(np.isfinite(xy)):
            raise ValueError("Координаты ROI preview должны быть конечными.")

        # A legacy/direct UV payload keeps the volume in its ROI-local frame
        # and relies on _scene_user_matrix for detector placement. Express the
        # independent detector-space preview in that same local frame before
        # scaling the axes; the persistent preview actors share the matrix.
        if str(getattr(self.payload, "coord_system", "xy")).lower() == "uv":
            base_roi = getattr(self.payload, "roi", None)
            if base_roi is None:
                raise ValueError("Для UV-объёма отсутствует ROI-трансформация предпросмотра.")
            local_x, local_y = base_roi.to_local(xy[:, 0], xy[:, 1])
            xy = np.column_stack((local_x, local_y))

        xy[:, 0] = [self._roi_preview_map_axis(v, self._physical_x, self._display_x) for v in xy[:, 0]]
        xy[:, 1] = [self._roi_preview_map_axis(v, self._physical_y, self._display_y) for v in xy[:, 1]]
        # The shader receives exactly these same display-space coordinates.
        # Thus cloud highlighting and the wireframe footprint cannot drift.
        self._update_roi_preview_shader_uniforms(kind, xy, coordinates_are_display=True)
        z0, z1 = z_low, z_high
        coordinates: list[tuple[float, float, float]] = []
        polygons: list[list[int]] = []
        segments: list[list[int]] = []
        marker_indices: list[int] = []

        def add_point(x: float, y: float, z: float) -> int:
            index = len(coordinates)
            coordinates.append((float(x), float(y), float(z)))
            return index

        def add_segment(a: int, b: int) -> None:
            segments.append([int(a), int(b)])

        if kind == "rect":
            # Each lateral face spans the complete visible theta interval.
            lower = [add_point(x, y, z0) for x, y in xy]
            upper = [add_point(x, y, z1) for x, y in xy]
            for i in range(4):
                j = (i + 1) % 4
                polygons.append([lower[i], lower[j], upper[j], upper[i]])
                add_segment(lower[i], lower[j])
                add_segment(upper[i], upper[j])
                add_segment(lower[i], upper[i])
            marker_indices.extend(lower)
            marker_indices.extend(upper)
        elif kind == "line":
            # A mathematical line has zero cross-section. Its display-only
            # footprint is a capsule: a segment dilated by an ellipse whose
            # radii are exactly the same X/Y voxel-step radii used by the GPU
            # distance test. Construct it in normalized ellipse coordinates,
            # where the shader's footprint is a unit-radius capsule.
            direction = xy[1] - xy[0]
            display_length = float(np.linalg.norm(direction))
            dx = float(np.median(np.diff(self._display_x)))
            dy = float(np.median(np.diff(self._display_y)))
            if not np.isfinite(display_length) or display_length <= 1e-12:
                for data in (
                    self._roi_preview_surface_polydata,
                    self._roi_preview_outline_polydata,
                    self._roi_preview_points_polydata,
                ):
                    if data is not None:
                        data.Initialize()
                        data.Modified()
                self._apply_roi_preview_visibility()
                if self._theta_slice_uniforms is not None:
                    self._theta_slice_uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                return

            half_voxels = 0.5 * float(self._roi_preview_options.get("thickness_voxels", 1.5))
            radius_x = max(abs(dx) * half_voxels, 1e-12)
            radius_y = max(abs(dy) * half_voxels, 1e-12)
            radii = np.asarray((radius_x, radius_y), dtype=np.float64)
            p0n = xy[0] / radii
            p1n = xy[1] / radii
            normalized_direction = p1n - p0n
            normalized_length = float(np.linalg.norm(normalized_direction))
            if not np.isfinite(normalized_length) or normalized_length <= 1e-12:
                for data in (
                    self._roi_preview_surface_polydata,
                    self._roi_preview_outline_polydata,
                    self._roi_preview_points_polydata,
                ):
                    if data is not None:
                        data.Initialize()
                        data.Modified()
                self._apply_roi_preview_visibility()
                if self._theta_slice_uniforms is not None:
                    self._theta_slice_uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
                return

            direction_angle = float(np.arctan2(normalized_direction[1], normalized_direction[0]))
            end_angles = np.linspace(
                direction_angle + 0.5 * np.pi,
                direction_angle - 0.5 * np.pi,
                13,
            )
            start_angles = np.linspace(
                direction_angle - 0.5 * np.pi,
                direction_angle - 1.5 * np.pi,
                13,
            )
            end_cap = p1n + np.column_stack((np.cos(end_angles), np.sin(end_angles)))
            start_cap = p0n + np.column_stack((np.cos(start_angles), np.sin(start_angles)))
            footprint = np.vstack((end_cap, start_cap)) * radii
            lower = [add_point(x, y, z0) for x, y in footprint]
            upper = [add_point(x, y, z1) for x, y in footprint]
            ring_count = len(footprint)
            for i in range(ring_count):
                j = (i + 1) % ring_count
                polygons.append([lower[i], lower[j], upper[j], upper[i]])
                add_segment(lower[i], lower[j])
                add_segment(upper[i], upper[j])
                if i in (0, 12, 13, 25):
                    add_segment(lower[i], upper[i])
            # Endpoint beads, rather than a dense lattice of markers, keep the
            # thin selector readable without turning the preview into a cage.
            for point in (xy[0], xy[1]):
                marker_indices.append(add_point(float(point[0]), float(point[1]), z0))
                marker_indices.append(add_point(float(point[0]), float(point[1]), z1))
        else:
            # A narrow elliptical column is the point-selector analogue of
            # the line plane: translucent body plus sparse theta-sample dots.
            dx = float(np.median(np.diff(self._physical_x))) if self._physical_x is not None else 1.0
            dy = float(np.median(np.diff(self._physical_y))) if self._physical_y is not None else 1.0
            sx = (float(self._display_x[-1] - self._display_x[0]) /
                  float(self._physical_x[-1] - self._physical_x[0])) if self._physical_x is not None else 1.0
            sy = (float(self._display_y[-1] - self._display_y[0]) /
                  float(self._physical_y[-1] - self._physical_y[0])) if self._physical_y is not None else 1.0
            # A point selector is a ray, not a zero-area mathematical line:
            # give the ray a sub-voxel body so it remains legible at normal
            # camera distances without visually masking adjacent samples.
            thickness = float(self._roi_preview_options.get("thickness_voxels", 1.5))
            rx = max(abs(dx * sx) * 0.5 * thickness, 1e-6)
            ry = max(abs(dy * sy) * 0.5 * thickness, 1e-6)
            cx, cy = float(xy[0, 0]), float(xy[0, 1])
            ring_size = 24
            lower_ring = []
            upper_ring = []
            for i in range(ring_size):
                angle = 2.0 * np.pi * i / ring_size
                ox, oy = rx * np.cos(angle), ry * np.sin(angle)
                lower_ring.append(add_point(cx + ox, cy + oy, z0))
                upper_ring.append(add_point(cx + ox, cy + oy, z1))
            for i in range(ring_size):
                j = (i + 1) % ring_size
                polygons.append([lower_ring[i], lower_ring[j], upper_ring[j], upper_ring[i]])
                add_segment(lower_ring[i], lower_ring[j])
                add_segment(upper_ring[i], upper_ring[j])
                if i % 6 == 0:
                    add_segment(lower_ring[i], upper_ring[i])
            # The central beam remains visible even if the translucent fill is off.
            center_low = add_point(cx, cy, z0)
            center_high = add_point(cx, cy, z1)
            add_segment(center_low, center_high)
            sample_count = min(15, max(5, int(len(theta) // 5) + 1))
            for z in np.linspace(z0, z1, sample_count):
                marker_indices.append(add_point(cx, cy, float(z)))

        surface = self._roi_preview_polydata(self.vtk, coordinates, polygons, "polys")
        outline = self._roi_preview_polydata(self.vtk, coordinates, segments, "lines")
        marker_points = self._roi_preview_polydata(
            self.vtk, coordinates, marker_indices, "verts"
        )
        self._roi_preview_surface_polydata.DeepCopy(surface)
        self._roi_preview_outline_polydata.DeepCopy(outline)
        self._roi_preview_points_polydata.DeepCopy(marker_points)
        for data in (
            self._roi_preview_surface_polydata,
            self._roi_preview_outline_polydata,
            self._roi_preview_points_polydata,
        ):
            data.Modified()
        self._apply_roi_preview_visibility()

    def _style_theta_boundary_planes(self) -> None:
        """Use a restrained theme-aware accent for the optional theta caps."""
        actor = self._theta_boundary_plane_actor
        if actor is None:
            return
        if self.theme == "dark":
            accent = (0.25, 0.74, 0.69)
        else:
            accent = (0.06, 0.43, 0.40)
        prop = actor.GetProperty()
        prop.SetColor(*accent)
        prop.SetEdgeColor(*accent)
        prop.SetOpacity(float(self._theta_boundary_plane_opacity))
        prop.SetEdgeOpacity(0.92)
        prop.SetLineWidth(1.5)

    def _update_theta_boundary_planes(self) -> None:
        """Update the two optional cap planes from the current display axes.

        The marker is placed a small fraction of one local theta step inside
        the selected slab. This keeps the lower cap visually distinct from
        the coplanar XY projection while still marking the actual crop edge.
        All points stay inside the volume's X/Y/Z bounds, so enabling the
        marker cannot change camera fitting or clipping-range calculations.
        """
        actor = self._theta_boundary_plane_actor
        if actor is None:
            return
        x_axis = self._display_x
        y_axis = self._display_y
        theta = self._display_theta
        if (
            self.payload is None
            or x_axis is None
            or y_axis is None
            or theta is None
            or len(theta) < 2
        ):
            actor.SetVisibility(0)
            return

        count = int(len(theta))
        indices = self._theta_display_range_indices or (0, count - 1)
        low, high = int(indices[0]), int(indices[1])
        if not (0 <= low < high < count):
            actor.SetVisibility(0)
            return

        if low == 0:
            z_low = float(theta[0])
            low_step = float(theta[1] - theta[0])
        else:
            low_step = float(theta[low] - theta[low - 1])
            z_low = float(0.5 * (theta[low - 1] + theta[low]))
        if high == count - 1:
            z_high = float(theta[-1])
            high_step = float(theta[-1] - theta[-2])
        else:
            high_step = float(theta[high + 1] - theta[high])
            z_high = float(0.5 * (theta[high] + theta[high + 1]))

        # Nudge caps into the selected slab by 3% of the adjacent local step.
        # This avoids coplanar depth fighting with the volume's first/last
        # sample faces and, at the lower end, with the XY projection actor.
        z_low += 0.03 * low_step
        z_high -= 0.03 * high_step
        if not np.isfinite(z_low) or not np.isfinite(z_high) or not z_low < z_high:
            actor.SetVisibility(0)
            return

        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])
        if not (x0 < x1 and y0 < y1):
            actor.SetVisibility(0)
            return
        bounds = ((x0, y0, z_low), (x1, y0, z_low), (x0, y1, z_low))
        upper = ((x0, y0, z_high), (x1, y0, z_high), (x0, y1, z_high))
        for source, points in zip(self._theta_boundary_plane_sources, (bounds, upper)):
            source.SetOrigin(*points[0])
            source.SetPoint1(*points[1])
            source.SetPoint2(*points[2])
            source.Modified()
        if self._theta_boundary_plane_append is not None:
            self._theta_boundary_plane_append.Modified()
            self._theta_boundary_plane_append.Update()
        actor.SetUserMatrix(self._scene_user_matrix)
        actor.SetVisibility(1 if self._theta_boundary_planes_visible else 0)

    def _make_axis_text(self, text_value: str, position: np.ndarray, *, title: bool = False) -> Any:
        actor = self.vtk.vtkBillboardTextActor3D()
        actor.SetInput(str(text_value))
        # Billboard text actors are added directly to the renderer. VTK does not
        # apply vtkAssembly transforms consistently to billboard text, so convert
        # the local anchor to world coordinates explicitly using the very same
        # matrix as the volume/frame/ticks.
        world = self._scene_user_matrix.MultiplyPoint(
            (float(position[0]), float(position[1]), float(position[2]), 1.0)
        )
        actor.SetPosition(float(world[0]), float(world[1]), float(world[2]))
        actor.PickableOff()
        self.renderer.AddActor(actor)
        self._axis_label_actors.append(actor)
        self._axis_label_title_flags.append(bool(title))
        return actor

    def _camera_local_position(self) -> np.ndarray:
        camera = self.renderer.GetActiveCamera()
        world = (*map(float, camera.GetPosition()), 1.0)
        inverse = self.vtk.vtkMatrix4x4()
        self.vtk.vtkMatrix4x4.Invert(self._scene_user_matrix, inverse)
        local = inverse.MultiplyPoint(world)
        return np.asarray(local[:3], dtype=np.float64)

    @staticmethod
    def _format_axis_value(value: float) -> str:
        return f"{float(value):.4g}"

    def _update_geometry_axes(self, *, force: bool = False) -> None:
        """Build ticks/grid in volume-local coordinates, exactly on the frame."""
        if self.payload is None or self.grid is None:
            return
        x = np.asarray(self._display_x, dtype=np.float64)
        y = np.asarray(self._display_y, dtype=np.float64)
        z = np.asarray(self._display_theta, dtype=np.float64)
        axes = (x, y, z)
        ranges = [(float(axis[0]), float(axis[-1])) for axis in axes]
        center = np.asarray([(lo + hi) * 0.5 for lo, hi in ranges], dtype=np.float64)
        local_camera = self._camera_local_position()
        corner_key = _stable_axis_corner_bits(
            local_camera, ranges, self._axis_corner, hysteresis_fraction=0.025
        )
        corner = np.asarray([
            hi if corner_key[i] else lo for i, (lo, hi) in enumerate(ranges)
        ], dtype=np.float64)
        if not force and self._axis_corner == corner_key:
            return
        self._axis_corner = corner_key

        # Frame and tick geometry follows the scaled display axes. Numeric
        # labels are converted back to the immutable physical coordinate frame.
        axis_tick_values = [np.linspace(lo, hi, 5) for lo, hi in ranges]
        tick_segments: list[tuple[np.ndarray, np.ndarray]] = []
        grid_segments: list[tuple[np.ndarray, np.ndarray]] = []

        for old_label_actor in self._axis_label_actors:
            self.renderer.RemoveActor(old_label_actor)
        self._axis_label_actors.clear()
        self._axis_label_title_flags.clear()

        # Inward directions are deliberately orthogonal to the corresponding
        # frame edge: X ticks extend along Y, Y ticks along θ, θ ticks along X.
        inward_dimensions = (1, 2, 0)
        titles = (
            ("u [px]" if str(getattr(self.payload, "coord_system", "xy")).lower() == "uv" else "X [px]"),
            ("v [px]" if str(getattr(self.payload, "coord_system", "xy")).lower() == "uv" else "Y [px]"),
            "theta [deg]",
        )
        for axis_index, axis_values in enumerate(axis_tick_values):
            inward_index = inward_dimensions[axis_index]
            inward_lo, inward_hi = ranges[inward_index]
            inward_sign = 1.0 if corner[inward_index] == inward_lo else -1.0
            inward_extent = max(inward_hi - inward_lo, 1e-12)
            tick_length = max(inward_extent * 0.018, max(hi - lo for lo, hi in ranges) * 0.002)
            for value in axis_values:
                anchor = corner.copy()
                anchor[axis_index] = float(value)
                tick_end = anchor.copy()
                tick_end[inward_index] += inward_sign * tick_length
                tick_segments.append((anchor, tick_end))
                label_position = anchor.copy()
                # Keep numeric labels clearly outside the frame face; a shorter
                # offset makes labels collide with the frame and adjacent axes.
                label_position[inward_index] -= inward_sign * inward_extent * 0.065
                physical_axis = np.asarray(
                    (self._physical_x, self._physical_y, self._physical_theta)[axis_index],
                    dtype=np.float64,
                )
                physical_center = 0.5 * (float(physical_axis[0]) + float(physical_axis[-1]))
                display_center = 0.5 * (float(axis_values[0]) + float(axis_values[-1]))
                scale_factor = float(self._axis_scales[axis_index])
                physical_value = physical_center + (float(value) - display_center) / scale_factor
                label = self._format_axis_value(physical_value)
                self._make_axis_text(label, label_position, title=False)

            title_position = corner.copy()
            # Put the axis title beyond the positive end of its own frame edge,
            # rather than over the midpoint tick label. The perpendicular offset
            # separates it from the box and from the neighbouring axis labels.
            axis_extent = max(ranges[axis_index][1] - ranges[axis_index][0], 1e-12)
            title_position[axis_index] = ranges[axis_index][1] + axis_extent * 0.20
            title_position[inward_index] -= inward_sign * inward_extent * 0.14
            self._make_axis_text(titles[axis_index], title_position, title=True)

        # Plotly-like coordinate grid on the three camera-facing faces only.
        # Every line terminates at the same min/max coordinates as the frame.
        x_ticks, y_ticks, z_ticks = axis_tick_values
        x0, x1 = ranges[0]
        y0, y1 = ranges[1]
        z0, z1 = ranges[2]
        cx, cy, cz = map(float, corner)
        for xv in x_ticks:
            grid_segments.append((np.asarray([xv, y0, cz]), np.asarray([xv, y1, cz])))
        for yv in y_ticks:
            grid_segments.append((np.asarray([x0, yv, cz]), np.asarray([x1, yv, cz])))
        for xv in x_ticks:
            grid_segments.append((np.asarray([xv, cy, z0]), np.asarray([xv, cy, z1])))
        for zv in z_ticks:
            grid_segments.append((np.asarray([x0, cy, zv]), np.asarray([x1, cy, zv])))
        for yv in y_ticks:
            grid_segments.append((np.asarray([cx, yv, z0]), np.asarray([cx, yv, z1])))
        for zv in z_ticks:
            grid_segments.append((np.asarray([cx, y0, zv]), np.asarray([cx, y1, zv])))

        tick_data = _line_polydata(self.vtk, tick_segments)
        grid_data = _line_polydata(self.vtk, grid_segments)
        self.axis_tick_polydata.DeepCopy(tick_data)
        self.grid_polydata.DeepCopy(grid_data)
        self.axis_tick_polydata.Modified()
        self.grid_polydata.Modified()
        self.outline_filter.SetInputData(self.grid)
        self.outline_filter.Update()
        for prop in (self.outline_actor, self.axis_tick_actor, self.grid_actor):
            prop.SetUserMatrix(self._scene_user_matrix)
        has_payload = self.payload is not None
        self.outline_actor.SetVisibility(
            1 if self._bounding_box_visible and has_payload else 0
        )
        self.axis_tick_actor.SetVisibility(
            1 if self._axis_ticks_visible and has_payload else 0
        )
        for label_actor, is_title in zip(
            self._axis_label_actors, self._axis_label_title_flags
        ):
            label_actor.SetVisibility(
                1 if (is_title or self._axis_ticks_visible) and has_payload else 0
            )
        self.grid_actor.SetVisibility(
            1 if self._grid_visible and has_payload else 0
        )
        self._style_geometry_axes()

    def _on_camera_interaction(self, _caller=None, _event=None) -> None:
        if self._closed or self.payload is None:
            return
        self._update_geometry_axes(force=False)

    def set_theme(self, theme: str) -> None:
        theme = str(theme).lower()
        if theme not in ("dark", "light"):
            raise ValueError("theme должен быть 'dark' или 'light'.")
        self.theme = theme
        if theme == "dark":
            self.renderer.SetBackground(17 / 255.0, 19 / 255.0, 22 / 255.0)
        else:
            self.renderer.SetBackground(1.0, 1.0, 1.0)
        self._style_geometry_axes()
        self._style_roi_preview()
        self._update_projection_palette()

    def _update_projection_palette(self) -> None:
        """Keep the XY projection on the grayscale scale used by Plotly.

        The volume palette is controlled independently. The historical ROI
        Explorer figure uses Greys_r for the dark theme and Greys for light.
        Invalid projection samples remain transparent via the LUT below-range
        and NaN colors configured when the LUT is created.
        """
        lut = self.projection_lut
        info = self.projection_info
        if lut is None or info is None:
            return
        configured = getattr(self.settings, "projection_palette", None)
        name = resolve_projection_palette(configured, self.theme)
        colors = _palette_samples(name, 256)
        value_min = float(info.get("display_scalar_min", info["scalar_min"]))
        value_max = float(info.get("display_scalar_max", info["scalar_max"]))
        table_max = value_max if value_max > value_min else value_min + 1.0
        lut.SetNumberOfTableValues(int(colors.shape[0]))
        lut.SetTableRange(value_min, table_max)
        for index, rgb in enumerate(colors):
            lut.SetTableValue(index, float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        lut.SetBelowRangeColor(0.0, 0.0, 0.0, 0.0)
        lut.UseBelowRangeColorOn()
        lut.SetNanColor(0.0, 0.0, 0.0, 0.0)
        lut.Build()
        if self.projection_mapper is not None:
            self.projection_mapper.SetLookupTable(lut)
            self.projection_mapper.SetScalarRange(value_min, table_max)

    _THETA_PALETTE_UNIFORM_COUNT = 16

    @staticmethod
    def _theta_overlay_position_shader_code() -> str:
        """Declare the shared detector-local position before VTK's alpha guard.

        VTK evaluates the volume opacity before its ``if (g_srcColor.a > 0)``
        block. The cursor must not depend on that alpha, so its position is
        prepared at the enclosing ray-march sample scope. The same transform
        is reused by theta emphasis and the cursor marker.
        """
        return "\n".join((
            "vec3 rocOverlayDataPos = vec3(0.0);",
            "if (rocThetaSliceEnabled != 0 || rocCursorLineEnabled != 0 || rocRoiSelectionEnabled != 0) {",
            "  vec4 rocOverlayDataPosH = in_textureDatasetMatrix[0] * vec4(g_dataPos, 1.0);",
            "  float rocOverlayW = abs(rocOverlayDataPosH.w) > 1e-20 ? rocOverlayDataPosH.w : 1.0;",
            "  rocOverlayDataPos = rocOverlayDataPosH.xyz / rocOverlayW;",
            "}",
        ))

    @staticmethod
    def _theta_slice_shader_color_code() -> str:
        """Tint the volume sample for the selected theta frame.

        This snippet is inserted at VTK's premultiplication statement, which
        is inside the native ``g_srcColor.a > 0`` guard. That is intentional:
        theta highlighting changes only existing, visible volume samples.
        The independent cursor is inserted separately after the native volume
        sample has been composited, outside that guard.
        """
        lines = [
            "if (rocThetaSliceEnabled != 0 && g_srcColor.a > 0.0) {",
            "  float rocThetaDataZ = rocOverlayDataPos.z;",
            "  if (rocThetaDataZ >= rocThetaSliceLowBound && rocThetaDataZ <= rocThetaSliceHighBound) {",
            "    float rocThetaScalarRange = in_scalarsRange[0][1] - in_scalarsRange[0][0];",
            "    float rocThetaIntensity = in_scalarsRange[0][0] + rocThetaScalarRange * scalar.x;",
            "    float rocThetaT = clamp((rocThetaIntensity - rocThetaColorLow) / max(rocThetaColorHigh - rocThetaColorLow, 1e-20), 0.0, 1.0);",
            f"    float rocThetaPalettePos = rocThetaT * {float(VTKVolumeScene._THETA_PALETTE_UNIFORM_COUNT - 1):.1f};",
            f"    vec3 rocThetaHighlightColor = rocThetaPalette{VTKVolumeScene._THETA_PALETTE_UNIFORM_COUNT - 1:02d};",
        ]
        for index in range(VTKVolumeScene._THETA_PALETTE_UNIFORM_COUNT - 1):
            keyword = "if" if index == 0 else "else if"
            lines.append(
                f"    {keyword} (rocThetaPalettePos < {index + 1}.0) "
                f"{{ rocThetaHighlightColor = mix(rocThetaPalette{index:02d}, "
                f"rocThetaPalette{index + 1:02d}, rocThetaPalettePos - {index}.0); }}"
            )
        lines.extend((
            "    g_srcColor.rgb = mix(g_srcColor.rgb, rocThetaHighlightColor, clamp(rocThetaStrength, 0.0, 1.0));",
            "  }",
            "}",
        ))
        return "\n".join(lines)

    @staticmethod
    def _roi_selection_shader_color_code() -> str:
        """Color only ray-marched volume samples belonging to the live ROI.

        This code runs inside VTK's native ``g_srcColor.a > 0`` guard. It is
        deliberately color-only: transfer-function opacity, ray integration,
        depth ordering and the source scalar data remain untouched. Rectangles
        use the exact four display-space corners (a convex-quad test), while
        line/point selections use a small elliptical footprint expressed in
        displayed grid-step units.
        """
        return "\n".join((
            "if (rocRoiSelectionEnabled != 0) {",
            "  vec2 rocRoiSampleXY = rocOverlayDataPos.xy;",
            "  bool rocRoiInside = false;",
            "  if (rocRoiSelectionType == 0) {",
            "    float rocRoiCross0 = (rocRoiQuad1.x - rocRoiQuad0.x) * (rocRoiSampleXY.y - rocRoiQuad0.y) - (rocRoiQuad1.y - rocRoiQuad0.y) * (rocRoiSampleXY.x - rocRoiQuad0.x);",
            "    float rocRoiCross1 = (rocRoiQuad2.x - rocRoiQuad1.x) * (rocRoiSampleXY.y - rocRoiQuad1.y) - (rocRoiQuad2.y - rocRoiQuad1.y) * (rocRoiSampleXY.x - rocRoiQuad1.x);",
            "    float rocRoiCross2 = (rocRoiQuad3.x - rocRoiQuad2.x) * (rocRoiSampleXY.y - rocRoiQuad2.y) - (rocRoiQuad3.y - rocRoiQuad2.y) * (rocRoiSampleXY.x - rocRoiQuad2.x);",
            "    float rocRoiCross3 = (rocRoiQuad0.x - rocRoiQuad3.x) * (rocRoiSampleXY.y - rocRoiQuad3.y) - (rocRoiQuad0.y - rocRoiQuad3.y) * (rocRoiSampleXY.x - rocRoiQuad3.x);",
            "    float rocRoiOrientation = rocRoiQuadOrientation;",
            "    rocRoiInside = rocRoiOrientation * rocRoiCross0 >= -1e-6 && rocRoiOrientation * rocRoiCross1 >= -1e-6 && rocRoiOrientation * rocRoiCross2 >= -1e-6 && rocRoiOrientation * rocRoiCross3 >= -1e-6;",
            "  } else if (rocRoiSelectionType == 1) {",
            "    vec2 rocRoiSegment = rocRoiLineEnd - rocRoiLineStart;",
            "    float rocRoiSegmentLength2 = max(dot(rocRoiSegment, rocRoiSegment), 1e-20);",
            "    float rocRoiT = clamp(dot(rocRoiSampleXY - rocRoiLineStart, rocRoiSegment) / rocRoiSegmentLength2, 0.0, 1.0);",
            "    vec2 rocRoiDelta = rocRoiSampleXY - (rocRoiLineStart + rocRoiT * rocRoiSegment);",
            "    vec2 rocRoiNormalizedDelta = rocRoiDelta / max(rocRoiFootprintRadius, vec2(1e-12));",
            "    rocRoiInside = dot(rocRoiNormalizedDelta, rocRoiNormalizedDelta) <= 1.0;",
            "  } else {",
            "    vec2 rocRoiDelta = (rocRoiSampleXY - rocRoiPoint) / max(rocRoiFootprintRadius, vec2(1e-12));",
            "    rocRoiInside = dot(rocRoiDelta, rocRoiDelta) <= 1.0;",
            "  }",
            "  if (rocRoiInside) {",
            "    g_srcColor.rgb = mix(g_srcColor.rgb, rocRoiHighlightColor, clamp(rocRoiHighlightStrength, 0.0, 1.0));",
            "  } else if (rocRoiHighlightMode == 2) {",
            "    g_srcColor.rgb *= mix(1.0, 0.32, clamp(rocRoiHighlightStrength, 0.0, 1.0));",
            "  }",
            "}",
        ))

    @staticmethod
    def _cursor_line_shader_composite_code() -> str:
        """Composite a continuous cursor tube as an independent ray-march sample.

        This is inserted *after* VTK's native per-sample volume contribution
        and after the native ``if (g_srcColor.a > 0)`` block has been closed.
        Therefore the marker is evaluated even when the volume is empty or
        transparent. It is not an RGB tint of cloud samples.

        The cursor has a finite elliptical XY cross-section and spans the full
        theta axis. Its alpha is Beer-Lambert integrated over the actual ray
        step in texture-voxel units. It is composited into the same
        front-to-back accumulator: foreground cloud attenuates it, and the
        marker attenuates cloud samples behind it. No VTK actor or geometry is
        created, and no physical coordinates are offset.
        """
        lines = [
            "if (rocCursorLineEnabled != 0) {",
            "  vec2 rocCursorDelta = vec2(",
            "    (rocOverlayDataPos.x - rocCursorX) / max(rocCursorRadiusX, 1e-20),",
            "    (rocOverlayDataPos.y - rocCursorY) / max(rocCursorRadiusY, 1e-20));",
            "  float rocCursorDistance = length(rocCursorDelta);",
            "  if (rocCursorDistance < 1.0) {",
            # A broad flat core keeps the navigation marker visually continuous;
            # a smooth edge avoids aliasing without turning it into a plane.
            "    float rocCursorProfile = 1.0 - smoothstep(0.78, 1.0, rocCursorDistance);",
            "    vec3 rocCursorTextureDimensions = vec3(textureSize(in_volume[0], 0));",
            "    float rocCursorStepVoxels = length(g_dirStep * rocCursorTextureDimensions);",
            "    float rocCursorOpticalDepth = rocCursorLineDensity * rocCursorStepVoxels * rocCursorProfile;",
            "    float rocCursorSampleAlpha = 1.0 - exp(-clamp(rocCursorOpticalDepth, 0.0, 20.0));",
            "    if (rocCursorSampleAlpha > 1e-6) {",
            "      vec4 rocCursorPremultiplied = vec4(rocCursorLineColor * rocCursorSampleAlpha, rocCursorSampleAlpha);",
            "      g_fragColor = (1.0 - g_fragColor.a) * rocCursorPremultiplied + g_fragColor;",
            "    }",
            "  }",
            "}",
        ]
        return "\n".join(lines)

    def _install_theta_slice_shader(self) -> None:
        """Install the shared theta/cursor shader hooks and mutable uniforms.

        VTK's fragment custom-uniform container uploads values on every render,
        while changing an existing uniform does not change shader source or its
        shader-build timestamp. Thus theta/cursor movement does not recompile
        the shader and does not rebuild the volume texture/mapper. Three short
        post-replacement anchors are used: shared sample coordinates, theta
        RGB tint before native premultiplication, and the independent cursor
        optical contribution after the volume sample's native composite.
        """
        if self.volume is None:
            raise RuntimeError("Нельзя установить theta highlight без vtkVolume.")
        shader_property = self.volume.GetShaderProperty()
        required = ("GetFragmentCustomUniforms", "AddFragmentShaderReplacement")
        if any(not hasattr(shader_property, name) for name in required):
            raise RuntimeError(
                "Текущая сборка VTK не предоставляет OpenGL custom shader uniforms; "
                "безопасное выделение theta внутри GPU volume недоступно."
            )
        uniforms = shader_property.GetFragmentCustomUniforms()
        uniforms.SetUniformi("rocThetaSliceEnabled", 0)
        uniforms.SetUniformf("rocThetaSliceLowBound", 0.0)
        uniforms.SetUniformf("rocThetaSliceHighBound", 0.0)
        uniforms.SetUniformf("rocThetaColorLow", 0.0)
        uniforms.SetUniformf("rocThetaColorHigh", 1.0)
        uniforms.SetUniformf("rocThetaStrength", float(self._theta_slice_strength))
        uniforms.SetUniformi("rocCursorLineEnabled", 0)
        uniforms.SetUniformf("rocCursorX", 0.0)
        uniforms.SetUniformf("rocCursorY", 0.0)
        uniforms.SetUniformf("rocCursorRadiusX", 1.0)
        uniforms.SetUniformf("rocCursorRadiusY", 1.0)
        uniforms.SetUniformf("rocCursorLineDensity", float(self._cursor_line_density))
        uniforms.SetUniform3f(
            "rocCursorLineColor",
            self._hex_rgb(self._cursor_line_color),
        )
        # ROI emphasis is a color annotation inside the current GPU ray march.
        # All uniforms exist from the first shader build; selector edits only
        # update their values and never trigger another shader compilation.
        uniforms.SetUniformi("rocRoiSelectionEnabled", 0)
        uniforms.SetUniformi("rocRoiHighlightMode", 1)
        uniforms.SetUniformi("rocRoiSelectionType", 0)
        uniforms.SetUniform2f("rocRoiQuad0", (0.0, 0.0))
        uniforms.SetUniform2f("rocRoiQuad1", (0.0, 0.0))
        uniforms.SetUniform2f("rocRoiQuad2", (0.0, 0.0))
        uniforms.SetUniform2f("rocRoiQuad3", (0.0, 0.0))
        uniforms.SetUniformf("rocRoiQuadOrientation", 1.0)
        uniforms.SetUniform2f("rocRoiLineStart", (0.0, 0.0))
        uniforms.SetUniform2f("rocRoiLineEnd", (0.0, 0.0))
        uniforms.SetUniform2f("rocRoiPoint", (0.0, 0.0))
        uniforms.SetUniform2f("rocRoiFootprintRadius", (1.0, 1.0))
        uniforms.SetUniform3f("rocRoiHighlightColor", self._hex_rgb("#FFB547"))
        uniforms.SetUniformf("rocRoiHighlightStrength", 0.92)
        for index in range(self._THETA_PALETTE_UNIFORM_COUNT):
            uniforms.SetUniform3f(f"rocThetaPalette{index:02d}", (0.0, 0.0, 0.0))

        # VTK 9.6 builds the volume shader dynamically. Its public post-
        # replacement pass (ReplaceFirst=False) runs after the mapper has
        # expanded //VTK::Shading::Impl, so anchor on short, stable statements
        # rather than whitespace-sensitive multi-line fragments.
        # 1) Prepare data-local XYZ before VTK's source-alpha guard.
        shader_property.AddFragmentShaderReplacement(
            "if (g_srcColor.a > 0.0)",
            False,
            self._theta_overlay_position_shader_code() + "\nif (g_srcColor.a > 0.0)",
            False,
        )
        # 2) Color-only theta emphasis remains inside the native alpha guard.
        theta_anchor = "g_srcColor.rgb *= g_srcColor.a;"
        shader_property.AddFragmentShaderReplacement(
            theta_anchor,
            False,
            self._theta_slice_shader_color_code()
            + "\n" + self._roi_selection_shader_color_code()
            + "\n" + theta_anchor,
            False,
        )
        # 3) Add an independent optical cursor immediately after the native
        #    volume sample has been composited. Close the alpha guard before
        #    the cursor, then reopen a scope to consume VTK's original closing
        #    brace. The marker now runs for zero-alpha/background samples too.
        volume_composite_anchor = (
            "g_fragColor = (1.0f - g_fragColor.a) * g_srcColor + g_fragColor;"
        )
        shader_property.AddFragmentShaderReplacement(
            volume_composite_anchor,
            False,
            volume_composite_anchor
            + "\n        }\n"
            + self._cursor_line_shader_composite_code()
            + "\n        {",
            False,
        )
        self._theta_slice_shader_property = shader_property
        self._theta_slice_uniforms = uniforms
        self._theta_slice_shader_installed = True
        self._update_theta_slice_palette_uniforms()
        self._update_theta_slice_shader_uniforms()

    def _update_theta_slice_palette_uniforms(self) -> None:
        if self._theta_slice_uniforms is None:
            return
        colors = _palette_samples(self._theta_slice_palette, self._THETA_PALETTE_UNIFORM_COUNT)
        for index, rgb in enumerate(colors):
            self._theta_slice_uniforms.SetUniform3f(
                f"rocThetaPalette{index:02d}",
                (float(rgb[0]), float(rgb[1]), float(rgb[2])),
            )

    def _theta_slice_cell_bounds(self, index: int) -> tuple[float, float]:
        """Return the physical/display-space bin assigned to one theta frame.

        A discrete measured frame is assigned the interval bounded by the
        midpoints to its neighbours (Voronoi partition). This partitions the
        entire theta axis into non-overlapping, gap-free frame bins and works
        with nonuniform angular steps. The shader tints ray-casting samples
        whose *dataset-local* theta coordinate lies in this bin. It deliberately
        uses dataset coordinates before the ROI user matrix: the current ROI
        transform rotates/translates local X/Y into detector X/Y, while theta
        remains the dataset's local Z coordinate.
        """
        if self._display_theta is None or len(self._display_theta) == 0:
            raise RuntimeError("Координаты theta ещё не подготовлены.")
        theta = np.asarray(self._display_theta, dtype=np.float64)
        index = int(np.clip(int(index), 0, theta.size - 1))
        if theta.size == 1:
            half_width = max(float(self._min_spacing) * 0.5, 1e-6)
            return float(theta[0] - half_width), float(theta[0] + half_width)
        left_delta = (
            float(theta[index] - theta[index - 1])
            if index > 0 else float(theta[1] - theta[0])
        )
        right_delta = (
            float(theta[index + 1] - theta[index])
            if index < theta.size - 1 else float(theta[-1] - theta[-2])
        )
        edge0 = float(theta[index] - 0.5 * left_delta)
        edge1 = float(theta[index] + 0.5 * right_delta)
        return min(edge0, edge1), max(edge0, edge1)

    def _update_theta_slice_shader_uniforms(
        self, low: float | None = None, high: float | None = None
    ) -> None:
        """Update uniforms only; never mutate the volume data or shader source."""
        if self._theta_slice_uniforms is None:
            return
        if low is None or high is None:
            if self._theta_slice_transfer_range is not None:
                low, high = self._theta_slice_transfer_range
            elif self._last_transfer_info:
                low = float(self._last_transfer_info.get("threshold_min", 0.0))
                high = float(self._last_transfer_info.get("threshold_max", low + 1.0))
        if low is not None and high is not None and np.isfinite(low) and np.isfinite(high) and high > low:
            self._theta_slice_transfer_range = (float(low), float(high))
            self._theta_slice_uniforms.SetUniformf("rocThetaColorLow", float(low))
            self._theta_slice_uniforms.SetUniformf("rocThetaColorHigh", float(high))

        if self._theta_slice_index is not None and self._display_theta is not None:
            bounds = self._theta_slice_cell_bounds(self._theta_slice_index)
            self._theta_slice_bounds = bounds
            self._theta_slice_uniforms.SetUniformf("rocThetaSliceLowBound", float(bounds[0]))
            self._theta_slice_uniforms.SetUniformf("rocThetaSliceHighBound", float(bounds[1]))
        self._theta_slice_uniforms.SetUniformf("rocThetaStrength", float(self._theta_slice_strength))
        self._theta_slice_uniforms.SetUniformi(
            "rocThetaSliceEnabled", int(bool(self._theta_slice_enabled))
        )

    def _style_theta_slice_marker(self) -> None:
        """Match the restrained theme-aware style of the theta-range cap planes."""
        actor = self._theta_slice_marker_actor
        if actor is None:
            return
        if self.theme == "dark":
            accent = (0.25, 0.74, 0.69)
        else:
            accent = (0.06, 0.43, 0.40)
        prop = actor.GetProperty()
        prop.SetColor(*accent)
        prop.SetEdgeColor(*accent)
        prop.SetLighting(False)
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetOpacity(float(self._theta_slice_marker_opacity))
        prop.SetEdgeVisibility(True)
        prop.SetEdgeOpacity(0.92)
        prop.SetLineWidth(1.5)

    def _update_theta_slice_marker_geometry(self) -> None:
        """Place one abstract XY marker at the exact physical coordinate of θ[i]."""
        source = self._theta_slice_marker_source
        actor = self._theta_slice_marker_actor
        x_axis, y_axis, theta = self._display_x, self._display_y, self._display_theta
        if (
            source is None
            or actor is None
            or self.payload is None
            or x_axis is None
            or y_axis is None
            or theta is None
            or self._theta_slice_index is None
            or len(x_axis) < 2
            or len(y_axis) < 2
            or len(theta) == 0
        ):
            if actor is not None:
                actor.SetVisibility(0)
            return
        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])
        if not (np.isfinite((x0, x1, y0, y1)).all() and x0 < x1 and y0 < y1):
            actor.SetVisibility(0)
            return
        index = int(np.clip(self._theta_slice_index, 0, len(theta) - 1))
        z = float(theta[index])
        if not np.isfinite(z):
            actor.SetVisibility(0)
            return
        source.SetOrigin(x0, y0, z)
        source.SetPoint1(x1, y0, z)
        source.SetPoint2(x0, y1, z)
        source.Modified()
        actor.SetUserMatrix(self._scene_user_matrix)
        actor.GetProperty().SetOpacity(float(self._theta_slice_marker_opacity))
        actor.SetVisibility(1 if self._theta_slice_marker_enabled else 0)
        self._theta_slice_marker_mapper.Modified()

    @property
    def theta_slice_index(self) -> int | None:
        """Current selected theta-slice index (None before a volume is loaded)."""
        return self._theta_slice_index

    @property
    def theta_slice_enabled(self) -> bool:
        """Whether the in-volume theta-slice shader emphasis is enabled."""
        return bool(self._theta_slice_enabled)

    @property
    def theta_slice_palette(self) -> str:
        """Palette used by the in-volume theta-slice shader emphasis."""
        return str(self._theta_slice_palette)

    @property
    def theta_slice_strength(self) -> float:
        """Strength of the in-volume theta-slice shader emphasis."""
        return float(self._theta_slice_strength)

    @property
    def theta_slice_marker_enabled(self) -> bool:
        return bool(self._theta_slice_marker_enabled)

    @property
    def theta_slice_marker_opacity(self) -> float:
        return float(self._theta_slice_marker_opacity)

    def set_theta_slice_marker_visible(
        self, visible: bool, *, render: bool = True
    ) -> bool:
        requested = bool(visible)
        self._theta_slice_marker_enabled = requested
        self._update_theta_slice_marker_geometry()
        if render:
            self._request_render()
        return bool(
            requested
            and self.payload is not None
            and self._theta_slice_marker_actor is not None
            and self._theta_slice_marker_actor.GetVisibility()
        )

    def set_theta_slice_marker_opacity(
        self, opacity: float, *, render: bool = True
    ) -> float:
        try:
            value = float(opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Прозрачность плоскости среза θ должна быть числом.") from exc
        if not np.isfinite(value) or not (0.02 <= value <= 0.25):
            raise ValueError("Прозрачность плоскости среза θ должна лежать в [0.02, 0.25].")
        self._theta_slice_marker_opacity = value
        if self._theta_slice_marker_actor is not None:
            prop = self._theta_slice_marker_actor.GetProperty()
            prop.SetOpacity(value)
            prop.Modified()
        if render:
            self._request_render()
        return value

    @property
    def theta_slice_shader_installed(self) -> bool:
        return bool(self._theta_slice_shader_installed)

    def set_theta_slice_enabled(self, enabled: bool, *, render: bool = True) -> None:
        if self.payload is None or self._theta_slice_uniforms is None:
            if enabled:
                raise RuntimeError("Сначала загрузите объём для выделения среза θ.")
            self._theta_slice_enabled = False
            return
        self._theta_slice_enabled = bool(enabled)
        self._update_theta_slice_shader_uniforms()
        if render:
            self._request_render()

    def set_theta_slice_index(self, index: int, *, render: bool = True) -> int:
        if self.payload is None or self._theta_slice_uniforms is None:
            raise RuntimeError("Сначала загрузите объём для выбора среза θ.")
        ntheta = int(len(self.payload.theta))
        index = int(np.clip(int(index), 0, ntheta - 1))
        self._theta_slice_index = index
        self._update_theta_slice_shader_uniforms()
        self._update_theta_slice_marker_geometry()
        if render:
            self._request_render()
        return index

    def set_theta_slice_palette(self, palette: str, *, render: bool = True) -> None:
        palette = str(palette).strip()
        if not palette:
            raise ValueError("Палитра выделения theta не может быть пустой.")
        self._theta_slice_palette = palette
        self._update_theta_slice_palette_uniforms()
        
        if render:
            self._request_render()

    def set_theta_slice_strength(self, strength: float, *, render: bool = True) -> None:
        strength = float(strength)
        if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("Сила цветового выделения должна лежать в [0, 1].")
        self._theta_slice_strength = strength
        if self._theta_slice_uniforms is not None:
            self._theta_slice_uniforms.SetUniformf("rocThetaStrength", strength)
        if render:
            self._request_render()

    @staticmethod
    def _hex_rgb(color: str) -> tuple[float, float, float]:
        """Parse a strict #RRGGBB color without introducing a Qt dependency."""
        value = str(color).strip()
        if len(value) != 7 or value[0] != "#":
            raise ValueError("Цвет должен быть задан в формате #RRGGBB.")
        try:
            return tuple(int(value[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
        except ValueError as exc:
            raise ValueError("Цвет должен быть задан в формате #RRGGBB.") from exc

    @property
    def roi_preview_color(self) -> str:
        """Current ROI cloud-highlight color in normalized #RRGGBB form."""
        return str(self._roi_preview_options.get("highlight_color", "#FFB547"))

    def set_roi_preview_color(self, color: str, *, render: bool = True) -> None:
        """Update only the ROI highlight color, preserving geometry and volume state."""
        rgb = self._hex_rgb(color)
        normalized = "#" + "".join(f"{round(component * 255):02X}" for component in rgb)
        self._roi_preview_options["highlight_color"] = normalized
        if self._theta_slice_uniforms is not None:
            self._theta_slice_uniforms.SetUniform3f("rocRoiHighlightColor", rgb)
        if (
            render
            and self._roi_preview_enabled
            and str(self._roi_preview_options.get("highlight_mode", "off")) != "off"
        ):
            self._request_render()

    @property
    def cursor_line_color(self) -> str:
        return str(self._cursor_line_color)

    def set_cursor_line_color(self, color: str, *, render: bool = True) -> None:
        """Change only the cursor shader RGB uniform; keep width/optical depth separate."""
        rgb = self._hex_rgb(color)
        normalized = "#" + "".join(f"{round(component * 255):02X}" for component in rgb)
        self._cursor_line_color = normalized
        if self._theta_slice_uniforms is not None:
            self._theta_slice_uniforms.SetUniform3f("rocCursorLineColor", rgb)
        if render and self._cursor_line_enabled:
            self._request_render()

    @property
    def cursor_line_enabled(self) -> bool:
        return bool(self._cursor_line_enabled)

    @property
    def cursor_line_position(self) -> tuple[float, float] | None:
        """Current cursor location in dataset-local X/Y coordinates."""
        return self._cursor_line_position

    @property
    def cursor_line_radii(self) -> tuple[float, float] | None:
        return self._cursor_line_radii

    def set_cursor_line_position(
        self,
        x: float | None,
        y: float | None,
        radius_x: float | None = None,
        radius_y: float | None = None,
        *,
        render: bool = True,
    ) -> None:
        """Update/hide a detector-map cursor in dataset-local X/Y coordinates.

        The detector→dataset transform and radius conversion live in
        ``vtk_backend.cursor_link``. This method only updates already-declared
        GPU shader uniforms; it never changes the payload, volume grid, mapper,
        transfer functions, or renderer prop collection. ``None, None`` hides
        the marker while leaving the requested checkbox state untouched.
        """
        if x is None and y is None:
            self._cursor_line_position = None
            self._cursor_line_radii = None
            if self._theta_slice_uniforms is not None:
                self._theta_slice_uniforms.SetUniformi("rocCursorLineEnabled", 0)
            if render and self._cursor_line_enabled:
                self._request_render()
            return
        if x is None or y is None or radius_x is None or radius_y is None:
            raise ValueError("Для линии курсора нужны обе координаты X/Y и оба радиуса.")
        values = tuple(float(value) for value in (x, y, radius_x, radius_y))
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Координаты и радиусы линии курсора должны быть конечными.")
        x_value, y_value, rx, ry = values
        if rx <= 0.0 or ry <= 0.0:
            raise ValueError("Радиусы линии курсора должны быть положительными.")
        if self.payload is None or self._theta_slice_uniforms is None:
            raise RuntimeError("Сначала загрузите объём для отображения линии курсора.")
        self._cursor_line_position = (x_value, y_value)
        self._cursor_line_radii = (rx, ry)
        self._update_cursor_line_shader_uniforms()
        uniforms = self._theta_slice_uniforms
        uniforms.SetUniformi(
            "rocCursorLineEnabled",
            int(self._cursor_line_enabled),
        )
        if render and self._cursor_line_enabled:
            self._request_render()

    def set_cursor_line_density(self, density: float, *, render: bool = True) -> None:
        """Set the cursor marker's independent optical density (per texture voxel).

        This affects only a predeclared fragment uniform. It does not change
        the source scalar transfer functions, volume data, mapper or shader.
        """
        density = float(density)
        if not np.isfinite(density) or not 0.05 <= density <= 3.0:
            raise ValueError("Оптическая плотность линии должна лежать в [0.05, 3.0] 1/воксель.")
        self._cursor_line_density = density
        if self._theta_slice_uniforms is not None:
            self._theta_slice_uniforms.SetUniformf("rocCursorLineDensity", density)
        if render:
            self._request_render()

    def set_cursor_line_enabled(self, enabled: bool, *, render: bool = True) -> None:
        """Enable/disable a continuous optical marker integrated into volume rays."""
        self._cursor_line_enabled = bool(enabled)
        if self._theta_slice_uniforms is not None:
            self._theta_slice_uniforms.SetUniformi(
                "rocCursorLineEnabled",
                int(self._cursor_line_enabled and self._cursor_line_position is not None),
            )
        if render:
            self._request_render()

    def clear_cursor_line(self, *, render: bool = True) -> None:
        """Clear cursor state when data are unloaded or the scan changes."""
        self._cursor_line_enabled = False
        self._cursor_line_position = None
        self._cursor_line_radii = None
        if self._theta_slice_uniforms is not None:
            self._theta_slice_uniforms.SetUniformi("rocCursorLineEnabled", 0)
        if render:
            self._request_render()

    def set_axes_visible(self, visible: bool) -> None:
        """Show/hide only the small X/Y/Z orientation marker in the viewport corner.

        Kept under the historical public name for compatibility with clients;
        the bounding frame, ticks, labels, and grid have independent setters.
        """
        self._axes_visible = bool(visible)
        if self.axes_widget is not None:
            self.axes_widget.SetEnabled(1 if self._axes_visible else 0)
        self._request_render()

    @property
    def axes_visible(self) -> bool:
        """Whether the corner orientation marker is visible."""
        return bool(self._axes_visible)

    def set_bounding_box_visible(self, visible: bool, *, render: bool = True) -> None:
        """Show/hide only the exact volume/ROI outline frame."""
        self._bounding_box_visible = bool(visible)
        if self.outline_actor is not None:
            self.outline_actor.SetVisibility(
                1 if self._bounding_box_visible and self.payload is not None else 0
            )
        if render:
            self._request_render()

    @property
    def bounding_box_visible(self) -> bool:
        return bool(self._bounding_box_visible)

    def set_axis_ticks_visible(self, visible: bool, *, render: bool = True) -> None:
        """Show/hide tick segments and numeric tick labels, retaining axis titles."""
        self._axis_ticks_visible = bool(visible)
        has_payload = self.payload is not None
        if self.axis_tick_actor is not None:
            self.axis_tick_actor.SetVisibility(
                1 if self._axis_ticks_visible and has_payload else 0
            )
        for actor, is_title in zip(
            self._axis_label_actors, self._axis_label_title_flags
        ):
            actor.SetVisibility(
                1 if (is_title or self._axis_ticks_visible) and has_payload else 0
            )
        if render:
            self._request_render()

    @property
    def axis_ticks_visible(self) -> bool:
        return bool(self._axis_ticks_visible)

    @property
    def perspective_enabled(self) -> bool:
        """True when the active VTK camera uses perspective projection."""
        return not bool(self.renderer.GetActiveCamera().GetParallelProjection())

    @property
    def grid_visible(self) -> bool:
        return bool(self._grid_visible)

    def set_grid_visible(self, visible: bool, *, render: bool = True) -> None:
        """Show/hide the subtle face grid without changing the authoritative frame/ticks."""
        self._grid_visible = bool(visible)
        if self.grid_actor is not None:
            self.grid_actor.SetVisibility(
                1 if self._grid_visible and self.payload is not None else 0
            )
        if render:
            self._request_render()

    @tracked_activity("VTK · построение/обновление объёмной сцены")
    def set_payload(
        self,
        payload: Any,
        settings: TransferSettings | None = None,
        *,
        theta_scale: float = 1.0,
        axis_scales: tuple[float, float, float] | None = None,
    ) -> dict:
        vtk = self.vtk
        # Validate settings and construct CPU-side datasets before touching the
        # currently visible scene. Invalid controls must never destroy a valid
        # volume or leave half-updated actors in the renderer.
        settings = settings or TransferSettings()
        self._validate_transfer_settings(settings)
        # Validate the display transform before removing anything from the
        # currently visible scene. A bad programmatic value must fail without
        # leaving the renderer half-replaced.
        raw_scales = axis_scales if axis_scales is not None else (1.0, 1.0, theta_scale)
        try:
            requested_scales = tuple(float(value) for value in raw_scales)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Ожидаются три конечных коэффициента масштаба X/Y/θ.") from exc
        if len(requested_scales) != 3 or any(
            not np.isfinite(value) or not (0.1 <= value <= 10.0)
            for value in requested_scales
        ):
            raise ValueError("Коэффициенты масштабирования X/Y/θ должны лежать в [0.1, 10].")
        new_grid, new_grid_info = build_rectilinear_grid(payload)
        new_projection_grid, new_projection_info = build_projection_grid(payload)
        flat_values = np.asarray(payload.volume).reshape(-1)
        max_threshold_samples = 8_000_000
        new_sample_stride = max(1, int(np.ceil(flat_values.size / max_threshold_samples)))
        new_sample = flat_values[::new_sample_stride]
        new_finite_sample = new_sample[np.isfinite(new_sample)]
        if new_finite_sample.size == 0:
            raise ValueError("Нет конечных значений для настройки transfer function.")
        # p99.9 is invariant while the user moves Vmin/Vmax.  Compute it once
        # per payload instead of filtering and percentile-sorting up to eight
        # million samples on every slider tick.
        new_positive_sample = new_finite_sample[new_finite_sample > 0]
        if settings.lower_percentile is None and new_positive_sample.size == 0:
            raise ValueError("В объёме нет положительных конечных интенсивностей для p99.9 reference.")
        if new_positive_sample.size:
            new_positive_reference_999 = float(np.percentile(new_positive_sample, 99.9))
            new_positive_max = float(np.max(new_positive_sample))
        else:
            # Preserve the legacy percentile-mode fallback for all-nonpositive
            # volumes; fraction mode has already rejected this input above.
            new_positive_reference_999 = float(np.max(new_finite_sample))
            new_positive_max = None

        # A payload replacement ends any transient low-resolution drag mode;
        # the newly created mapper always starts with the normal full-quality
        # settings configured below.
        self.set_theta_range_interaction_active(False, render=False)

        # vtkRenderer.RemoveVolume/RemoveActor already call MakeCurrent() and
        # ReleaseGraphicsResources() on the removed prop. Do not manually
        # release the same mapper a second time: that is redundant and makes
        # GPU resource lifetime harder to reason about across driver versions.
        if self.volume is not None:
            self.renderer.RemoveVolume(self.volume)
        if self.projection_actor is not None:
            self.renderer.RemoveActor(self.projection_actor)
        self._theta_slice_shader_property = None
        self._theta_slice_uniforms = None
        self._theta_slice_shader_installed = False
        self._theta_slice_bounds = None
        self._theta_slice_transfer_range = None
        self._cursor_line_enabled = False
        self._cursor_line_position = None
        self._cursor_line_radii = None
        self._theta_boundary_planes_visible = False
        self._theta_boundary_plane_opacity = 0.12
        self._apply_roi_preview_visibility()
        if self._theta_boundary_plane_actor is not None:
            self._theta_boundary_plane_actor.SetVisibility(0)
            self._theta_boundary_plane_actor.GetProperty().SetOpacity(
                self._theta_boundary_plane_opacity
            )

        self._peak_marker_actor.SetVisibility(0)
        self._peak_marker_polydata.Initialize()
        self._peak_marker_positions_physical = np.empty((0, 3), dtype=np.float64)
        self._peak_marker_colors = np.empty((0, 4), dtype=np.uint8)
        self._peak_marker_base_colors = np.empty((0, 4), dtype=np.uint8)
        self._peak_marker_peak_ids = np.empty(0, dtype=np.int64)
        self._peak_marker_branch_ids = np.empty(0, dtype=np.int64)
        self._peak_marker_analysis_included = np.empty(0, dtype=bool)
        self._peak_marker_rendered_indices = np.empty(0, dtype=np.int64)
        self._show_excluded_branches = False
        self._peak_branch_event_counts = {}
        self._peak_excluded_branch_event_counts = {}
        self._branch_display_colors: dict[int, str] = {}
        self._branch_display_selected_ids = np.empty(0, dtype=np.int64)
        self._branch_display_focus_selected = False
        self._branch_display_previous_visibility: tuple[bool, bool] | None = None
        self._peak_marker_visible = False
        self._branch_edge_actor.SetVisibility(0)
        self._branch_edge_polydata.Initialize()
        self._branch_edge_positions_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_edge_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_edge_base_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_edge_branch_ids = np.empty(0, dtype=np.int64)
        self._branch_edge_analysis_included = np.empty(0, dtype=bool)
        self._branch_edge_visible = False
        self._branch_correlation_matched_actor.SetVisibility(0)
        self._branch_correlation_candidate_actor.SetVisibility(0)
        self._branch_correlation_matched_polydata.Initialize()
        self._branch_correlation_candidate_polydata.Initialize()
        self._branch_correlation_matched_segments_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_correlation_matched_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_correlation_candidate_segments_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_correlation_matched_peak_ids = np.empty(0, dtype=np.int64)
        self._branch_correlation_focus_branch_ids = np.empty(0, dtype=np.int64)
        self._branch_correlation_visible = False
        self._branch_correlation_show_candidates = False
        self.payload = payload
        self.grid, self.grid_info = new_grid, new_grid_info
        self.projection_grid, self.projection_info = new_projection_grid, new_projection_info
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.mapper.SetInputData(self.grid)
        # A new payload starts with its complete theta extent visible. The
        # interactive crop is a viewer-only preview and is not inherited when
        # a different HDF5 payload is loaded.
        self._theta_display_range_indices = None
        self.mapper.SetCroppingRegionFlagsToSubVolume()
        self.mapper.CroppingOff()
        self.mapper.SetBlendModeToComposite()
        # Match the working native Qt/VTK diagnostic: let VTK adapt screen-space
        # ray density to the renderer update-rate budget. Keep the data-space
        # sample distance explicit below, but do not force full-resolution rays
        # at every screen pixel during interactive use.
        self.mapper.SetAutoAdjustSampleDistances(True)
        if hasattr(self.mapper, "SetImageSampleDistance"):
            self.mapper.SetImageSampleDistance(1.0)
        if hasattr(self.mapper, "SetMinimumImageSampleDistance"):
            self.mapper.SetMinimumImageSampleDistance(1.0)
        if hasattr(self.mapper, "SetMaximumImageSampleDistance"):
            self.mapper.SetMaximumImageSampleDistance(4.0)
        if hasattr(self.mapper, "SetReportProgress"):
            self.mapper.SetReportProgress(False)
        # Jittering is an optional anti-wood-grain feature. It adds a noise
        # texture pass and is not required for scientific intensity display.
        # Keep it OFF in the baseline until the target driver is validated.
        if hasattr(self.mapper, "SetUseJittering"):
            self.mapper.SetUseJittering(False)

        self.color_function = vtk.vtkColorTransferFunction()
        self.opacity_function = vtk.vtkPiecewiseFunction()
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetShade(False)
        self.volume_property.SetScalarOpacityUnitDistance(
            float(compute_opacity_unit_distance(self._min_spacing, settings.opacity_unit_distance_factor))
        )
        # The gradient-opacity function is neutral (constant 1.0) in this
        # renderer.  Do not allocate a second GPU opacity lookup texture for a
        # function that cannot change the image.
        self.volume_property.DisableGradientOpacityOn()

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.volume_property)
        if str(getattr(payload, "coord_system", "xy")).lower() == "uv":
            transform = roi_local_to_detector_matrix(payload.roi)
        else:
            transform = vtk.vtkMatrix4x4()
            transform.Identity()
        self._scene_base_matrix.DeepCopy(transform)
        self._scene_user_matrix.DeepCopy(transform)
        self.volume.SetUserMatrix(self._scene_user_matrix)
        self.renderer.AddVolume(self.volume)
        self.volume.SetVisibility(1 if self._volume_visible else 0)
        self._install_theta_slice_shader()

        projection_min = float(self.projection_info.get("display_scalar_min", self.projection_info["scalar_min"]))
        projection_max = float(self.projection_info.get("display_scalar_max", self.projection_info["scalar_max"]))
        self.projection_lut = vtk.vtkLookupTable()
        self.projection_lut.SetNumberOfTableValues(256)
        self.projection_lut.SetTableRange(
            projection_min,
            projection_max if projection_max > projection_min else projection_min + 1.0,
        )
        self.projection_lut.SetBelowRangeColor(0.0, 0.0, 0.0, 0.0)
        self.projection_lut.UseBelowRangeColorOn()
        self.projection_lut.SetNanColor(0.0, 0.0, 0.0, 0.0)
        self.projection_mapper = vtk.vtkDataSetMapper()
        self.projection_mapper.SetInputData(self.projection_grid)
        self.projection_mapper.SetLookupTable(self.projection_lut)
        self.projection_mapper.SetScalarRange(
            projection_min,
            projection_max if projection_max > projection_min else projection_min + 1.0,
        )
        self.projection_mapper.SetScalarModeToUsePointData()
        self.projection_mapper.SetColorModeToMapScalars()
        self.projection_actor = vtk.vtkActor()
        self.projection_actor.SetMapper(self.projection_mapper)
        projection_property = self.projection_actor.GetProperty()
        projection_property.SetOpacity(float(settings.projection_opacity))
        # XY projection is a scalar image plane. Scene lighting must not shade
        # its grayscale/color-map values according to the 3D camera direction.
        projection_property.LightingOff()
        projection_property.SetAmbient(1.0)
        projection_property.SetDiffuse(0.0)
        self.projection_actor.SetVisibility(bool(settings.show_projection))
        self.projection_actor.PickableOff()
        self.projection_actor.SetUserMatrix(self._scene_user_matrix)
        self.renderer.AddActor(self.projection_actor)

        self._percentile_sample_stride = new_sample_stride
        self._finite_sample = new_finite_sample
        self._positive_reference_999 = new_positive_reference_999
        self._positive_max = new_positive_max
        self._physical_x = np.asarray(payload.x, dtype=np.float64).copy()
        self._physical_y = np.asarray(payload.y, dtype=np.float64).copy()
        self._physical_theta = np.asarray(payload.theta, dtype=np.float64).copy()
        self._display_x = self._physical_x.copy()
        self._display_y = self._physical_y.copy()
        self._display_theta = self._physical_theta.copy()
        if self._theta_slice_index is None:
            self._theta_slice_index = int(self._physical_theta.size // 2)
        else:
            self._theta_slice_index = int(np.clip(
                self._theta_slice_index, 0, max(0, self._physical_theta.size - 1)
            ))
        self._projection_base_z = float(self.projection_info["z_plane"])
        self.settings = settings
        self._axis_scales = (1.0, 1.0, 1.0)
        self._theta_scale = 1.0
        self.set_axis_scales(*requested_scales, render=False)
        self._update_theta_boundary_planes()
        self._update_roi_preview_geometry()
        # Actor transforms contain only the ROI-to-detector transform.
        # Display scale lives in the rectilinear-grid coordinate arrays.
        for prop in (self.outline_actor, self.axis_tick_actor, self.grid_actor):
            if prop is not None:
                prop.SetUserMatrix(self._scene_user_matrix)
        self.set_theme(self.theme)
        self._last_transfer_info = self.update_transfer_functions(self.settings, render=False)
        self._update_theta_slice_shader_uniforms()
        self._update_theta_slice_marker_geometry()
        
        # Establish a non-degenerate oblique camera BEFORE fitting. The previous
        # sequence called ResetCamera(), then assigned world-Z as view-up while
        # the default view direction could be parallel to Z, causing VTK's
        # "Resetting view-up" warning and relying on an implicit correction.
        bounds = self.renderer.ComputeVisiblePropBounds()
        if len(bounds) == 6 and np.all(np.isfinite(bounds)) and bounds[0] <= bounds[1]:
            center = np.array([
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ], dtype=float)
            diagonal = float(np.linalg.norm([
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ]))
            radius = max(diagonal, 1.0)
            camera = self.renderer.GetActiveCamera()
            camera.SetFocalPoint(*center.tolist())
            camera.SetPosition(*(center + np.array([1.7, -2.3, 1.6]) * radius).tolist())
            camera.SetViewUp(0.0, 0.0, 1.0)
            camera.OrthogonalizeViewUp()
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._update_geometry_axes(force=True)
        self._request_render()

        # Do not call IsRenderSupported here: with QVTK's deferred paint model,
        # the OpenGL context may not yet be current. The first scheduled paint
        # performs the actual mapper initialization in the Qt paint event.
        return {
            **self.grid_info,
            "gpu_supported": None,
            "threshold_sample_count": int(self._finite_sample.size),
            "threshold_sample_stride": int(self._percentile_sample_stride),
            "sample_distance": float(self.mapper.GetSampleDistance()),
            "opacity_unit_distance": float(self.volume_property.GetScalarOpacityUnitDistance()),
            "threshold_reference": float(self._last_transfer_info.get("threshold_reference", np.nan)),
            "threshold_min": float(self._last_transfer_info.get("threshold_min", np.nan)),
            "threshold_max": float(self._last_transfer_info.get("threshold_max", np.nan)),
            "projection_visible": bool(self.projection_actor.GetVisibility()) if self.projection_actor is not None else False,
            "projection_bounds": tuple(float(v) for v in self.projection_actor.GetBounds()) if self.projection_actor is not None else None,
            "projection_dimensions": tuple(int(v) for v in self.projection_grid.GetDimensions()) if self.projection_grid is not None else None,
            "auto_adjust_sample_distances": bool(self.mapper.GetAutoAdjustSampleDistances()),
            "jittering": bool(self.mapper.GetUseJittering()) if hasattr(self.mapper, "GetUseJittering") else None,
            "theta_scale": float(self._theta_scale),
            "axis_scales": tuple(float(v) for v in self._axis_scales),
            "theta_display_crop_indices": self._theta_display_range_indices,
            "theta_display_crop_physical_range": (
                (
                    float(self._physical_theta[self._theta_display_range_indices[0]]),
                    float(self._physical_theta[self._theta_display_range_indices[1]]),
                )
                if self._theta_display_range_indices is not None and self._physical_theta is not None
                else (
                    (float(self._physical_theta[0]), float(self._physical_theta[-1]))
                    if self._physical_theta is not None else None
                )
            ),
            "theta_display_crop_enabled": bool(self.mapper.GetCropping()) if self.mapper is not None else False,
            "theta_display_crop_planes": (
                tuple(float(v) for v in self.mapper.GetCroppingRegionPlanes())
                if self.mapper is not None else None
            ),
            "theta_display_bounds": (
                tuple(float(v) for v in self.projection_actor.GetBounds()[4:6]) if self.projection_actor is not None else None
            ),
        }

    @staticmethod
    def _estimate_min_spacing(
        payload: Any,
        theta_scale: float = 1.0,
        *,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
    ) -> float:
        """Smallest displayed voxel spacing after the display-only scaling."""
        scales = (float(x_scale), float(y_scale), float(theta_scale))
        if any(not np.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("Коэффициенты масштаба должны быть конечными и положительными.")
        diffs = []
        for axis, multiplier in zip((payload.x, payload.y, payload.theta), scales):
            coordinates = np.asarray(axis, dtype=float)
            if coordinates.size > 1:
                delta = np.diff(coordinates) * multiplier
                delta = delta[np.isfinite(delta) & (delta > 0)]
                if delta.size:
                    diffs.append(float(np.min(delta)))
        return min(diffs) if diffs else 1.0

    @property
    def axis_scales(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._axis_scales)

    @staticmethod
    def _scaled_axis_coordinates(values: np.ndarray, factor: float) -> np.ndarray:
        """Scale one display axis about its physical midpoint without accumulation."""
        source = np.asarray(values, dtype=np.float64)
        if source.ndim != 1 or source.size < 2 or not np.all(np.isfinite(source)):
            raise ValueError("Координаты оси должны быть конечным одномерным массивом минимум из 2 узлов.")
        center = 0.5 * (float(source[0]) + float(source[-1]))
        displayed = np.ascontiguousarray(center + (source - center) * float(factor), dtype=np.float64)
        if not np.all(np.isfinite(displayed)) or not np.all(np.diff(displayed) > 0.0):
            raise ValueError("Масштабирование привело к некорректной координатной оси.")
        return displayed

    @staticmethod
    def _vtk_coordinate_array(numpy_to_vtk, values: np.ndarray):
        array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
        array.SetNumberOfComponents(1)
        return array

    def _update_cursor_line_shader_uniforms(self) -> None:
        """Map detector-local physical cursor values onto the displayed X/Y grid."""
        uniforms = self._theta_slice_uniforms
        if (
            uniforms is None
            or self._cursor_line_position is None
            or self._cursor_line_radii is None
            or self._physical_x is None
            or self._physical_y is None
        ):
            return
        sx, sy, _ = self._axis_scales
        center_x = 0.5 * (float(self._physical_x[0]) + float(self._physical_x[-1]))
        center_y = 0.5 * (float(self._physical_y[0]) + float(self._physical_y[-1]))
        x, y = self._cursor_line_position
        radius_x, radius_y = self._cursor_line_radii
        uniforms.SetUniformf("rocCursorX", float(center_x + (float(x) - center_x) * sx))
        uniforms.SetUniformf("rocCursorY", float(center_y + (float(y) - center_y) * sy))
        uniforms.SetUniformf("rocCursorRadiusX", float(radius_x * sx))
        uniforms.SetUniformf("rocCursorRadiusY", float(radius_y * sy))

    def _apply_theta_display_crop(self) -> None:
        """Apply the current inclusive theta interval using GPU mapper clipping.

        The clipping planes are expressed in the current *display-grid*
        coordinates, while the selected indices refer to the immutable payload
        axis. Boundaries are placed halfway between neighboring theta samples
        (and half a local step beyond the first/last sample) so non-uniformly
        spaced samples are not cut through their centers.
        """
        mapper = self.mapper
        if mapper is None:
            return
        indices = self._theta_display_range_indices
        theta = self._display_theta
        x_axis = self._display_x
        y_axis = self._display_y
        if indices is None or theta is None or x_axis is None or y_axis is None:
            mapper.CroppingOff()
            mapper.Modified()
            return

        count = int(len(theta))
        low, high = (int(indices[0]), int(indices[1]))
        if not (0 <= low < high < count):
            raise ValueError(
                "Индексы предпросмотра θ должны удовлетворять 0 ≤ low < high < N."
            )
        if low == 0 and high == count - 1:
            mapper.CroppingOff()
            mapper.Modified()
            return

        # VTK volume cropping operates in the input dataset's coordinates.
        # X/Y are deliberately left at the complete data bounds; only Z=theta
        # is narrowed. This does not touch the separate XY projection actor.
        x_min, x_max = float(x_axis[0]), float(x_axis[-1])
        y_min, y_max = float(y_axis[0]), float(y_axis[-1])
        if low == 0:
            z_min = float(theta[0] - 0.5 * (theta[1] - theta[0]))
        else:
            z_min = float(0.5 * (theta[low - 1] + theta[low]))
        if high == count - 1:
            z_max = float(theta[-1] + 0.5 * (theta[-1] - theta[-2]))
        else:
            z_max = float(0.5 * (theta[high] + theta[high + 1]))

        planes = (x_min, x_max, y_min, y_max, z_min, z_max)
        if not all(np.isfinite(value) for value in planes):
            raise ValueError("Плоскости предпросмотра θ содержат нечисловые координаты.")
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError("Некорректная геометрия диапазона предпросмотра θ.")

        mapper.SetCroppingRegionPlanes(*planes)
        mapper.SetCroppingRegionFlagsToSubVolume()
        mapper.CroppingOn()
        mapper.Modified()

    def set_theta_boundary_planes_visible(
        self, visible: bool, *, render: bool = True
    ) -> bool:
        """Show or hide the two non-interactive visual caps of the θ preview."""
        requested = bool(visible)
        self._theta_boundary_planes_visible = requested
        if requested:
            self._update_theta_boundary_planes()
        elif self._theta_boundary_plane_actor is not None:
            self._theta_boundary_plane_actor.SetVisibility(0)
        if render:
            self._request_render()
        return bool(
            requested
            and self.payload is not None
            and self._theta_boundary_plane_actor is not None
            and self._theta_boundary_plane_actor.GetVisibility()
        )

    def set_theta_boundary_plane_opacity(
        self, opacity: float, *, render: bool = True
    ) -> float:
        """Set the subtle fill opacity of the two θ boundary caps (2..25%)."""
        try:
            value = float(opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Прозрачность граничных плоскостей должна быть числом.") from exc
        if not np.isfinite(value) or not (0.02 <= value <= 0.25):
            raise ValueError("Прозрачность граничных плоскостей должна лежать в [0.02, 0.25].")
        self._theta_boundary_plane_opacity = value
        if self._theta_boundary_plane_actor is not None:
            prop = self._theta_boundary_plane_actor.GetProperty()
            prop.SetOpacity(value)
            prop.Modified()
        if render:
            self._request_render()
        return value

    @property
    def theta_boundary_planes_visible(self) -> bool:
        return bool(self._theta_boundary_planes_visible)

    @property
    def theta_boundary_plane_opacity(self) -> float:
        return float(self._theta_boundary_plane_opacity)

    def set_theta_display_range_indices(
        self,
        low_index: int,
        high_index: int,
        *,
        render: bool = True,
    ) -> tuple[int, int]:
        """Preview an inclusive theta sample interval without rebuilding data.

        Only vtkGPUVolumeRayCastMapper's cropping planes are changed. The
        HDF5 payload, scalar array, rectilinear grid, camera, actor transform,
        axis scales, and independent XY projection remain untouched.
        """
        if self.payload is None or self.mapper is None or self._display_theta is None:
            raise RuntimeError("Сначала загрузите 3D-объём.")
        try:
            low = int(low_index)
            high = int(high_index)
            if (
                isinstance(low_index, (bool, np.bool_))
                or isinstance(high_index, (bool, np.bool_))
                or float(low_index) != float(low)
                or float(high_index) != float(high)
            ):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Индексы диапазона θ должны быть целыми числами.") from exc
        count = int(len(self._display_theta))
        if not (0 <= low < high < count):
            raise ValueError(
                f"Ожидаются индексы 0 ≤ low < high < {count}, получено ({low}, {high})."
            )

        # Store None for the complete range so the mapper returns to its native
        # uncropped path (not a full-bounds clipping pass).
        requested = None if (low == 0 and high == count - 1) else (low, high)
        previous_indices = self._theta_display_range_indices
        # Ignore duplicate snapped mouse events before making any native VTK
        # queries. The continuous toolbar slider can emit repeatedly while
        # both handles remain on the same discrete θ samples.
        if requested == previous_indices:
            return (low, high)

        previous_enabled = bool(self.mapper.GetCropping())
        previous_planes = tuple(float(v) for v in self.mapper.GetCroppingRegionPlanes())
        previous_flags = int(self.mapper.GetCroppingRegionFlags())
        self._theta_display_range_indices = requested
        try:
            self._apply_theta_display_crop()
            if self._theta_boundary_planes_visible:
                self._update_theta_boundary_planes()
        except Exception:
            self._theta_display_range_indices = previous_indices
            try:
                self.mapper.SetCroppingRegionPlanes(*previous_planes)
                self.mapper.SetCroppingRegionFlags(previous_flags)
                self.mapper.SetCropping(1 if previous_enabled else 0)
                self.mapper.Modified()
                self._update_theta_boundary_planes()
            except Exception:
                pass
            raise

        if self._roi_preview_enabled:
            self._update_roi_preview_geometry()
        if render:
            self._request_render()
        return (low, high)

    def set_theta_range_interaction_active(
        self, active: bool, *, render: bool = True
    ) -> bool:
        """Temporarily trade 3D screen sampling for responsive linked dragging.

        This changes only vtkGPUVolumeRayCastMapper's screen-space image sample
        distance. The volume data, voxel sampling distance, transfer functions,
        crop coordinates, camera, and all scientific results remain untouched.
        Original mapper settings are restored exactly when the gesture ends.
        """
        mapper = self.mapper
        if mapper is None:
            self._theta_range_interaction_render_state = None
            return False
        if not hasattr(mapper, "SetImageSampleDistance"):
            return False

        if bool(active):
            if self._theta_range_interaction_render_state is not None:
                return False
            try:
                original_distance = float(mapper.GetImageSampleDistance())
                original_auto_adjust = bool(mapper.GetAutoAdjustSampleDistances())
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return False
            if not np.isfinite(original_distance) or original_distance <= 0:
                return False

            self._theta_range_interaction_render_state = (
                original_distance, original_auto_adjust
            )
            try:
                # Lock auto-adjust for the duration of the gesture. A modest
                # 1.5x screen-space step renders ~44% as many rays as 1.0 while
                # keeping the live volume preview legible. This is display-only.
                if hasattr(mapper, "SetAutoAdjustSampleDistances"):
                    mapper.SetAutoAdjustSampleDistances(False)
                mapper.SetImageSampleDistance(max(original_distance, 1.5))
                mapper.Modified()
            except Exception:
                self._theta_range_interaction_render_state = None
                try:
                    mapper.SetImageSampleDistance(original_distance)
                    if hasattr(mapper, "SetAutoAdjustSampleDistances"):
                        mapper.SetAutoAdjustSampleDistances(original_auto_adjust)
                    mapper.Modified()
                except Exception:
                    pass
                raise
        else:
            previous = self._theta_range_interaction_render_state
            if previous is None:
                return False
            original_distance, original_auto_adjust = previous
            try:
                mapper.SetImageSampleDistance(float(original_distance))
                if hasattr(mapper, "SetAutoAdjustSampleDistances"):
                    mapper.SetAutoAdjustSampleDistances(bool(original_auto_adjust))
                mapper.Modified()
            except Exception:
                # Keep the restore token so a later release/cleanup can retry.
                self._theta_range_interaction_render_state = previous
                raise
            self._theta_range_interaction_render_state = None

        if render:
            self._request_render()
        return True

    @property
    def theta_display_range_indices(self) -> tuple[int, int] | None:
        """Current GPU-preview crop; None represents the complete theta extent."""
        return self._theta_display_range_indices

    @staticmethod
    def _branch_rgba(branch_ids: np.ndarray) -> np.ndarray:
        """Stable categorical RGBA colours keyed by integer branch_id."""
        import colorsys

        ids = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
        palette = np.empty((ids.size, 4), dtype=np.uint8)
        for row, value in enumerate(ids):
            hue = (int(value) * 0.618033988749895) % 1.0
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
            palette[row] = (
                int(round(red * 255)), int(round(green * 255)),
                int(round(blue * 255)), 255,
            )
        return palette

    @staticmethod
    def _numeric_column(values: Any) -> np.ndarray:
        """Convert a nullable table column to float, mapping missing values to NaN.

        Pandas nullable dtypes (for example ``Int64`` with ``pd.NA``) cannot
        always be passed directly to ``np.asarray(..., dtype=float)``.  Keep
        this conversion local to the presentation adapter and never mutate the
        source catalogue.
        """
        if hasattr(values, "to_numpy"):
            try:
                return np.asarray(values.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
            except (TypeError, ValueError):
                pass
        raw = np.asarray(values, dtype=object).reshape(-1)
        result = np.empty(raw.size, dtype=np.float64)
        for index, value in enumerate(raw):
            try:
                result[index] = float(value)
            except (TypeError, ValueError, OverflowError):
                result[index] = np.nan
        return result

    @staticmethod
    def _hex_rgba(color: str) -> np.ndarray:
        """Validate a user-selected #RRGGBB colour and return opaque RGBA."""
        raw_hex = str(color).strip().lstrip("#")
        if len(raw_hex) != 6:
            raise ValueError(f"Некорректный цвет маркеров: {color!r}")
        try:
            rgb = tuple(int(raw_hex[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError as exc:
            raise ValueError(f"Некорректный цвет маркеров: {color!r}") from exc
        return np.asarray([*rgb, 255], dtype=np.uint8)

    def _peak_catalog_geometry(
        self, peak_catalog: Any | None, x_coords: Any, y_coords: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ROI-clipped physical peak XYZ positions and source row indices."""
        if peak_catalog is None or getattr(peak_catalog, "empty", False):
            return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
        columns = set(getattr(peak_catalog, "columns", ()))
        missing = {"x", "y", "theta"}.difference(columns)
        if missing:
            raise ValueError(f"Каталог центров пиков не содержит столбцы {sorted(missing)}")
        source_x = np.asarray(x_coords, dtype=np.float64)
        source_y = np.asarray(y_coords, dtype=np.float64)
        if source_x.ndim != 1 or source_y.ndim != 1:
            raise ValueError("Координаты исходной детекторной сетки должны быть одномерными")
        x_index = self._numeric_column(peak_catalog["x"])
        y_index = self._numeric_column(peak_catalog["y"])
        theta = self._numeric_column(peak_catalog["theta"])
        finite = np.isfinite(x_index) & np.isfinite(y_index) & np.isfinite(theta)
        integer = (x_index == np.rint(x_index)) & (y_index == np.rint(y_index))
        in_range = (
            (x_index >= 0) & (x_index < source_x.size)
            & (y_index >= 0) & (y_index < source_y.size)
        )
        valid_rows = np.flatnonzero(finite & integer & in_range)
        if valid_rows.size == 0:
            return np.empty((0, 3), dtype=np.float64), valid_rows
        xi = np.rint(x_index[valid_rows]).astype(np.int64)
        yi = np.rint(y_index[valid_rows]).astype(np.int64)
        physical = np.column_stack((source_x[xi], source_y[yi], theta[valid_rows]))
        # An ROI-specific volume may expose a strict sub-grid. Apply the same
        # physical bounds to points and links, including the theta dimension.
        if self._physical_x is not None and self._physical_y is not None:
            # Axes may be strictly ascending OR descending (ROI and scan views
            # preserve the source orientation).  Compare against physical bounds,
            # not first/last samples; the latter silently removed every marker
            # for a descending detector axis while correspondence lines used
            # the correct orientation-independent mask.
            x_low, x_high = sorted((float(self._physical_x[0]), float(self._physical_x[-1])))
            y_low, y_high = sorted((float(self._physical_y[0]), float(self._physical_y[-1])))
            in_roi = (
                (physical[:, 0] >= x_low)
                & (physical[:, 0] <= x_high)
                & (physical[:, 1] >= y_low)
                & (physical[:, 1] <= y_high)
            )
            if self._physical_theta is not None and len(self._physical_theta):
                theta_low, theta_high = sorted((
                    float(self._physical_theta[0]), float(self._physical_theta[-1])
                ))
                in_roi &= (physical[:, 2] >= theta_low) & (physical[:, 2] <= theta_high)
            physical = physical[in_roi]
            valid_rows = valid_rows[in_roi]
        return np.ascontiguousarray(physical, dtype=np.float64), valid_rows

    @staticmethod
    def _stable_id_key(value: Any) -> Any:
        """Normalize scalar catalogue IDs without conflating distinct strings."""
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or type(value).__name__ in {"NAType", "NaTType"}:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return None
            if float(value).is_integer():
                return int(value)
            return float(value)
        return value if value is not None else None

    def set_peak_markers(
        self,
        peak_catalog: Any | None,
        x_coords: Any,
        y_coords: Any,
        *,
        visible: bool = False,
        point_size: float = 7.0,
        opacity: float = 0.95,
        color_by_branch: bool = True,
        uniform_color: str = "#FFC857",
        show_excluded_branches: bool | None = None,
        render: bool = True,
    ) -> int:
        """Set detected/fitted peak centres as a separate display-only layer.

        Missing ``branch_id`` values deliberately keep the uniform fallback
        colour.  They are not filtered out: colouring is a presentation choice,
        never a data-validity condition.
        """
        try:
            point_size = float(point_size)
            opacity = float(opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Размер и непрозрачность маркеров должны быть числами") from exc
        if not np.isfinite(point_size) or not (1.0 <= point_size <= 20.0):
            raise ValueError("Диаметр маркеров должен находиться в диапазоне 1–20 px")
        if not np.isfinite(opacity) or not (0.05 <= opacity <= 1.0):
            raise ValueError("Непрозрачность маркеров должна находиться в диапазоне 0.05–1.0")

        self._peak_marker_size = point_size
        self._peak_marker_opacity = opacity
        self._peak_marker_actor.GetProperty().SetOpacity(opacity)
        physical, valid_rows = self._peak_catalog_geometry(peak_catalog, x_coords, y_coords)
        fallback = self._hex_rgba(uniform_color)
        colors = np.repeat(fallback[None, :], physical.shape[0], axis=0)
        columns = set(getattr(peak_catalog, "columns", ())) if peak_catalog is not None else set()

        # Keep picking metadata aligned one-to-one with the input points passed
        # to vtkGlyph3DMapper. Full branch counts are computed before the
        # display-only glyph cap, so tooltip/legend counts remain scientifically
        # exact even when a very large cloud is thinned for rendering.
        peak_ids = np.full(physical.shape[0], -1, dtype=np.int64)
        if "peak_id" in columns and physical.shape[0]:
            peak_id_values = self._numeric_column(peak_catalog["peak_id"])[valid_rows]
            peak_id_valid = np.isfinite(peak_id_values) & (peak_id_values >= 0)
            peak_ids[peak_id_valid] = np.rint(peak_id_values[peak_id_valid]).astype(np.int64)

        branch_ids = np.full(physical.shape[0], -1, dtype=np.int64)
        if "analysis_included" in columns and physical.shape[0]:
            raw_included = np.asarray(peak_catalog["analysis_included"], dtype=object)[valid_rows]
            analysis_included = np.asarray([
                (bool(value) if isinstance(value, (bool, np.bool_))
                 else str(value).strip().lower() in {"true", "1", "yes", "y"})
                for value in raw_included
            ], dtype=bool)
        else:
            # Older immutable artifacts predate branch size selection.
            analysis_included = np.ones(physical.shape[0], dtype=bool)

        self._peak_branch_event_counts = {}
        self._peak_excluded_branch_event_counts = {}
        if "branch_id" in columns and physical.shape[0]:
            branch_values = self._numeric_column(peak_catalog["branch_id"])[valid_rows]
            branch_valid = (
                np.isfinite(branch_values) & (branch_values >= 0)
                & (branch_values == np.rint(branch_values))
            )
            branch_ids[branch_valid] = np.rint(branch_values[branch_valid]).astype(np.int64)
            if np.any(branch_valid & analysis_included):
                accepted_ids, accepted_counts = np.unique(
                    branch_ids[branch_valid & analysis_included], return_counts=True
                )
                self._peak_branch_event_counts = {
                    int(branch_id): int(count)
                    for branch_id, count in zip(accepted_ids, accepted_counts)
                }
            if np.any(branch_valid & ~analysis_included):
                excluded_ids, excluded_counts = np.unique(
                    branch_ids[branch_valid & ~analysis_included], return_counts=True
                )
                self._peak_excluded_branch_event_counts = {
                    int(branch_id): int(count)
                    for branch_id, count in zip(excluded_ids, excluded_counts)
                }
            if color_by_branch and np.any(branch_valid & analysis_included):
                colors[branch_valid & analysis_included] = self._branch_rgba(
                    branch_ids[branch_valid & analysis_included]
                )

        # Excluded components are always visually neutral, even when their
        # optional grouped display is enabled. They never inherit branch hues.
        if np.any(~analysis_included):
            colors[~analysis_included] = np.asarray([132, 139, 149, 255], dtype=np.uint8)
        if show_excluded_branches is not None:
            self._show_excluded_branches = bool(show_excluded_branches)

        if physical.shape[0] > 150_000:
            # Deterministic display-only thinning. Scientific catalogues remain
            # complete; this cap bounds VTK glyph/cell memory and frame time.
            indices = np.linspace(0, physical.shape[0] - 1, 150_000, dtype=np.int64)
            physical = physical[indices]
            colors = colors[indices]
            peak_ids = peak_ids[indices]
            branch_ids = branch_ids[indices]
            analysis_included = analysis_included[indices]

        self._peak_marker_positions_physical = np.ascontiguousarray(physical, dtype=np.float64)
        self._peak_marker_base_colors = np.ascontiguousarray(colors, dtype=np.uint8)
        self._peak_marker_peak_ids = np.ascontiguousarray(peak_ids, dtype=np.int64)
        self._peak_marker_branch_ids = np.ascontiguousarray(branch_ids, dtype=np.int64)
        self._peak_marker_analysis_included = np.ascontiguousarray(analysis_included, dtype=bool)
        self._apply_branch_correlation_peak_emphasis(update_polydata=False)
        self._update_peak_marker_polydata()
        self._peak_marker_visible = bool(visible)
        has_rendered_points = bool(self._peak_marker_rendered_indices.size)
        self._peak_marker_actor.SetVisibility(self._peak_marker_visible and has_rendered_points)
        if render:
            self._request_render()
        # Return the available catalog-overlay point count, not the current
        # visible subset: the legend may still offer an opt-in to show excluded
        # branches even when the default accepted-only subset is empty.
        return int(physical.shape[0])

    @staticmethod
    def _branch_hex_color(branch_id: int) -> str:
        rgba = VTKVolumeScene._branch_rgba(np.asarray([int(branch_id)], dtype=np.int64))[0]
        return "#{:02X}{:02X}{:02X}".format(int(rgba[0]), int(rgba[1]), int(rgba[2]))

    def branch_legend_data(self, *, limit: int = 40) -> dict[str, Any]:
        """Return the active 3D branch palette without conflating exclusions.

        In normal mode, counts and rows retain the established accepted-only
        contract; excluded branches remain one separate aggregate. During an
        explicit Branch Manager focus, the legend instead lists exactly the
        selected objects, including an individually selected excluded branch.
        """
        limit = max(1, min(200, int(limit)))
        accepted = dict(getattr(self, "_peak_branch_event_counts", {}) or {})
        excluded = dict(getattr(self, "_peak_excluded_branch_event_counts", {}) or {})
        color_map = getattr(self, "_branch_display_colors", {}) or {}
        focus = bool(getattr(self, "_branch_display_focus_selected", False))
        selected = np.asarray(
            getattr(self, "_branch_display_selected_ids", np.empty(0)), dtype=np.int64
        )
        if focus and selected.size:
            all_counts = {int(key): int(value) for key, value in accepted.items()}
            all_counts.update({int(key): int(value) for key, value in excluded.items()})
            selected_ids = sorted({int(value) for value in selected if int(value) >= 0})
            ordered = sorted(
                selected_ids,
                key=lambda branch_id: (-int(all_counts.get(branch_id, 0)), branch_id),
            )
            entries = []
            for branch_id in ordered[:limit]:
                included = branch_id in accepted
                color = str(color_map.get(branch_id, self._branch_hex_color(branch_id)))
                entries.append({
                    "branch_id": branch_id,
                    "event_count": int(all_counts.get(branch_id, 0)),
                    "color": color,
                    "analysis_included": bool(included),
                })
            return {
                "total_branches": int(len(accepted)),
                "total_events": int(sum(int(value) for value in accepted.values())),
                "excluded_branches": int(len(excluded)),
                "excluded_events": int(sum(int(value) for value in excluded.values())),
                "show_excluded": bool(getattr(self, "_show_excluded_branches", False)),
                "focus_selected": True,
                "selected_branches": int(len(selected_ids)),
                "entries": entries,
                "truncated": bool(len(ordered) > limit),
            }

        ordered = sorted(
            accepted.items(), key=lambda item: (-int(item[1]), int(item[0]))
        )
        return {
            "total_branches": int(len(accepted)),
            "total_events": int(sum(int(value) for value in accepted.values())),
            "excluded_branches": int(len(excluded)),
            "excluded_events": int(sum(int(value) for value in excluded.values())),
            "show_excluded": bool(getattr(self, "_show_excluded_branches", False)),
            "entries": [
                {
                    "branch_id": int(branch_id),
                    "event_count": int(count),
                    "color": str(color_map.get(int(branch_id), self._branch_hex_color(branch_id))),
                }
                for branch_id, count in ordered[:limit]
            ],
            "truncated": bool(len(ordered) > limit),
        }

    def set_excluded_branches_visible(self, visible: bool, *, render: bool = True) -> None:
        """Toggle the neutral, aggregate display of branches excluded by size."""
        self._show_excluded_branches = bool(visible)
        self._apply_branch_correlation_peak_emphasis(update_polydata=True)
        has_points = bool(self._peak_marker_display_positions.size)
        self._peak_marker_actor.SetVisibility(bool(self._peak_marker_visible and has_points))
        if render:
            self._request_render()

    def pick_peak_at_display(
        self, x: float, y: float, *, tolerance: float = 0.012
    ) -> dict[str, Any] | None:
        """Pick one rendered peak glyph and return compact branch metadata.

        Coordinates are VTK display coordinates (origin at the lower-left).
        The picker is restricted to the peak actor and never changes the active
        interactor style or consumes the mouse event.
        """
        if (
            self._peak_marker_actor is None
            or not self._peak_marker_actor.GetVisibility()
            or self._peak_marker_polydata.GetNumberOfPoints() == 0
        ):
            return None
        try:
            tolerance = float(tolerance)
            if not np.isfinite(tolerance):
                return None
            self._peak_point_picker.SetTolerance(max(0.002, min(0.05, tolerance)))
            hit = self._peak_point_picker.Pick(float(x), float(y), 0.0, self.renderer)
            if not hit:
                return None
            picked_actor = self._peak_point_picker.GetActor()
            if picked_actor is None or picked_actor != self._peak_marker_actor:
                return None
            point_id = int(self._peak_point_picker.GetPointId())
            if point_id < 0 or point_id >= self._peak_marker_rendered_indices.size:
                return None
            source_index = int(self._peak_marker_rendered_indices[point_id])
            position = self._peak_marker_positions_physical[source_index]
            peak_id = int(self._peak_marker_peak_ids[source_index])
            branch_id = int(self._peak_marker_branch_ids[source_index])
            analysis_included = bool(self._peak_marker_analysis_included[source_index])
            if branch_id >= 0:
                count_map = (
                    self._peak_branch_event_counts if analysis_included
                    else self._peak_excluded_branch_event_counts
                )
                event_count = int(count_map.get(branch_id, 0))
            else:
                event_count = 0
            # Reflect the actual color uploaded to this glyph. When colouring
            # by branch is disabled this is the user's uniform color, not the
            # branch palette; missing branch IDs use the marker fallback.
            rgba = self._peak_marker_colors[source_index]
            color = "#{:02X}{:02X}{:02X}".format(
                int(rgba[0]), int(rgba[1]), int(rgba[2])
            )
            return {
                "point_index": point_id,
                "peak_id": peak_id if peak_id >= 0 else None,
                "branch_id": branch_id if branch_id >= 0 else None,
                "branch_event_count": event_count,
                "analysis_included": analysis_included,
                "x": float(position[0]),
                "y": float(position[1]),
                "theta": float(position[2]),
                "color": color,
            }
        except Exception:
            # Hover is an optional inspection affordance. A picking failure
            # must never interrupt rendering or camera interaction.
            return None

    def set_branch_edges(
        self,
        peak_catalog: Any | None,
        branch_edges: Any | None,
        x_coords: Any,
        y_coords: Any,
        *,
        visible: bool = False,
        color_by_branch: bool = True,
        uniform_color: str = "#53C6D8",
        line_width: float = 2.0,
        opacity: float = 0.92,
        render: bool = True,
    ) -> int:
        """Build one VTK line per valid ``branch_edges`` peak-id pair."""
        try:
            line_width = float(line_width)
            opacity = float(opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Толщина и непрозрачность связей должны быть числами") from exc
        if not np.isfinite(line_width) or not (0.5 <= line_width <= 8.0):
            raise ValueError("Толщина связей должна находиться в диапазоне 0.5–8 px")
        if not np.isfinite(opacity) or not (0.05 <= opacity <= 1.0):
            raise ValueError("Непрозрачность связей должна находиться в диапазоне 0.05–1.0")
        self._branch_edge_width = line_width
        self._branch_edge_opacity = opacity
        prop = self._branch_edge_actor.GetProperty()
        prop.SetLineWidth(line_width)
        prop.SetOpacity(opacity)

        edge_columns = set(getattr(branch_edges, "columns", ())) if branch_edges is not None else set()
        catalog_columns = set(getattr(peak_catalog, "columns", ())) if peak_catalog is not None else set()
        if (
            branch_edges is None or getattr(branch_edges, "empty", False)
            or not {"peak_id_a", "peak_id_b"}.issubset(edge_columns)
            or "peak_id" not in catalog_columns
        ):
            positions = np.empty((0, 2, 3), dtype=np.float64)
            colors = np.empty((0, 4), dtype=np.uint8)
        else:
            peak_positions, valid_rows = self._peak_catalog_geometry(
                peak_catalog, x_coords, y_coords
            )
            ids = np.asarray(peak_catalog["peak_id"], dtype=object)[valid_rows]
            position_by_id = {}
            for peak_id, position in zip(ids, peak_positions):
                key = self._stable_id_key(peak_id)
                if key is not None:
                    position_by_id[key] = position
            branch_by_id = {}
            inclusion_by_id = {}
            if "branch_id" in catalog_columns:
                raw_branch_ids = self._numeric_column(peak_catalog["branch_id"])[valid_rows]
                if "analysis_included" in catalog_columns:
                    raw_flags = np.asarray(peak_catalog["analysis_included"], dtype=object)[valid_rows]
                    flags = np.asarray([
                        (bool(value) if isinstance(value, (bool, np.bool_))
                         else str(value).strip().lower() in {"true", "1", "yes", "y"})
                        for value in raw_flags
                    ], dtype=bool)
                else:
                    flags = np.ones(len(ids), dtype=bool)
                for peak_id, branch_id, included in zip(ids, raw_branch_ids, flags):
                    key = self._stable_id_key(peak_id)
                    if key is not None and np.isfinite(branch_id):
                        branch_by_id[key] = int(round(float(branch_id)))
                        inclusion_by_id[key] = bool(included)
            edge_a = np.asarray(branch_edges["peak_id_a"], dtype=object)
            edge_b = np.asarray(branch_edges["peak_id_b"], dtype=object)
            valid_pairs = []
            branch_values = []
            inclusion_values = []
            for left, right in zip(edge_a, edge_b):
                key_a, key_b = self._stable_id_key(left), self._stable_id_key(right)
                if key_a not in position_by_id or key_b not in position_by_id:
                    continue
                branch_a = branch_by_id.get(key_a, -1)
                branch_b = branch_by_id.get(key_b, branch_a)
                # branch_edges must connect events belonging to one connected
                # component. Refuse malformed cross-branch links rather than
                # colouring them ambiguously.
                if branch_a >= 0 and branch_b >= 0 and branch_a != branch_b:
                    continue
                valid_pairs.append((position_by_id[key_a], position_by_id[key_b]))
                branch_values.append(branch_a if branch_a >= 0 else branch_b)
                inclusion_values.append(
                    bool(inclusion_by_id.get(key_a, True) and inclusion_by_id.get(key_b, True))
                )
                if len(valid_pairs) >= 200_000:
                    break
            positions = (
                np.asarray(valid_pairs, dtype=np.float64).reshape(-1, 2, 3)
                if valid_pairs else np.empty((0, 2, 3), dtype=np.float64)
            )
            branch_numeric = np.asarray(branch_values, dtype=np.int64) if branch_values else np.empty(0, dtype=np.int64)
            edge_included = np.asarray(inclusion_values, dtype=bool)
            fallback = self._hex_rgba(uniform_color)
            colors = np.repeat(fallback[None, :], len(valid_pairs), axis=0)
            if color_by_branch and colors.size:
                branch_valid = branch_numeric >= 0
                if np.any(branch_valid):
                    colors[branch_valid] = self._branch_rgba(branch_numeric[branch_valid])
        # Actor property opacity is applied globally; preserve per-cell alpha
        # in the direct RGBA array as fully opaque to avoid accidental squaring.
        if colors.size:
            colors[:, 3] = 255
        self._branch_edge_positions_physical = np.ascontiguousarray(positions, dtype=np.float64)
        self._branch_edge_base_colors = np.ascontiguousarray(colors, dtype=np.uint8)
        self._branch_edge_branch_ids = np.ascontiguousarray(
            branch_numeric if "branch_numeric" in locals() else np.full(len(positions), -1, dtype=np.int64),
            dtype=np.int64,
        )
        self._branch_edge_analysis_included = np.ascontiguousarray(
            edge_included if "edge_included" in locals() else np.ones(len(positions), dtype=bool),
            dtype=bool,
        )
        self._branch_edge_colors = self._branch_edge_base_colors.copy()
        self._update_branch_edge_polydata()
        self._branch_edge_visible = bool(visible and positions.shape[0] > 0)
        self._branch_edge_actor.SetVisibility(self._branch_edge_visible)
        if render:
            self._request_render()
        return int(self._branch_edge_polydata.GetNumberOfLines())

    def _update_branch_edge_polydata(self) -> None:
        """Rebuild only the currently visible branch-edge subset."""
        vtk, numpy_to_vtk = _vtk()
        all_edges = self._branch_edge_positions_physical
        all_ids = np.asarray(getattr(self, "_branch_edge_branch_ids", np.full(len(all_edges), -1)), dtype=np.int64)
        all_included = np.asarray(
            getattr(self, "_branch_edge_analysis_included", np.ones(len(all_edges), dtype=bool)),
            dtype=bool,
        )
        if all_ids.size != len(all_edges):
            all_ids = np.full(len(all_edges), -1, dtype=np.int64)
        if all_included.size != len(all_edges):
            all_included = np.ones(len(all_edges), dtype=bool)
        focus = bool(getattr(self, "_branch_display_focus_selected", False))
        selected = np.asarray(getattr(self, "_branch_display_selected_ids", np.empty(0)), dtype=np.int64)
        visible_mask = (
            np.isin(all_ids, selected, assume_unique=False)
            if focus and selected.size else all_included
        )
        visible_indices = np.flatnonzero(visible_mask)
        edges = all_edges[visible_indices]
        base_colors = np.asarray(
            getattr(self, "_branch_edge_base_colors", self._branch_edge_colors), dtype=np.uint8
        )
        if base_colors.shape != (len(all_edges), 4):
            base_colors = np.repeat(self._hex_rgba("#53C6D8")[None, :], len(all_edges), axis=0)
        colors = base_colors[visible_indices].copy()
        color_map = getattr(self, "_branch_display_colors", {}) or {}
        if colors.size and color_map:
            edge_ids = all_ids[visible_indices]
            for branch_id, color in color_map.items():
                mask = edge_ids == int(branch_id)
                if not np.any(mask):
                    continue
                try:
                    rgba = self._hex_rgba(str(color))
                except (TypeError, ValueError):
                    continue
                colors[mask, :3] = rgba[:3]
        self._branch_edge_colors = np.ascontiguousarray(colors, dtype=np.uint8)
        count = int(edges.shape[0])
        if count == 0:
            self._branch_edge_polydata.Initialize()
            self._branch_edge_polydata.Modified()
            self._branch_edge_actor.SetVisibility(False)
            if hasattr(self, "_branch_correlation_matched_polydata"):
                self._update_branch_correspondence_polydata()
            return
        display = edges.copy()
        for column, axis, factor in zip(
            range(3), (self._physical_x, self._physical_y, self._physical_theta),
            self._axis_scales,
        ):
            center = 0.5 * (float(axis[0]) + float(axis[-1]))
            display[:, :, column] = center + (display[:, :, column] - center) * float(factor)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(display.reshape(-1, 3)), deep=True))
        self._branch_edge_polydata.SetPoints(vtk_points)
        cells = vtk.vtkCellArray()
        from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
        offsets = numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64) * 2, deep=True)
        connectivity = numpy_to_vtkIdTypeArray(np.arange(count * 2, dtype=np.int64), deep=True)
        cells.SetData(offsets, connectivity)
        self._branch_edge_polydata.SetLines(cells)
        rgba = numpy_to_vtk(np.ascontiguousarray(colors), deep=True)
        rgba.SetNumberOfComponents(4)
        rgba.SetName("BranchEdgeRGBA")
        self._branch_edge_polydata.GetCellData().SetScalars(rgba)
        self._branch_edge_polydata.Modified()
        self._branch_edge_actor.SetVisibility(bool(self._branch_edge_visible))
        if hasattr(self, "_branch_correlation_matched_polydata"):
            self._update_branch_correspondence_polydata()

    def set_branch_display_style(
        self,
        branch_colors: dict[int, str] | None = None,
        selected_branch_ids=(),
        *,
        focus_selected: bool = False,
        render: bool = True,
    ) -> None:
        """Apply one branch palette/selection to both 3D event and edge layers.

        This is display state only: inclusion flags, branch IDs and scientific
        artifact tables are not edited. When focus is enabled it temporarily
        reveals the selected branch overlays and restores their previous layer
        visibility when focus is disabled.
        """
        normalized: dict[int, str] = {}
        for raw_id, raw_color in (branch_colors or {}).items():
            try:
                branch_id = int(raw_id)
                color = str(raw_color).strip()
                if branch_id < 0 or len(color) != 7 or not color.startswith("#"):
                    continue
                int(color[1:], 16)
            except (TypeError, ValueError, OverflowError):
                continue
            normalized[branch_id] = color.upper()
        selected = sorted({int(value) for value in selected_branch_ids if int(value) >= 0})
        previous_focus = bool(getattr(self, "_branch_display_focus_selected", False))
        focus_selected = bool(focus_selected and selected)
        if focus_selected and not previous_focus:
            self._branch_display_previous_visibility = (
                bool(self._peak_marker_visible), bool(self._branch_edge_visible)
            )
            self._peak_marker_visible = True
            self._branch_edge_visible = True
        elif previous_focus and not focus_selected:
            previous = self._branch_display_previous_visibility
            if previous is not None:
                self._peak_marker_visible, self._branch_edge_visible = previous
            self._branch_display_previous_visibility = None
        self._branch_display_colors = normalized
        self._branch_display_selected_ids = np.asarray(selected, dtype=np.int64)
        self._branch_display_focus_selected = focus_selected
        self._apply_branch_correlation_peak_emphasis(update_polydata=False)
        self._update_peak_marker_polydata()
        self._peak_marker_actor.SetVisibility(
            bool(self._peak_marker_visible and self._peak_marker_rendered_indices.size)
        )
        self._update_branch_edge_polydata()
        self._branch_edge_actor.SetVisibility(
            bool(self._branch_edge_visible and self._branch_edge_polydata.GetNumberOfLines())
        )
        if render:
            self._request_render()

    def set_branch_edges_visible(self, visible: bool, *, render: bool = True) -> None:
        has_visible_lines = bool(self._branch_edge_polydata.GetNumberOfLines() > 0)
        self._branch_edge_visible = bool(visible and has_visible_lines)
        self._branch_edge_actor.SetVisibility(self._branch_edge_visible)
        if render:
            self._request_render()

    def set_branch_edge_settings(
        self, *, line_width: float | None = None, opacity: float | None = None,
        render: bool = True,
    ) -> None:
        if line_width is not None:
            value = float(line_width)
            if not np.isfinite(value) or not (0.5 <= value <= 8.0):
                raise ValueError("Толщина связей должна находиться в диапазоне 0.5–8 px")
            self._branch_edge_width = value
            self._branch_edge_actor.GetProperty().SetLineWidth(value)
        if opacity is not None:
            value = float(opacity)
            if not np.isfinite(value) or not (0.05 <= value <= 1.0):
                raise ValueError("Непрозрачность связей должна находиться в диапазоне 0.05–1.0")
            self._branch_edge_opacity = value
            self._branch_edge_actor.GetProperty().SetOpacity(value)
        if render:
            self._request_render()

    @staticmethod
    def _normalise_line_segments(value: Any, *, name: str) -> np.ndarray:
        """Validate physical line geometry as finite N×2×3 coordinates."""
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.empty((0, 2, 3), dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (2, 3):
            raise ValueError(f"{name} должны иметь форму N×2×3, получено {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} содержат не конечные координаты")
        return np.ascontiguousarray(array, dtype=np.float64)

    def _correspondence_roi_mask(self, segments: np.ndarray) -> np.ndarray:
        """Return one inclusive physical-bounds mask per line segment."""
        if segments.size == 0:
            return np.zeros(0, dtype=bool)
        keep = np.ones(segments.shape[0], dtype=bool)
        for column, axis in enumerate((self._physical_x, self._physical_y, self._physical_theta)):
            if axis is None:
                continue
            low, high = sorted((float(axis[0]), float(axis[-1])))
            values = segments[:, :, column]
            keep &= np.all((values >= low) & (values <= high), axis=1)
        return keep

    def _correspondence_roi_filter(self, segments: np.ndarray) -> np.ndarray:
        """Clip link list to the physical bounds of the currently loaded ROI."""
        if segments.size == 0:
            return segments
        return np.ascontiguousarray(segments[self._correspondence_roi_mask(segments)], dtype=np.float64)

    def _line_polydata_from_segments(
        self, polydata: Any, segments: np.ndarray, colors: np.ndarray | None = None,
        *, dashed: bool = False,
    ) -> None:
        """Replace one line actor's geometry; all input coordinates are physical."""
        vtk, numpy_to_vtk = _vtk()
        if segments.size == 0:
            polydata.Initialize()
            polydata.Modified()
            return
        keep = self._correspondence_roi_mask(segments)
        source = np.ascontiguousarray(segments[keep], dtype=np.float64)
        if colors is not None:
            colors = np.ascontiguousarray(np.asarray(colors, dtype=np.uint8)[keep])
        if source.size == 0:
            polydata.Initialize()
            polydata.Modified()
            return
        if dashed:
            fractions = ((0.00, 0.22), (0.37, 0.59), (0.74, 0.96))
            delta = source[:, 1, :] - source[:, 0, :]
            starts = np.concatenate([source[:, 0, :] + delta * start for start, _end in fractions])
            ends = np.concatenate([source[:, 0, :] + delta * end for _start, end in fractions])
            source = np.stack((starts, ends), axis=1)
        display = source.copy()
        for column, axis, factor in zip(
            range(3), (self._physical_x, self._physical_y, self._physical_theta),
            self._axis_scales,
        ):
            if axis is None:
                continue
            center = 0.5 * (float(axis[0]) + float(axis[-1]))
            display[:, :, column] = center + (display[:, :, column] - center) * float(factor)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(display.reshape(-1, 3)), deep=True))
        polydata.SetPoints(vtk_points)
        cells = vtk.vtkCellArray()
        from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
        count = int(source.shape[0])
        offsets = numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64) * 2, deep=True)
        connectivity = numpy_to_vtkIdTypeArray(np.arange(count * 2, dtype=np.int64), deep=True)
        cells.SetData(offsets, connectivity)
        polydata.SetLines(cells)
        if colors is not None and not dashed:
            if colors.shape != (count, 4):
                raise ValueError(
                    f"RGBA-цвета связей должны иметь форму {(count, 4)}, "
                    f"получено {colors.shape}"
                )
            rgba = numpy_to_vtk(colors, deep=True)
            rgba.SetNumberOfComponents(4)
            rgba.SetName("BranchCorrelationRGBA")
            polydata.GetCellData().SetScalars(rgba)
        else:
            polydata.GetCellData().SetScalars(None)
        polydata.Modified()

    def _apply_branch_correlation_peak_emphasis(self, *, update_polydata: bool = True) -> None:
        """Fade non-core peaks only in the selected source/partner branches."""
        base = np.asarray(
            getattr(self, "_peak_marker_base_colors", np.empty((0, 4))), dtype=np.uint8
        )
        if base.ndim != 2 or base.shape[1:] != (4,):
            base = np.empty((0, 4), dtype=np.uint8)
        colors = base.copy()
        enabled = bool(
            getattr(self, "_branch_correlation_visible", False)
            and getattr(self, "_branch_correlation_emphasize_core", True)
        )
        branch_ids = np.asarray(getattr(self, "_peak_marker_branch_ids", np.empty(0)), dtype=np.int64)
        peak_ids = np.asarray(getattr(self, "_peak_marker_peak_ids", np.empty(0)), dtype=np.int64)
        focus = np.asarray(getattr(self, "_branch_correlation_focus_branch_ids", np.empty(0)), dtype=np.int64)
        matched = np.asarray(getattr(self, "_branch_correlation_matched_peak_ids", np.empty(0)), dtype=np.int64)
        if enabled and colors.shape[0] and branch_ids.size == colors.shape[0] and peak_ids.size == colors.shape[0] and focus.size:
            in_focus = np.isin(branch_ids, focus, assume_unique=False)
            is_core = np.isin(peak_ids, matched, assume_unique=False) if matched.size else np.zeros(peak_ids.size, dtype=bool)
            residual = in_focus & ~is_core
            if np.any(residual):
                factor = float(np.clip(getattr(self, "_branch_correlation_residual_opacity", 0.28), 0.05, 1.0))
                colors[residual, 3] = np.rint(colors[residual, 3].astype(np.float64) * factor).astype(np.uint8)
        included = np.asarray(
            getattr(self, "_peak_marker_analysis_included", np.ones(colors.shape[0], dtype=bool)),
            dtype=bool,
        )
        if included.size == colors.shape[0] and np.any(~included):
            colors[~included] = np.asarray([132, 139, 149, 255], dtype=np.uint8)
        # User-defined display colours are applied after branch-correlation
        # emphasis. Only RGB changes; alpha still communicates the emphasis.
        color_map = getattr(self, "_branch_display_colors", {}) or {}
        focus = bool(getattr(self, "_branch_display_focus_selected", False))
        selected_ids = np.asarray(getattr(self, "_branch_display_selected_ids", np.empty(0)), dtype=np.int64)
        if colors.size and branch_ids.size == colors.shape[0] and color_map:
            for branch_id, color in color_map.items():
                mask = branch_ids == int(branch_id)
                if included.size == colors.shape[0]:
                    if focus and selected_ids.size and int(branch_id) in selected_ids:
                        pass
                    else:
                        mask &= included
                if not np.any(mask):
                    continue
                try:
                    rgba = self._hex_rgba(str(color))
                except (TypeError, ValueError):
                    continue
                colors[mask, :3] = rgba[:3]
        self._peak_marker_colors = np.ascontiguousarray(colors, dtype=np.uint8)
        if update_polydata and getattr(self, "_peak_marker_positions_physical", np.empty((0, 3))).size:
            self._update_peak_marker_polydata()

    def set_branch_correspondences(
        self,
        matched_segments: Any,
        partner_branch_ids: Any,
        candidate_segments: Any | None = None,
        matched_peak_ids: Any | None = None,
        focus_branch_ids: Any | None = None,
        *,
        visible: bool = True,
        show_candidates: bool = False,
        emphasize_core: bool = True,
        residual_opacity: float = 0.28,
        line_width: float = 2.0,
        opacity: float = 0.82,
        candidate_opacity: float = 0.82,
        render: bool = True,
    ) -> dict[str, int]:
        """Install accepted/alternative 3D links from a stored correlation result.

        Solid links are accepted directed peak-ID assignments. Dashed links are
        only supported alternatives and are optional. All positions are in the
        original, unregistered physical scene: the pair summary records the
        translation applied to the partner to align it to the source.
        """
        matched = self._normalise_line_segments(matched_segments, name="matched_segments")
        candidate = self._normalise_line_segments(
            candidate_segments if candidate_segments is not None else np.empty((0, 2, 3)),
            name="candidate_segments",
        )
        partner_ids = np.asarray(partner_branch_ids, dtype=np.int64).reshape(-1)
        if partner_ids.size != matched.shape[0]:
            raise ValueError("partner_branch_ids должен содержать один ID на каждую принятую связь")
        if np.any(partner_ids < 0):
            raise ValueError("partner_branch_ids не должны содержать отрицательные ID")
        try:
            width = float(line_width)
            layer_opacity = float(opacity)
            alt_opacity = float(candidate_opacity)
            residual_alpha = float(residual_opacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Параметры корреляционных связей должны быть числами") from exc
        if not np.isfinite(width) or not 0.5 <= width <= 8.0:
            raise ValueError("Толщина корреляционных связей должна быть 0.5–8 px")
        if not np.isfinite(layer_opacity) or not 0.05 <= layer_opacity <= 1.0:
            raise ValueError("Непрозрачность корреляционных связей должна быть 0.05–1.0")
        if not np.isfinite(alt_opacity) or not 0.05 <= alt_opacity <= 1.0:
            raise ValueError("Непрозрачность альтернативных связей должна быть 0.05–1.0")
        if not np.isfinite(residual_alpha) or not 0.05 <= residual_alpha <= 1.0:
            raise ValueError("Непрозрачность остатка должна быть 0.05–1.0")

        # The upstream preview builder imposes deterministic geometry caps;
        # keep a final guard here because this scene API is also reusable by
        # direct callers and must never accept unbounded GPU geometry.
        if matched.shape[0] > 100_000:
            select = np.linspace(0, matched.shape[0] - 1, 100_000, dtype=np.int64)
            matched = matched[select]
            partner_ids = partner_ids[select]
        if candidate.shape[0] > 20_000:
            select = np.linspace(0, candidate.shape[0] - 1, 20_000, dtype=np.int64)
            candidate = candidate[select]
        matched_keep = self._correspondence_roi_mask(matched)
        matched = np.ascontiguousarray(matched[matched_keep], dtype=np.float64)
        partner_ids = partner_ids[matched_keep]
        candidate_keep = self._correspondence_roi_mask(candidate)
        candidate = np.ascontiguousarray(candidate[candidate_keep], dtype=np.float64)

        ids = np.asarray(matched_peak_ids if matched_peak_ids is not None else (), dtype=np.int64).reshape(-1)
        focus = np.asarray(focus_branch_ids if focus_branch_ids is not None else (), dtype=np.int64).reshape(-1)
        self._branch_correlation_matched_segments_physical = np.ascontiguousarray(matched, dtype=np.float64)
        self._branch_correlation_matched_colors = (
            self._branch_rgba(partner_ids) if partner_ids.size else np.empty((0, 4), dtype=np.uint8)
        )
        self._branch_correlation_candidate_segments_physical = np.ascontiguousarray(candidate, dtype=np.float64)
        self._branch_correlation_matched_peak_ids = np.unique(ids[ids >= 0])
        self._branch_correlation_focus_branch_ids = np.unique(focus[focus >= 0])
        self._branch_correlation_show_candidates = bool(show_candidates)
        self._branch_correlation_emphasize_core = bool(emphasize_core)
        self._branch_correlation_residual_opacity = residual_alpha
        self._branch_correlation_line_width = width
        self._branch_correlation_opacity = layer_opacity
        self._branch_correlation_candidate_opacity = alt_opacity
        self._branch_correlation_matched_actor.GetProperty().SetLineWidth(width)
        self._branch_correlation_matched_actor.GetProperty().SetOpacity(layer_opacity)
        self._branch_correlation_candidate_actor.GetProperty().SetLineWidth(max(0.5, width * 0.75))
        self._branch_correlation_candidate_actor.GetProperty().SetOpacity(alt_opacity)
        self._update_branch_correspondence_polydata()
        self.set_branch_correspondences_visible(visible, render=False)
        if render:
            self._request_render()
        return {
            "matched_links": int(self._branch_correlation_matched_segments_physical.shape[0]),
            "candidate_links": int(self._branch_correlation_candidate_segments_physical.shape[0]),
            "matched_peaks": int(self._branch_correlation_matched_peak_ids.size),
            "focus_branches": int(self._branch_correlation_focus_branch_ids.size),
        }

    def _update_branch_correspondence_polydata(self) -> None:
        """Regenerate physical-to-display line geometry after ROI/axis changes."""
        self._line_polydata_from_segments(
            self._branch_correlation_matched_polydata,
            self._branch_correlation_matched_segments_physical,
            self._branch_correlation_matched_colors,
        )
        self._line_polydata_from_segments(
            self._branch_correlation_candidate_polydata,
            self._branch_correlation_candidate_segments_physical,
            dashed=True,
        )

    def set_branch_correspondences_visible(
        self, visible: bool, *, show_candidates: bool | None = None,
        render: bool = True,
    ) -> None:
        self._branch_correlation_visible = bool(visible)
        if show_candidates is not None:
            self._branch_correlation_show_candidates = bool(show_candidates)
        has_matched = bool(self._branch_correlation_matched_segments_physical.size)
        has_candidates = bool(self._branch_correlation_candidate_segments_physical.size)
        self._branch_correlation_matched_actor.SetVisibility(
            bool(self._branch_correlation_visible and has_matched)
        )
        self._branch_correlation_candidate_actor.SetVisibility(
            bool(self._branch_correlation_visible and self._branch_correlation_show_candidates and has_candidates)
        )
        self._apply_branch_correlation_peak_emphasis(update_polydata=True)
        if render:
            self._request_render()

    def set_branch_correspondence_settings(
        self, *, show_candidates: bool | None = None,
        emphasize_core: bool | None = None,
        residual_opacity: float | None = None,
        line_width: float | None = None,
        opacity: float | None = None,
        render: bool = True,
    ) -> None:
        if show_candidates is not None:
            self._branch_correlation_show_candidates = bool(show_candidates)
        if emphasize_core is not None:
            self._branch_correlation_emphasize_core = bool(emphasize_core)
        if residual_opacity is not None:
            value = float(residual_opacity)
            if not np.isfinite(value) or not 0.05 <= value <= 1.0:
                raise ValueError("Непрозрачность остатка должна быть 0.05–1.0")
            self._branch_correlation_residual_opacity = value
        if line_width is not None:
            value = float(line_width)
            if not np.isfinite(value) or not 0.5 <= value <= 8.0:
                raise ValueError("Толщина корреляционных связей должна быть 0.5–8 px")
            self._branch_correlation_line_width = value
            self._branch_correlation_matched_actor.GetProperty().SetLineWidth(value)
            self._branch_correlation_candidate_actor.GetProperty().SetLineWidth(max(0.5, value * 0.75))
        if opacity is not None:
            value = float(opacity)
            if not np.isfinite(value) or not 0.05 <= value <= 1.0:
                raise ValueError("Непрозрачность корреляционных связей должна быть 0.05–1.0")
            self._branch_correlation_opacity = value
            self._branch_correlation_matched_actor.GetProperty().SetOpacity(value)
        self.set_branch_correspondences_visible(
            self._branch_correlation_visible,
            show_candidates=self._branch_correlation_show_candidates,
            render=False,
        )
        if render:
            self._request_render()

    def set_volume_visible(self, visible: bool, *, render: bool = True) -> None:
        """Show/hide the volume without affecting independent point/edge layers."""
        self._volume_visible = bool(visible)
        if self.volume is not None:
            self.volume.SetVisibility(1 if self._volume_visible else 0)
        if render:
            self._request_render()

    @property
    def volume_visible(self) -> bool:
        return bool(self._volume_visible)

    def _on_peak_marker_view_changed(self, _caller=None, _event=None) -> None:
        """Refresh per-point world scales after camera or viewport changes."""
        if getattr(self, "_closed", False):
            return
        if getattr(self, "_peak_marker_display_positions", np.empty((0, 3))).size:
            self._update_peak_marker_glyph_scale()

    def _update_peak_marker_glyph_scale(self) -> float:
        """Convert the requested screen diameter to per-point world radii.

        A shared world-space sphere size is not a screen-space ``px`` size: it
        becomes sub-pixel when a large detector is fitted into a small window,
        and changes apparent size on zoom. For parallel projection the
        conversion is constant. For perspective projection it is calculated
        independently at each event depth, so points at different theta/depth
        remain equally legible. The X/Y pitch is retained as a safe fallback
        for headless tests or a not-yet-sized render window.
        """
        spatial_steps: list[float] = []
        for axis, factor in zip(
            (self._physical_x, self._physical_y), self._axis_scales[:2],
        ):
            if axis is None:
                continue
            values = np.asarray(axis, dtype=np.float64).reshape(-1)
            if values.size < 2:
                continue
            steps = np.abs(np.diff(values)) * abs(float(factor))
            steps = steps[np.isfinite(steps) & (steps > 0.0)]
            if steps.size:
                spatial_steps.append(float(np.median(steps)))

        if spatial_steps:
            reference_pitch = min(spatial_steps)
        else:
            spatial_extents = [
                float(np.ptp(np.asarray(axis, dtype=np.float64))) * abs(float(factor))
                for axis, factor in zip(
                    (self._physical_x, self._physical_y), self._axis_scales[:2],
                )
                if axis is not None and np.asarray(axis).size > 1
            ]
            if spatial_extents:
                reference_pitch = max(max(spatial_extents) / 50.0, 1e-6)
            else:
                theta = self._physical_theta
                theta_values = np.asarray(theta, dtype=np.float64).reshape(-1) if theta is not None else np.empty(0)
                if theta_values.size > 1:
                    theta_steps = np.abs(np.diff(theta_values)) * abs(float(self._axis_scales[2]))
                    theta_steps = theta_steps[np.isfinite(theta_steps) & (theta_steps > 0.0)]
                    reference_pitch = float(np.median(theta_steps)) if theta_steps.size else 1.0
                else:
                    reference_pitch = 1.0
        self._peak_marker_reference_pitch = float(reference_pitch)

        display_positions = np.asarray(
            getattr(self, "_peak_marker_display_positions", np.empty((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)
        if display_positions.shape[0] == 0:
            self._peak_marker_scale_values = np.empty(0, dtype=np.float64)
            self._peak_marker_world_radius = max(reference_pitch * 0.07, 1e-9)
            self._peak_marker_world_radius_range = (
                self._peak_marker_world_radius, self._peak_marker_world_radius,
            )
            self._peak_marker_mapper.SetScaleFactor(float(self._peak_marker_world_radius))
            return float(self._peak_marker_world_radius)

        # Fallback is used only before a valid viewport/camera exists. In a
        # live QVTK view, radius_px is converted from the actual render size.
        fallback_radius = max(
            reference_pitch * min(max(0.07 * float(self._peak_marker_size), 0.07), 0.45),
            1e-9,
        )
        world_radii = np.full(display_positions.shape[0], fallback_radius, dtype=np.float64)
        try:
            window_width, window_height = (
                int(value) for value in self.render_window.GetSize()
            )
            viewport = tuple(float(value) for value in self.renderer.GetViewport())
            viewport_width = window_width * max(viewport[2] - viewport[0], 0.0)
            viewport_height = window_height * max(viewport[3] - viewport[1], 0.0)
            camera = self.renderer.GetActiveCamera()
            if (
                window_width > 1 and window_height > 1
                and viewport_width > 1.0 and viewport_height > 1.0
                and camera is not None
            ):
                # Apply the actor's shared display transform before measuring
                # depth; camera distance is defined in rendered world space.
                actor_matrix = self._peak_marker_actor.GetMatrix()
                matrix = np.asarray([
                    [actor_matrix.GetElement(row, col) for col in range(4)]
                    for row in range(4)
                ], dtype=np.float64)
                homogeneous = np.column_stack((display_positions, np.ones(display_positions.shape[0])))
                world_h = homogeneous @ matrix.T
                denominator = world_h[:, 3]
                valid_w = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
                world_positions = np.full((display_positions.shape[0], 3), np.nan, dtype=np.float64)
                world_positions[valid_w] = world_h[valid_w, :3] / denominator[valid_w, None]

                diameter_radius_px = max(float(self._peak_marker_size) * 0.5, 0.5)
                if bool(camera.GetParallelProjection()):
                    parallel_scale = max(float(camera.GetParallelScale()), 1e-12)
                    radius = diameter_radius_px * (2.0 * parallel_scale) / viewport_height
                    world_radii.fill(max(radius, 1e-12))
                else:
                    view_angle = float(camera.GetViewAngle())
                    tangent_half = float(np.tan(np.deg2rad(view_angle) * 0.5))
                    if bool(camera.GetUseHorizontalViewAngle()):
                        tangent_half *= viewport_height / viewport_width
                    camera_position = np.asarray(camera.GetPosition(), dtype=np.float64)
                    direction = np.asarray(camera.GetDirectionOfProjection(), dtype=np.float64)
                    direction_norm = float(np.linalg.norm(direction))
                    if direction_norm > 1e-12 and np.isfinite(tangent_half) and tangent_half > 0.0:
                        direction /= direction_norm
                        depth = (world_positions - camera_position[None, :]) @ direction
                        valid_depth = np.isfinite(depth) & (depth > 1e-9)
                        positive_depth = depth[valid_depth]
                        fallback_depth = (
                            float(np.median(positive_depth))
                            if positive_depth.size else max(float(camera.GetDistance()), 1.0)
                        )
                        safe_depth = np.where(valid_depth, depth, fallback_depth)
                        world_radii = (
                            diameter_radius_px * (2.0 * safe_depth * tangent_half)
                            / viewport_height
                        )
                world_radii = np.where(
                    np.isfinite(world_radii) & (world_radii > 1e-12),
                    world_radii, fallback_radius,
                )
        except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
            # A render-window may not have a realised size during construction;
            # the next camera/resize event will recalculate the exact pixel scale.
            pass

        median_radius = max(float(np.median(world_radii)), 1e-12)
        relative_scales = np.asarray(world_radii / median_radius, dtype=np.float64)
        relative_scales[~np.isfinite(relative_scales) | (relative_scales <= 0.0)] = 1.0
        self._peak_marker_scale_values = relative_scales
        self._peak_marker_world_radius = median_radius
        self._peak_marker_world_radius_range = (
            float(np.min(world_radii)), float(np.max(world_radii)),
        )

        vtk, numpy_to_vtk = _vtk()
        scale_array = numpy_to_vtk(np.ascontiguousarray(relative_scales), deep=True)
        scale_array.SetName("PeakMarkerGlyphScale")
        point_data = self._peak_marker_polydata.GetPointData()
        point_data.RemoveArray("PeakMarkerGlyphScale")
        point_data.AddArray(scale_array)
        point_data.Modified()
        self._peak_marker_mapper.SetScaleModeToScaleByMagnitude()
        self._peak_marker_mapper.SetScaleArray("PeakMarkerGlyphScale")
        self._peak_marker_mapper.SetScaleFactor(float(median_radius))
        self._peak_marker_mapper.Modified()
        self._peak_marker_polydata.Modified()
        return float(median_radius)

    def _update_peak_marker_polydata(self) -> None:
        """Rebuild VTK glyph data from the currently visible event subset."""
        vtk, numpy_to_vtk = _vtk()
        physical = self._peak_marker_positions_physical
        included = np.asarray(
            getattr(self, "_peak_marker_analysis_included", np.ones(len(physical), dtype=bool)),
            dtype=bool,
        )
        if included.size != len(physical):
            included = np.ones(len(physical), dtype=bool)
        show_excluded = bool(getattr(self, "_show_excluded_branches", False))
        focus = bool(getattr(self, "_branch_display_focus_selected", False))
        selected = np.asarray(getattr(self, "_branch_display_selected_ids", np.empty(0)), dtype=np.int64)
        branch_ids = np.asarray(getattr(self, "_peak_marker_branch_ids", np.full(len(physical), -1)), dtype=np.int64)
        if focus and selected.size and branch_ids.size == len(physical):
            # Explicit branch-manager focus may inspect an otherwise excluded
            # branch, but does not change its scientific inclusion flag.
            visible_mask = np.isin(branch_ids, selected, assume_unique=False)
        else:
            visible_mask = included | (show_excluded & ~included)
        visible_indices = np.flatnonzero(visible_mask).astype(np.int64, copy=False)
        self._peak_marker_rendered_indices = np.ascontiguousarray(visible_indices, dtype=np.int64)
        if visible_indices.size == 0:
            self._peak_marker_display_positions = np.empty((0, 3), dtype=np.float64)
            self._peak_marker_polydata.Initialize()
            self._peak_marker_polydata.Modified()
            self._update_peak_marker_glyph_scale()
            return

        display = physical[visible_indices].copy()
        for column, axis, factor in zip(
            range(3), (self._physical_x, self._physical_y, self._physical_theta),
            self._axis_scales,
        ):
            center = 0.5 * (float(axis[0]) + float(axis[-1]))
            display[:, column] = center + (display[:, column] - center) * float(factor)
        self._peak_marker_display_positions = np.ascontiguousarray(display, dtype=np.float64)

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(self._peak_marker_display_positions, deep=True))
        self._peak_marker_polydata.SetPoints(vtk_points)
        cells = vtk.vtkCellArray()
        from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
        if hasattr(cells, "SetData"):
            offsets = numpy_to_vtkIdTypeArray(
                np.arange(len(visible_indices) + 1, dtype=np.int64), deep=True
            )
            connectivity = numpy_to_vtkIdTypeArray(
                np.arange(len(visible_indices), dtype=np.int64), deep=True
            )
            cells.SetData(offsets, connectivity)
        else:  # pragma: no cover - compatibility with older VTK releases
            packed = np.empty((len(visible_indices), 2), dtype=np.int64)
            packed[:, 0] = 1
            packed[:, 1] = np.arange(len(visible_indices), dtype=np.int64)
            cells.SetCells(len(visible_indices), numpy_to_vtkIdTypeArray(packed.ravel(), deep=True))
        self._peak_marker_polydata.SetVerts(cells)
        rgba = numpy_to_vtk(
            np.ascontiguousarray(self._peak_marker_colors[visible_indices]), deep=True
        )
        rgba.SetNumberOfComponents(4)
        rgba.SetName("PeakMarkerRGBA")
        self._peak_marker_polydata.GetPointData().SetScalars(rgba)
        self._peak_marker_polydata.Modified()
        self._update_peak_marker_glyph_scale()

    def set_peak_markers_visible(self, visible: bool, *, render: bool = True) -> None:
        self._peak_marker_visible = bool(visible)
        self._peak_marker_actor.SetVisibility(
            bool(self._peak_marker_visible and self._peak_marker_rendered_indices.size)
        )
        if render:
            self._request_render()

    def set_peak_marker_settings(
        self, *, point_size: float | None = None, opacity: float | None = None,
        render: bool = True,
    ) -> None:
        if point_size is not None:
            value = float(point_size)
            if not np.isfinite(value) or not (1.0 <= value <= 20.0):
                raise ValueError("Диаметр маркеров должен находиться в диапазоне 1–20 px")
            self._peak_marker_size = value
            self._update_peak_marker_glyph_scale()
        if opacity is not None:
            value = float(opacity)
            if not np.isfinite(value) or not (0.05 <= value <= 1.0):
                raise ValueError("Непрозрачность маркеров должна находиться в диапазоне 0.05–1.0")
            self._peak_marker_opacity = value
            self._peak_marker_actor.GetProperty().SetOpacity(value)
        if render:
            self._request_render()

    def set_axis_scales(
        self,
        x_scale: float,
        y_scale: float,
        theta_scale: float,
        *,
        render: bool = True,
    ) -> None:
        """Apply X/Y/θ scale by replacing display-grid coordinates atomically.

        Physical payload coordinates and intensities are immutable. The 3D
        grid, coplanar XY projection, selected-theta shader, optical cursor and
        frame/ticks/grid all share the same displayed coordinate system.
        Every target axis is derived from its original physical array, so
        repeated edits never accumulate floating-point scaling error.
        """
        try:
            values = tuple(float(value) for value in (x_scale, y_scale, theta_scale))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Ожидаются три конечных коэффициента масштаба X/Y/θ.") from exc
        if len(values) != 3 or any(
            not np.isfinite(value) or not (0.1 <= value <= 10.0) for value in values
        ):
            raise ValueError("Коэффициенты масштабирования X/Y/θ должны лежать в [0.1, 10].")

        # Before a volume is loaded, remember only a validated preference.
        if self.payload is None or self.grid is None:
            self._axis_scales = values
            self._theta_scale = values[2]
            return

        source_axes = (self._physical_x, self._physical_y, self._physical_theta)
        if any(axis is None for axis in source_axes):
            raise RuntimeError("Исходные координаты объёма ещё не подготовлены.")

        # Prepare every numpy and VTK coordinate array before mutating either grid.
        display_x, display_y, display_theta = tuple(
            self._scaled_axis_coordinates(axis, factor)
            for axis, factor in zip(source_axes, values)
        )
        vtk, numpy_to_vtk = _vtk()
        grid_arrays = tuple(
            self._vtk_coordinate_array(numpy_to_vtk, axis)
            for axis in (display_x, display_y, display_theta)
        )

        projection_arrays = None
        if self.projection_grid is not None:
            theta_center = 0.5 * (
                float(self._physical_theta[0]) + float(self._physical_theta[-1])
            )
            projection_base_z = (
                float(self._projection_base_z)
                if self._projection_base_z is not None
                else float(self._physical_theta[0])
            )
            projection_display_z = theta_center + (projection_base_z - theta_center) * values[2]
            projection_arrays = (
                self._vtk_coordinate_array(numpy_to_vtk, display_x),
                self._vtk_coordinate_array(numpy_to_vtk, display_y),
                self._vtk_coordinate_array(numpy_to_vtk, np.asarray([projection_display_z], dtype=np.float64)),
            )

        # Snapshot VTK arrays and all authoritative display state for rollback.
        grid_old = (
            self.grid.GetXCoordinates(),
            self.grid.GetYCoordinates(),
            self.grid.GetZCoordinates(),
        )
        projection_old = None
        if self.projection_grid is not None:
            projection_old = (
                self.projection_grid.GetXCoordinates(),
                self.projection_grid.GetYCoordinates(),
                self.projection_grid.GetZCoordinates(),
            )
        previous_scales = self._axis_scales
        previous_theta_scale = self._theta_scale
        previous_display = (self._display_x, self._display_y, self._display_theta)
        previous_min_spacing = self._min_spacing
        previous_opacity_distance = (
            float(self.volume_property.GetScalarOpacityUnitDistance())
            if self.volume_property is not None else None
        )
        previous_sample_distance = (
            float(self.mapper.GetSampleDistance()) if self.mapper is not None else None
        )
        previous_transfer_info = dict(self._last_transfer_info)
        previous_crop_indices = self._theta_display_range_indices
        previous_crop_enabled = bool(self.mapper.GetCropping()) if self.mapper is not None else False
        previous_crop_planes = (
            tuple(float(v) for v in self.mapper.GetCroppingRegionPlanes())
            if self.mapper is not None else None
        )
        previous_crop_flags = (
            int(self.mapper.GetCroppingRegionFlags()) if self.mapper is not None else None
        )

        try:
            self.grid.SetXCoordinates(grid_arrays[0])
            self.grid.SetYCoordinates(grid_arrays[1])
            self.grid.SetZCoordinates(grid_arrays[2])
            self.grid.Modified()

            if self.projection_grid is not None and projection_arrays is not None:
                self.projection_grid.SetXCoordinates(projection_arrays[0])
                self.projection_grid.SetYCoordinates(projection_arrays[1])
                self.projection_grid.SetZCoordinates(projection_arrays[2])
                self.projection_grid.Modified()

            self._axis_scales = values
            self._theta_scale = values[2]  # legacy diagnostics/API compatibility
            self._display_x = display_x
            self._display_y = display_y
            self._display_theta = display_theta
            self._min_spacing = self._estimate_min_spacing(
                self.payload,
                theta_scale=values[2],
                x_scale=values[0],
                y_scale=values[1],
            )

            sample_distance = max(
                self._min_spacing * float(self.settings.sample_distance_factor), 1e-6
            )
            opacity_unit_distance = compute_opacity_unit_distance(
                self._min_spacing, self.settings.opacity_unit_distance_factor
            )
            if self.volume_property is not None:
                self.volume_property.SetScalarOpacityUnitDistance(float(opacity_unit_distance))
            if self.mapper is not None:
                self.mapper.SetSampleDistance(float(sample_distance))
                self.mapper.Modified()
            if self._last_transfer_info:
                self._last_transfer_info["sample_distance"] = float(sample_distance)
                self._last_transfer_info["opacity_unit_distance"] = float(opacity_unit_distance)
            if self.projection_mapper is not None:
                self.projection_mapper.Modified()
            if self.outline_filter is not None:
                self.outline_filter.SetInputData(self.grid)
                self.outline_filter.Modified()
                self.outline_filter.Update()

            self._update_theta_slice_shader_uniforms()
            self._update_theta_slice_marker_geometry()
            self._update_peak_marker_polydata()
            self._update_branch_edge_polydata()
            self._update_cursor_line_shader_uniforms()
            self._apply_theta_display_crop()
            self._update_theta_boundary_planes()
            self._axis_corner = None
            self._update_geometry_axes(force=True)
            self.renderer.ResetCameraClippingRange()
        except Exception:
            # Restore both rectilinear datasets and renderer state together.
            try:
                self.grid.SetXCoordinates(grid_old[0])
                self.grid.SetYCoordinates(grid_old[1])
                self.grid.SetZCoordinates(grid_old[2])
                self.grid.Modified()
                if self.projection_grid is not None and projection_old is not None:
                    self.projection_grid.SetXCoordinates(projection_old[0])
                    self.projection_grid.SetYCoordinates(projection_old[1])
                    self.projection_grid.SetZCoordinates(projection_old[2])
                    self.projection_grid.Modified()
                self._axis_scales = previous_scales
                self._theta_scale = previous_theta_scale
                self._display_x, self._display_y, self._display_theta = previous_display
                self._min_spacing = previous_min_spacing
                if self.volume_property is not None and previous_opacity_distance is not None:
                    self.volume_property.SetScalarOpacityUnitDistance(previous_opacity_distance)
                if self.mapper is not None:
                    if previous_sample_distance is not None:
                        self.mapper.SetSampleDistance(previous_sample_distance)
                    self.mapper.Modified()
                self._last_transfer_info = previous_transfer_info
                self._theta_display_range_indices = previous_crop_indices
                if self.mapper is not None and previous_crop_planes is not None:
                    self.mapper.SetCroppingRegionPlanes(*previous_crop_planes)
                    if previous_crop_flags is not None:
                        self.mapper.SetCroppingRegionFlags(previous_crop_flags)
                    self.mapper.SetCropping(1 if previous_crop_enabled else 0)
                if self.projection_mapper is not None:
                    self.projection_mapper.Modified()
                if self.outline_filter is not None:
                    self.outline_filter.SetInputData(self.grid)
                    self.outline_filter.Modified()
                    self.outline_filter.Update()
                self._update_theta_slice_shader_uniforms()
                try:
                    self._update_theta_slice_marker_geometry()
                except Exception:
                    pass
                self._update_cursor_line_shader_uniforms()
                try:
                    self._update_theta_boundary_planes()
                except Exception:
                    pass
                self._axis_corner = None
                try:
                    self._update_geometry_axes(force=True)
                except Exception:
                    pass
                self.renderer.ResetCameraClippingRange()
            except Exception:
                # Preserve the original failure; all replacement arrays were
                # prepared first, so this branch is only a defensive fallback.
                pass
            if render:
                self._request_render()
            raise

        self._update_roi_preview_geometry()
        if render:
            self._request_render()

    def set_theta_scale(self, scale: float, *, render: bool = True) -> None:
        """Backward-compatible wrapper for the θ-only display scale."""
        self.set_axis_scales(self._axis_scales[0], self._axis_scales[1], scale, render=render)

    @staticmethod
    def _validate_transfer_settings(settings: TransferSettings) -> None:
        if settings.lower_percentile is not None or settings.upper_percentile is not None:
            if settings.lower_percentile is None or settings.upper_percentile is None:
                raise ValueError("Для legacy percentile-режима нужно задать обе границы.")
            low = float(settings.lower_percentile)
            high = float(settings.upper_percentile)
            if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high <= 100.0):
                raise ValueError("Нужно 0 <= lower_percentile < upper_percentile <= 100.")
        else:
            low = float(settings.lower_fraction)
            high = float(settings.upper_fraction)
            if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high <= 1.0):
                raise ValueError("Нужно 0 <= нижняя доля < верхняя доля <= 1.0 (от p99.9).")
        if not (np.isfinite(settings.max_opacity) and 0.0 <= settings.max_opacity <= 1.0):
            raise ValueError("max_opacity должен лежать в [0,1].")
        if not (np.isfinite(settings.gamma) and 0.05 <= settings.gamma <= 8.0):
            raise ValueError("gamma должен лежать в [0.05,8].")
        if not (np.isfinite(settings.sample_distance_factor) and 0.25 <= settings.sample_distance_factor <= 8.0):
            raise ValueError("Коэффициент sample distance должен лежать в [0.25,8].")
        if not (np.isfinite(settings.projection_opacity) and 0.0 <= settings.projection_opacity <= 1.0):
            raise ValueError("projection_opacity должен лежать в [0,1].")
        # Validate the unit-distance factor independently of current grid spacing.
        factor = float(settings.opacity_unit_distance_factor)
        if not (np.isfinite(factor) and 0.05 <= factor <= 20.0):
            raise ValueError("opacity_unit_distance_factor должен лежать в [0.05,20].")

    def _resolve_thresholds(self, settings: TransferSettings) -> tuple[float, float, float]:
        """Return absolute scalar bounds and the cached p99.9 positive reference."""
        finite = self._finite_sample
        if finite is None or finite.size == 0:
            raise ValueError("Нет конечных значений для настройки transfer function.")

        reference = self._positive_reference_999
        positive_max = self._positive_max
        if reference is None:
            # Defensive fallback for a scene restored from an older serialized
            # state or a test fixture that assigned _finite_sample directly.
            positive = finite[finite > 0]
            if positive.size:
                reference = float(np.percentile(positive, 99.9))
                positive_max = float(np.max(positive))
            else:
                reference = float(np.max(finite))
                positive_max = None

        if settings.lower_percentile is not None:
            low, high = np.percentile(
                finite,
                [float(settings.lower_percentile), float(settings.upper_percentile)],
            )
        else:
            if positive_max is None:
                raise ValueError("В объёме нет положительных конечных интенсивностей для p99.9 reference.")
            reference = max(float(reference), float(positive_max) * 1e-9, 1e-12)
            low = reference * float(settings.lower_fraction)
            high = reference * float(settings.upper_fraction)

        low, high = float(low), float(high)
        if not (np.isfinite(low) and np.isfinite(high) and np.isfinite(reference)):
            raise ValueError("Пороговые значения или p99.9 reference не конечны.")
        if high <= low:
            # Only legacy percentile mode can become flat for a constant volume.
            delta = max(abs(low) * 1e-6, 1e-6)
            low -= delta
            high += delta
        return low, high, float(reference)

    def _apply_volume_threshold_curves(
        self, settings: TransferSettings, low: float, high: float
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Update only the scalar colour/opacity functions used by the volume."""
        value_min = float(self.grid_info["scalar_min"])
        value_max = float(self.grid_info["scalar_max"])
        sentinel = float(self.grid_info["invalid_sentinel"])
        color_low, color_high = float(low), float(high)
        if color_high <= color_low:
            color_high = color_low + max(abs(color_low) * 1e-6, 1e-6)
        color_table_info = _set_adaptive_volume_palette(
            self.color_function,
            settings.palette,
            color_low,
            color_high,
            value_min,
            value_max,
            self.vtk.vtkMath,
        )
        opacity_table_info = _set_adaptive_volume_opacity(
            self.opacity_function,
            sentinel,
            value_min,
            value_max,
            float(low),
            float(high),
            float(settings.max_opacity),
            float(settings.gamma),
            self.vtk.vtkMath,
        )
        self.settings = settings
        self._update_theta_slice_shader_uniforms(low, high)
        self.color_function.Modified()
        self.opacity_function.Modified()
        self.volume_property.Modified()
        return color_table_info, opacity_table_info

    def update_volume_contrast(
        self,
        lower_fraction: float,
        upper_fraction: float,
        *,
        render: bool = True,
    ) -> dict:
        """Apply a live Vmin/Vmax gesture without rebuilding unrelated settings.

        The positive p99.9 reference is cached per payload.  This path touches
        only the two transfer functions and the theta shader's range uniforms;
        ray sampling, projection LUTs, opacity unit distance and other staged
        display options are left untouched.
        """
        if self.payload is None or self._finite_sample is None:
            raise RuntimeError("Сначала загрузите объём.")
        settings = replace(
            self.settings,
            lower_fraction=float(lower_fraction),
            upper_fraction=float(upper_fraction),
            lower_percentile=None,
            upper_percentile=None,
        )
        self._validate_transfer_settings(settings)
        low, high, reference = self._resolve_thresholds(settings)
        color_info, opacity_info = self._apply_volume_threshold_curves(
            settings, low, high
        )
        # Preserve the last full-settings diagnostics while replacing values
        # that belong to the live contrast range.
        result = dict(self._last_transfer_info)
        result.update({
            "threshold_min": low,
            "threshold_max": high,
            "threshold_reference": reference,
            "color_table_control_points": int(color_info["control_points"]),
            "color_table_estimated_width": int(color_info["estimated_texture_width"]),
            "opacity_table_control_points": int(opacity_info["control_points"]),
            "opacity_table_estimated_width": int(opacity_info["estimated_texture_width"]),
        })
        self._last_transfer_info = result
        if render:
            self._request_render()
        return result

    def update_transfer_functions(self, settings: TransferSettings, *, render: bool = True) -> dict:
        if self.payload is None or self._finite_sample is None:
            raise RuntimeError("Сначала загрузите объём.")
        self._validate_transfer_settings(settings)
        low, high, reference = self._resolve_thresholds(settings)

        color_table_info, opacity_table_info = self._apply_volume_threshold_curves(
            settings, low, high
        )

        if self.projection_lut is not None:
            self._update_projection_palette()
        if self.projection_actor is not None:
            self.projection_actor.GetProperty().SetOpacity(float(settings.projection_opacity))
            self.projection_actor.SetVisibility(bool(settings.show_projection))
        sample_distance = max(self._min_spacing * float(settings.sample_distance_factor), 1e-6)
        opacity_unit_distance = compute_opacity_unit_distance(
            self._min_spacing, settings.opacity_unit_distance_factor
        )
        self.mapper.SetSampleDistance(float(sample_distance))
        self.volume_property.SetScalarOpacityUnitDistance(float(opacity_unit_distance))
        # _apply_volume_threshold_curves() has already marked the transfer
        # functions and volume property modified.  The mapper itself changes
        # here because its ray-sampling distance may also have changed.
        self.mapper.Modified()
        result = {
            "threshold_min": low,
            "threshold_max": high,
            "threshold_reference": reference,
            "sample_distance": float(sample_distance),
            "opacity_unit_distance": float(opacity_unit_distance),
            "projection_visible": bool(self.projection_actor.GetVisibility()) if self.projection_actor is not None else False,
            "color_table_control_points": int(color_table_info["control_points"]),
            "color_table_estimated_width": int(color_table_info["estimated_texture_width"]),
            "opacity_table_control_points": int(opacity_table_info["control_points"]),
            "opacity_table_estimated_width": int(opacity_table_info["estimated_texture_width"]),
        }
        self._last_transfer_info = result
        if render:
            self._request_render()
        return result

    def diagnostics(self) -> dict:
        """Return non-rendering scene state for logs/support bundles."""
        def _safe_bounds(prop):
            if prop is None:
                return None
            try:
                values = tuple(float(v) for v in prop.GetBounds())
                return values if len(values) == 6 and np.all(np.isfinite(values)) else None
            except Exception:
                return None

        try:
            actor_count = int(self.renderer.GetActors().GetNumberOfItems())
            volume_count = int(self.renderer.GetVolumes().GetNumberOfItems())
            prop_count = int(self.renderer.GetViewProps().GetNumberOfItems())
        except Exception:
            actor_count = volume_count = prop_count = None
        interactor_style = (
            self.interactor.GetInteractorStyle()
            if self.interactor is not None else None
        )
        def _actual_memory_kb(data_object):
            try:
                return int(data_object.GetActualMemorySize()) if data_object is not None else None
            except Exception:
                return None

        payload_volume = getattr(self.payload, "volume", None)
        payload_projection = getattr(self.payload, "projection", None)
        return {
            "render_window": self.render_window.GetClassName() if self.render_window is not None else None,
            "interactor_style": interactor_style.GetClassName() if interactor_style is not None else None,
            "trackball_motion_factor": (
                float(interactor_style.GetMotionFactor())
                if interactor_style is not None and hasattr(interactor_style, "GetMotionFactor")
                else None
            ),
            "mouse_wheel_motion_factor": (
                float(interactor_style.GetMouseWheelMotionFactor())
                if interactor_style is not None and hasattr(interactor_style, "GetMouseWheelMotionFactor")
                else None
            ),
            "renderer_prop_count": prop_count,
            "renderer_actor_count": actor_count,
            "renderer_volume_count": volume_count,
            "volume_mapper": self.mapper.GetClassName() if self.mapper is not None else None,
            "volume_dimensions": tuple(int(v) for v in self.grid.GetDimensions()) if self.grid is not None else None,
            "volume_grid_memory_kb": _actual_memory_kb(self.grid),
            "projection_grid_memory_kb": _actual_memory_kb(self.projection_grid),
            "payload_volume_bytes": int(payload_volume.nbytes) if payload_volume is not None else None,
            "payload_projection_bytes": int(payload_projection.nbytes) if payload_projection is not None else None,
            "threshold_sample_bytes": int(self._finite_sample.nbytes) if self._finite_sample is not None else None,
            "projection_mapper": self.projection_mapper.GetClassName() if self.projection_mapper is not None else None,
            "projection_dimensions": tuple(int(v) for v in self.projection_grid.GetDimensions()) if self.projection_grid is not None else None,
            "projection_visible": bool(self.projection_actor.GetVisibility()) if self.projection_actor is not None else False,
            "projection_requested_visible": bool(self.settings.show_projection),
            "peak_marker_count": int(self._peak_marker_polydata.GetNumberOfPoints()) if self._peak_marker_polydata is not None else 0,
            "peak_marker_requested_visible": bool(self._peak_marker_visible),
            "peak_marker_actor_visible": bool(self._peak_marker_actor.GetVisibility()) if self._peak_marker_actor is not None else False,
            "peak_marker_mapper": self._peak_marker_mapper.GetClassName() if self._peak_marker_mapper is not None else None,
            "peak_marker_pickable": bool(self._peak_marker_actor.GetPickable()) if self._peak_marker_actor is not None else False,
            "peak_marker_peak_id_count": int(self._peak_marker_peak_ids.size),
            "peak_marker_branch_id_count": int(np.count_nonzero(self._peak_marker_branch_ids >= 0)),
            "peak_marker_branch_count": int(len(self._peak_branch_event_counts)),
            "peak_marker_world_radius": float(self._peak_marker_world_radius),
            "peak_marker_world_radius_range": tuple(float(v) for v in getattr(self, "_peak_marker_world_radius_range", (self._peak_marker_world_radius, self._peak_marker_world_radius))),
            "peak_marker_diameter_px": float(self._peak_marker_size),
            "peak_marker_scale_count": int(getattr(self, "_peak_marker_scale_values", np.empty(0)).size),
            "peak_marker_reference_xy_pitch": float(getattr(self, "_peak_marker_reference_pitch", 0.0)),
            "peak_marker_bounds": _safe_bounds(self._peak_marker_actor),
            "branch_edge_count": int(self._branch_edge_polydata.GetNumberOfLines()) if self._branch_edge_polydata is not None else 0,
            "branch_edge_actor_visible": bool(self._branch_edge_actor.GetVisibility()) if self._branch_edge_actor is not None else False,
            "branch_correlation_matched_link_count": int(self._branch_correlation_matched_polydata.GetNumberOfLines()) if self._branch_correlation_matched_polydata is not None else 0,
            "branch_correlation_candidate_dash_count": int(self._branch_correlation_candidate_polydata.GetNumberOfLines()) if self._branch_correlation_candidate_polydata is not None else 0,
            "branch_correlation_matched_actor_visible": bool(self._branch_correlation_matched_actor.GetVisibility()) if self._branch_correlation_matched_actor is not None else False,
            "branch_correlation_candidates_visible": bool(self._branch_correlation_candidate_actor.GetVisibility()) if self._branch_correlation_candidate_actor is not None else False,
            "branch_correlation_matched_peak_count": int(self._branch_correlation_matched_peak_ids.size),
            "branch_correlation_focus_branch_count": int(self._branch_correlation_focus_branch_ids.size),
            "theta_slice_enabled": bool(self._theta_slice_enabled),
            "theta_slice_shader_installed": bool(self._theta_slice_shader_installed),
            "theta_slice_index": self._theta_slice_index,
            "theta_slice_value": (
                float(self._physical_theta[self._theta_slice_index])
                if self._physical_theta is not None and self._theta_slice_index is not None else None
            ),
            "theta_slice_display_bounds": self._theta_slice_bounds,
            "theta_slice_palette": self._theta_slice_palette,
            "theta_slice_strength": float(self._theta_slice_strength),
            "theta_slice_extra_actors": 0,  # shader highlight itself remains actor-free
            "theta_slice_marker_requested_visible": bool(self._theta_slice_marker_enabled),
            "theta_slice_marker_visible": bool(
                self._theta_slice_marker_actor.GetVisibility()
            ) if self._theta_slice_marker_actor is not None else False,
            "theta_slice_marker_opacity": float(self._theta_slice_marker_opacity),
            "theta_slice_marker_actor_present": self._theta_slice_marker_actor is not None,
            "theta_slice_marker_bounds": _safe_bounds(self._theta_slice_marker_actor),
            "theta_boundary_planes_requested_visible": bool(self._theta_boundary_planes_visible),
            "theta_boundary_planes_visible": bool(
                self._theta_boundary_plane_actor.GetVisibility()
            ) if self._theta_boundary_plane_actor is not None else False,
            "theta_boundary_plane_opacity": float(self._theta_boundary_plane_opacity),
            "theta_boundary_plane_actor_present": self._theta_boundary_plane_actor is not None,
            "theta_boundary_plane_bounds": _safe_bounds(self._theta_boundary_plane_actor),
            "theta_slice_highlight_mode": "in_volume_ray_sample_rgb",
            "theta_slice_shader_build_mtime": (
                int(self._theta_slice_shader_property.GetShaderMTime())
                if self._theta_slice_shader_property is not None else None
            ),
            "theta_slice_uniform_list_mtime": (
                int(self._theta_slice_uniforms.GetUniformListMTime())
                if self._theta_slice_uniforms is not None else None
            ),
            "theta_slice_shader_replacement_count": (
                int(self._theta_slice_shader_property.GetNumberOfShaderReplacements())
                if self._theta_slice_shader_property is not None else 0
            ),
            "cursor_line_requested_enabled": bool(self._cursor_line_enabled),
            "cursor_line_enabled": bool(self._cursor_line_enabled and self._cursor_line_position is not None),
            "cursor_line_dataset_xy": self._cursor_line_position,
            "cursor_line_radii": self._cursor_line_radii,
            "cursor_line_extra_actors": 0,
            "cursor_line_highlight_mode": "continuous_embedded_ray_marched_marker",
            "cursor_line_render_mode": "independent_marker_sample_not_volume_tint",
            "cursor_line_compositing": "front_to_back_with_volume_opacity",
            "cursor_line_theta_extent": (
                (float(self._physical_theta[0]), float(self._physical_theta[-1]))
                if self._physical_theta is not None and len(self._physical_theta) else None
            ),
            "cursor_line_density_per_voxel": float(self._cursor_line_density),
            "cursor_line_source_runs_at_zero_volume_opacity": True,
            "cursor_line_changes_volume_opacity_function": False,
            "roi_preview_highlight_mode": str(self._roi_preview_options.get("highlight_mode", "off")),
            "roi_preview_highlight_enabled": bool(
                self._roi_preview_enabled
                and self._roi_preview_options.get("highlight_mode", "off") != "off"
                and self._theta_slice_uniforms is not None
            ),
            "roi_preview_highlight_strength": float(self._roi_preview_options.get("highlight_strength", 0.0)),
            "roi_preview_highlight_color": str(self._roi_preview_options.get("highlight_color", "#FFB547")),
            "roi_preview_thickness_voxels": float(self._roi_preview_options.get("thickness_voxels", 1.5)),
            "roi_preview_highlight_mode_description": "in_volume_ray_sample_rgb_preserving_source_alpha",
            "projection_opacity": float(self.projection_actor.GetProperty().GetOpacity()) if self.projection_actor is not None else None,
            "projection_bounds": _safe_bounds(self.projection_actor),
            "projection_scalar_range": (
                (float(self.projection_info["scalar_min"]), float(self.projection_info["scalar_max"]))
                if self.projection_info is not None else None
            ),
            "projection_display_range": (
                (float(self.projection_info.get("display_scalar_min", self.projection_info["scalar_min"])),
                 float(self.projection_info.get("display_scalar_max", self.projection_info["scalar_max"])))
                if self.projection_info is not None else None
            ),
            "theta_scale": float(self._theta_scale),
            "axis_scales": tuple(float(v) for v in self._axis_scales),
            "grid_visible": bool(self._grid_visible),
            "outline_actor_present": self.outline_actor is not None,
            "outline_frame_visible": bool(self.outline_actor.GetVisibility()) if self.outline_actor is not None else False,
            "axis_tick_actor_present": self.axis_tick_actor is not None,
            "axis_tick_segment_count": int(self.axis_tick_polydata.GetNumberOfLines()) if self.axis_tick_polydata is not None else 0,
            "axis_label_count": len(self._axis_label_actors),
            "grid_actor_visible": bool(self.grid_actor.GetVisibility()) if self.grid_actor is not None else False,
            "axis_geometry_transform": tuple(
                float(self._scene_user_matrix.GetElement(r, c)) for r in range(4) for c in range(4)
            ),
            "theta_physical_range": (
                (float(self._physical_theta[0]), float(self._physical_theta[-1]))
                if self._physical_theta is not None else None
            ),
            "volume_bounds": _safe_bounds(self.volume),
            "sample_distance": float(self.mapper.GetSampleDistance()) if self.mapper is not None else None,
            "opacity_unit_distance": float(self.volume_property.GetScalarOpacityUnitDistance()) if self.volume_property is not None else None,
            "thresholds": dict(self._last_transfer_info),
            "cursor_line_color": self._cursor_line_color,
            "cursor_line_density": float(self._cursor_line_density),
            "axes_visible": bool(self._axes_visible),
            "orientation_marker_visible": bool(self._axes_visible),
            "bounding_box_visible": bool(self._bounding_box_visible),
            "axis_ticks_visible": bool(self._axis_ticks_visible),
            "theme": self.theme,
        }

    def reset_camera_fit_preserve_direction(self) -> None:
        """Clear camera pan and fit visible scene bounds without changing view direction.

        vtkRenderer.ResetCamera(bounds) computes the fitting distance/parallel
        scale for the current viewport and retains the camera projection mode.
        The original view direction and view-up are restored explicitly after
        fitting so a toolbar reset cannot unexpectedly rotate the scene.
        """
        if self.payload is None:
            return
        bounds = self.renderer.ComputeVisiblePropBounds()
        bounds_array = np.asarray(bounds, dtype=float)
        if (
            bounds_array.shape != (6,)
            or not np.all(np.isfinite(bounds_array))
            or bounds_array[0] > bounds_array[1]
            or bounds_array[2] > bounds_array[3]
            or bounds_array[4] > bounds_array[5]
        ):
            return

        camera = self.renderer.GetActiveCamera()
        old_position = np.asarray(camera.GetPosition(), dtype=float)
        old_focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        old_view_up = np.asarray(camera.GetViewUp(), dtype=float)
        direction = old_position - old_focal
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
            direction = np.asarray([1.0, -1.0, 1.0], dtype=float)
            direction_norm = float(np.linalg.norm(direction))
        direction /= direction_norm
        projection_mode = bool(camera.GetParallelProjection())

        # Native VTK fit is aspect-ratio aware and adjusts ParallelScale in
        # parallel mode or camera distance in perspective mode.
        self.renderer.ResetCamera(*bounds_array.tolist())
        fitted_focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        fitted_position = np.asarray(camera.GetPosition(), dtype=float)
        fitted_distance = float(np.linalg.norm(fitted_position - fitted_focal))
        if not np.isfinite(fitted_distance) or fitted_distance <= 1e-12:
            fitted_distance = max(
                float(bounds_array[1] - bounds_array[0]),
                float(bounds_array[3] - bounds_array[2]),
                float(bounds_array[5] - bounds_array[4]),
                1.0,
            ) * 2.8

        camera.SetFocalPoint(*fitted_focal.tolist())
        camera.SetPosition(*(fitted_focal + direction * fitted_distance).tolist())
        camera.SetViewUp(*old_view_up.tolist())
        camera.OrthogonalizeViewUp()
        camera.SetParallelProjection(int(projection_mode))
        self.renderer.ResetCameraClippingRange(*bounds_array.tolist())
        self._prepare_camera_preset_interaction()
        self._axis_corner = None
        self._update_geometry_axes(force=True)
        self._request_render()

    def _recover_trackball_interaction(self, interactor) -> None:
        """Release any stale VTK gesture/focus and synchronize the baseline."""
        style = getattr(self, "interactor_style", None)
        if style is not None:
            try:
                # StopState emits EndInteractionEvent only when a gesture is
                # actually active; ReleaseFocus is separate and is essential
                # when Qt consumed a mouse-release event outside the viewport.
                if int(style.GetState()) != 0:
                    style.StopState()
            except Exception:
                logger.debug("Could not stop stale VTK trackball state", exc_info=True)
            try:
                style.ReleaseFocus()
            except Exception:
                # ReleaseFocus is intentionally idempotent on current VTK
                # builds; guard older wrappers that expose it differently.
                logger.debug("Could not release stale VTK trackball focus", exc_info=True)
            try:
                style.SetDefaultRenderer(self.renderer)
                style.SetCurrentRenderer(self.renderer)
            except Exception:
                logger.debug("Could not restore VTK trackball renderer", exc_info=True)
        if interactor is not None:
            try:
                x, y = map(int, interactor.GetEventPosition())
                # SetLastEventPosition does not synthesize a mouse event. Do
                # not call SetEventPosition here: VTK's SetEventPosition also
                # shifts the previous position automatically.
                interactor.SetLastEventPosition(x, y)
            except Exception:
                logger.debug("Could not synchronize VTK pointer baseline", exc_info=True)

    def capture_camera_state(self) -> dict[str, Any] | None:
        """Return a JSON-safe camera pose and projection configuration."""
        try:
            camera = self.renderer.GetActiveCamera()
            values = {
                "position": [float(v) for v in camera.GetPosition()],
                "focal_point": [float(v) for v in camera.GetFocalPoint()],
                "view_up": [float(v) for v in camera.GetViewUp()],
                "parallel_projection": bool(camera.GetParallelProjection()),
                "parallel_scale": float(camera.GetParallelScale()),
                "view_angle": float(camera.GetViewAngle()),
            }
            numeric = (
                *values["position"], *values["focal_point"], *values["view_up"],
                values["parallel_scale"], values["view_angle"],
            )
            if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
                return None
            return values
        except Exception:
            logger.debug("Could not capture VTK camera state", exc_info=True)
            return None

    def restore_camera_state(self, state: dict[str, Any], *, render: bool = True) -> bool:
        """Restore a previously validated camera pose for the current payload."""
        try:
            position = np.asarray(state.get("position"), dtype=np.float64)
            focal = np.asarray(state.get("focal_point"), dtype=np.float64)
            view_up = np.asarray(state.get("view_up"), dtype=np.float64)
            parallel_scale = float(state.get("parallel_scale", 1.0))
            view_angle = float(state.get("view_angle", 30.0))
            if (
                position.shape != (3,) or focal.shape != (3,) or view_up.shape != (3,)
                or not np.isfinite(position).all() or not np.isfinite(focal).all()
                or not np.isfinite(view_up).all()
                or float(np.linalg.norm(position - focal)) <= 1.0e-10
                or float(np.linalg.norm(view_up)) <= 1.0e-10
                or not np.isfinite(parallel_scale) or parallel_scale <= 0.0
                or not np.isfinite(view_angle) or not 1.0 <= view_angle <= 179.0
            ):
                return False
            camera = self.renderer.GetActiveCamera()
            camera.SetFocalPoint(*focal.tolist())
            camera.SetPosition(*position.tolist())
            camera.SetViewUp(*view_up.tolist())
            camera.OrthogonalizeViewUp()
            camera.SetParallelProjection(1 if bool(state.get("parallel_projection", False)) else 0)
            camera.SetParallelScale(parallel_scale)
            camera.SetViewAngle(view_angle)
            self.renderer.ResetCameraClippingRange()
            self._prepare_camera_preset_interaction()
            self._axis_corner = None
            self._update_geometry_axes(force=True)
            if render:
                self._request_render()
            return True
        except Exception:
            logger.exception("Could not restore saved VTK camera state")
            return False

    def _prepare_camera_preset_interaction(self) -> None:
        """End any stale gesture and anchor future drag to its first real press.

        VTK's TrackballCamera grabs interactor focus on button-down and normally
        releases it on button-up. Qt menus/toolbars can consume that release;
        StopState alone changes the state but does not release the observer's
        focus. Both are therefore reset before and after a programmatic preset.
        """
        self._recover_trackball_interaction(getattr(self, "interactor", None))

    def set_perspective_view(self) -> None:
        """Set the canonical oblique/isometric camera pose in perspective mode.

        This is a one-shot command for a useful three-axis oblique view. It is
        not a projection-mode checkbox and therefore cannot accidentally switch
        perspective off. The direction is based on this viewer's stable initial
        camera pose used by set_payload.
        """
        if self.payload is None:
            return
        self._prepare_camera_preset_interaction()
        bounds = self.renderer.ComputeVisiblePropBounds()
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if (
            bounds_array.shape != (6,)
            or not np.all(np.isfinite(bounds_array))
            or bounds_array[0] > bounds_array[1]
            or bounds_array[2] > bounds_array[3]
            or bounds_array[4] > bounds_array[5]
        ):
            return
        center = np.asarray(
            [
                0.5 * (bounds_array[0] + bounds_array[1]),
                0.5 * (bounds_array[2] + bounds_array[3]),
                0.5 * (bounds_array[4] + bounds_array[5]),
            ],
            dtype=np.float64,
        )
        camera = self.renderer.GetActiveCamera()
        diagonal = max(
            float(np.linalg.norm([
                bounds_array[1] - bounds_array[0],
                bounds_array[3] - bounds_array[2],
                bounds_array[5] - bounds_array[4],
            ])),
            1.0,
        )
        direction = np.asarray([1.7, -2.3, 1.6], dtype=np.float64)
        direction /= float(np.linalg.norm(direction))
        camera.SetFocalPoint(*center.tolist())
        camera.SetPosition(*(center + direction * diagonal).tolist())
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.OrthogonalizeViewUp()
        # Explicitly enable perspective; this command is never a toggle-off.
        camera.SetParallelProjection(0)
        self.renderer.ResetCamera(*bounds_array.tolist())
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.OrthogonalizeViewUp()
        camera.SetParallelProjection(0)
        self.renderer.ResetCameraClippingRange(*bounds_array.tolist())
        self._prepare_camera_preset_interaction()
        self._axis_corner = None
        self._update_geometry_axes(force=True)
        self._request_render()

    def reset_camera_axis(self, axis: str, polarity: int = 1) -> None:
        if self.payload is None:
            return
        axis = str(axis).lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("Ось камеры должна быть x, y или z.")
        polarity = int(polarity)
        if polarity not in (-1, 1):
            raise ValueError("Полярность направления камеры должна быть +1 или -1.")
        self._prepare_camera_preset_interaction()
        bounds = self.renderer.ComputeVisiblePropBounds()
        if not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
            return
        center = np.array([
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ], dtype=float)
        radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
        directions = {
            "x": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "y": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "z": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        }
        direction, up = directions[axis]
        direction = direction * float(polarity)
        camera = self.renderer.GetActiveCamera()
        projection_mode = bool(camera.GetParallelProjection())
        camera.SetFocalPoint(*center.tolist())
        camera.SetPosition(*(center + direction * radius * 2.8).tolist())
        camera.SetViewUp(*up.tolist())
        camera.OrthogonalizeViewUp()
        # Ask VTK to calculate a viewport/aspect-aware fit for the canonical
        # direction instead of relying on a fixed camera distance. ResetCamera
        # preserves the direction just established above and the projection
        # mode is restored explicitly as a guard against future VTK changes.
        self.renderer.ResetCamera(*[float(value) for value in bounds])
        camera.SetViewUp(*up.tolist())
        camera.OrthogonalizeViewUp()
        camera.SetParallelProjection(int(projection_mode))
        self.renderer.ResetCameraClippingRange(*[float(value) for value in bounds])
        self._prepare_camera_preset_interaction()
        self._axis_corner = None
        self._update_geometry_axes(force=True)
        self._request_render()

    def set_perspective(self, enabled: bool = True) -> None:
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelProjection(0 if enabled else 1)
        self.renderer.ResetCameraClippingRange()
        self._request_render()

    def clear(self, *, render: bool = True) -> None:
        """Remove current props; optionally schedule a QVTK repaint."""
        had_content = self.volume is not None or self.projection_actor is not None
        if self.volume is not None:
            self.renderer.RemoveVolume(self.volume)
        if self.projection_actor is not None:
            self.renderer.RemoveActor(self.projection_actor)
        self._theta_slice_shader_property = None
        self._theta_slice_uniforms = None
        self._theta_slice_shader_installed = False
        self._theta_slice_bounds = None
        self._theta_slice_transfer_range = None
        self._theta_slice_index = None
        self._theta_slice_enabled = False
        self._theta_slice_marker_enabled = False
        if self._theta_slice_marker_actor is not None:
            self._theta_slice_marker_actor.SetVisibility(0)
        if self._peak_marker_actor is not None:
            self._peak_marker_actor.SetVisibility(0)
            self._peak_marker_polydata.Initialize()
        self._peak_marker_positions_physical = np.empty((0, 3), dtype=np.float64)
        self._peak_marker_colors = np.empty((0, 4), dtype=np.uint8)
        self._peak_marker_base_colors = np.empty((0, 4), dtype=np.uint8)
        self._peak_marker_peak_ids = np.empty(0, dtype=np.int64)
        self._peak_marker_branch_ids = np.empty(0, dtype=np.int64)
        self._peak_marker_analysis_included = np.empty(0, dtype=bool)
        self._peak_marker_rendered_indices = np.empty(0, dtype=np.int64)
        self._show_excluded_branches = False
        self._peak_branch_event_counts = {}
        self._peak_excluded_branch_event_counts = {}
        self._peak_marker_visible = False
        if self._branch_edge_actor is not None:
            self._branch_edge_actor.SetVisibility(0)
            self._branch_edge_polydata.Initialize()
        self._branch_edge_positions_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_edge_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_edge_base_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_edge_branch_ids = np.empty(0, dtype=np.int64)
        self._branch_edge_analysis_included = np.empty(0, dtype=bool)
        self._branch_edge_visible = False
        self._branch_correlation_matched_actor.SetVisibility(0)
        self._branch_correlation_candidate_actor.SetVisibility(0)
        self._branch_correlation_matched_polydata.Initialize()
        self._branch_correlation_candidate_polydata.Initialize()
        self._branch_correlation_matched_segments_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_correlation_matched_colors = np.empty((0, 4), dtype=np.uint8)
        self._branch_correlation_candidate_segments_physical = np.empty((0, 2, 3), dtype=np.float64)
        self._branch_correlation_matched_peak_ids = np.empty(0, dtype=np.int64)
        self._branch_correlation_focus_branch_ids = np.empty(0, dtype=np.int64)
        self._branch_correlation_visible = False
        self._branch_correlation_show_candidates = False
        self._apply_branch_correlation_peak_emphasis(update_polydata=False)
        self._cursor_line_enabled = False
        self._cursor_line_position = None
        self._cursor_line_radii = None
        self.set_theta_range_interaction_active(False, render=False)
        self.payload = None
        self._apply_roi_preview_visibility()
        self.grid = None
        self.grid_info = None
        self.mapper = None
        self.volume = None
        self.volume_property = None
        if self.outline_filter is not None:
            self.outline_filter.RemoveAllInputConnections(0)
        for label_actor in self._axis_label_actors:
            self.renderer.RemoveActor(label_actor)
        self._axis_label_actors.clear()
        self._axis_label_title_flags.clear()
        if self.axis_tick_polydata is not None:
            self.axis_tick_polydata.Initialize()
        if self.grid_polydata is not None:
            self.grid_polydata.Initialize()
        self._axis_corner = None
        self._physical_x = None
        self._physical_y = None
        self._physical_theta = None
        self._display_x = None
        self._display_y = None
        self._display_theta = None
        self._theta_display_range_indices = None
        self._theta_boundary_planes_visible = False
        self._theta_boundary_plane_opacity = 0.12
        if self._theta_boundary_plane_actor is not None:
            self._theta_boundary_plane_actor.SetVisibility(0)
            self._theta_boundary_plane_actor.GetProperty().SetOpacity(
                self._theta_boundary_plane_opacity
            )
        self._scene_base_matrix.Identity()
        self._scene_user_matrix.Identity()
        self._axis_scales = (1.0, 1.0, 1.0)
        for prop in (
            self.outline_actor,
            self.axis_tick_actor,
            self.grid_actor,
            self._theta_boundary_plane_actor,
            self._theta_slice_marker_actor,
            self._peak_marker_actor,
            self._branch_edge_actor,
            self._branch_correlation_matched_actor,
            self._branch_correlation_candidate_actor,
        ):
            if prop is not None:
                prop.SetVisibility(0)
        self.projection_grid = None
        self.projection_mapper = None
        self.projection_actor = None
        self.projection_lut = None
        self.projection_info = None
        self._finite_sample = None
        self._positive_reference_999 = None
        self._positive_max = None
        self._percentile_sample_stride = 1
        self._projection_base_z = None
        self._theta_scale = 1.0
        self._last_transfer_info = {}
        # A combo-box signal can run while Qt is processing another GUI event.
        # _request_render() schedules QWidget.update through the QVTK owner;
        # it does not invoke the VTK interactor's immediate Render() method.
        if had_content and render:
            self._request_render()

    def close(self) -> None:
        """Stop scene-owned observers before the owning QVTK widget finalizes.

        QVTK is the sole owner of the native render-window/interactor teardown.
        This method removes only observers and the optional orientation widget
        that this scene explicitly registered. It deliberately does not detach
        VTK's interactor style/renderer graph node by node: doing so during a
        Qt close event can invalidate objects still used by QVTK's own
        Finalize()/parent-destroyed lifecycle.
        """
        if self._closed:
            return
        self._closed = True
        interactor = self.interactor

        if self._camera_observer_id is not None and interactor is not None:
            try:
                interactor.RemoveObserver(self._camera_observer_id)
            except Exception:
                logger.debug("Could not remove the scene camera observer during close", exc_info=True)
        self._camera_observer_id = None

        try:
            camera = self.renderer.GetActiveCamera() if self.renderer is not None else None
            if self._peak_marker_camera_observer_id is not None and camera is not None:
                camera.RemoveObserver(self._peak_marker_camera_observer_id)
        except Exception:
            logger.debug("Could not remove the peak-marker camera observer during close", exc_info=True)
        self._peak_marker_camera_observer_id = None
        try:
            if self._peak_marker_resize_observer_id is not None and self.render_window is not None:
                self.render_window.RemoveObserver(self._peak_marker_resize_observer_id)
        except Exception:
            logger.debug("Could not remove the peak-marker resize observer during close", exc_info=True)
        self._peak_marker_resize_observer_id = None

        style = getattr(self, "interactor_style", None)
        if style is not None:
            try:
                style.StopState()
            except Exception:
                logger.debug("Could not stop VTK interaction state during close", exc_info=True)

        axes_widget = self.axes_widget
        if axes_widget is not None:
            try:
                # SetEnabled(0) is the widget's supported observer-unregistration
                # path. Avoid redundant SetInteractor/SetRenderer/SetMarker(None)
                # calls while QVTK is about to finalize the interactor.
                axes_widget.SetEnabled(0)
            except Exception:
                logger.debug("Could not disable the VTK orientation marker during close", exc_info=True)
        self.axes_widget = None

        # Keep renderer/interactor/style ownership intact until QVTK Finalize.
        self._render_callback = None

