"""VTK GPU volume scene for the ROI Explorer HDF5 adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
        # In QVTKRenderWindowInteractor, Render() schedules QWidget.update();
        # the actual VTK render then happens inside Qt's paintEvent, when the
        # native OpenGL context is current. Never call render_window.Render()
        # directly from GUI slots.
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
        self._camera_observer_id = None
        self.projection_grid = None
        self.projection_mapper = None
        self.projection_actor = None
        self.projection_lut = None
        self.projection_info = None
        # Selected-theta emphasis is injected into the existing GPU volume
        # ray-casting shader. No second actor, point mesh, plane or geometry is
        # created; only the color of samples inside the selected theta slab is
        # blended with a user-selected palette. The original opacity function
        # and volume mapper remain authoritative.
        self._theta_slice_index: int | None = None
        self._theta_slice_enabled = False
        self._theta_slice_palette = "turbo"
        self._theta_slice_strength = 0.90
        self._theta_slice_shader_property = None
        self._theta_slice_uniforms = None
        self._theta_slice_shader_installed = False
        self._theta_slice_bounds: tuple[float, float] | None = None
        self._theta_slice_transfer_range: tuple[float, float] | None = None
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
        corner = np.asarray([hi if local_camera[i] >= center[i] else lo for i, (lo, hi) in enumerate(ranges)])
        corner_key = tuple(int(local_camera[i] >= center[i]) for i in range(3))
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
        name = str(configured) if configured else ("Greys_r" if self.theme == "dark" else "Greys")
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
            "if (rocThetaSliceEnabled != 0 || rocCursorLineEnabled != 0) {",
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
            self._theta_slice_shader_color_code() + "\n" + theta_anchor,
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

    @property
    def theta_slice_index(self) -> int | None:
        return self._theta_slice_index

    @property
    def theta_slice_enabled(self) -> bool:
        return bool(self._theta_slice_enabled)

    @property
    def theta_slice_palette(self) -> str:
        return str(self._theta_slice_palette)

    @property
    def theta_slice_strength(self) -> float:
        return float(self._theta_slice_strength)

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
            raise ValueError("Цвет линии должен быть задан в формате #RRGGBB.")
        try:
            return tuple(int(value[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
        except ValueError as exc:
            raise ValueError("Цвет линии должен быть задан в формате #RRGGBB.") from exc

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
        if settings.lower_percentile is None and not np.any(new_finite_sample > 0):
            raise ValueError("В объёме нет положительных конечных интенсивностей для p99.9 reference.")

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
        if self._theta_boundary_plane_actor is not None:
            self._theta_boundary_plane_actor.SetVisibility(0)
            self._theta_boundary_plane_actor.GetProperty().SetOpacity(
                self._theta_boundary_plane_opacity
            )

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
        # Gradient opacity is intentionally neutral; weak data are not suppressed by gradients.
        gradient = vtk.vtkPiecewiseFunction()
        data_min = float(self.grid_info["scalar_min"])
        data_max = float(self.grid_info["scalar_max"])
        gradient.AddPoint(data_min, 1.0)
        gradient.AddPoint(data_max if data_max > data_min else data_min + 1.0, 1.0)
        self.volume_property.SetGradientOpacity(gradient)

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
        # Actor transforms contain only the ROI-to-detector transform.
        # Display scale lives in the rectilinear-grid coordinate arrays.
        for prop in (self.outline_actor, self.axis_tick_actor, self.grid_actor):
            if prop is not None:
                prop.SetUserMatrix(self._scene_user_matrix)
        self.set_theme(self.theme)
        self._last_transfer_info = self.update_transfer_functions(self.settings, render=False)
        self._update_theta_slice_shader_uniforms()
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
        self._update_theta_boundary_planes()
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
        previous_enabled = bool(self.mapper.GetCropping())
        previous_planes = tuple(float(v) for v in self.mapper.GetCroppingRegionPlanes())
        previous_flags = int(self.mapper.GetCroppingRegionFlags())

        self._theta_display_range_indices = requested
        try:
            self._apply_theta_display_crop()
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

        if render:
            self._request_render()
        return (low, high)

    @property
    def theta_display_range_indices(self) -> tuple[int, int] | None:
        """Current GPU-preview crop; None represents the complete theta extent."""
        return self._theta_display_range_indices

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
        """Return absolute scalar bounds and the p99.9 positive reference."""
        finite = self._finite_sample
        if finite is None or finite.size == 0:
            raise ValueError("Нет конечных значений для настройки transfer function.")
        if settings.lower_percentile is not None:
            low, high = np.percentile(
                finite,
                [float(settings.lower_percentile), float(settings.upper_percentile)],
            )
            reference = float(np.percentile(finite[finite > 0], 99.9)) if np.any(finite > 0) else float(np.max(finite))
        else:
            positive = finite[finite > 0]
            if positive.size == 0:
                raise ValueError("В объёме нет положительных конечных интенсивностей для p99.9 reference.")
            reference = float(np.percentile(positive, 99.9))
            reference = max(reference, float(np.max(positive)) * 1e-9, 1e-12)
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

    def update_transfer_functions(self, settings: TransferSettings, *, render: bool = True) -> dict:
        if self.payload is None or self._finite_sample is None:
            raise RuntimeError("Сначала загрузите объём.")
        self._validate_transfer_settings(settings)
        low, high, reference = self._resolve_thresholds(settings)

        value_min = float(self.grid_info["scalar_min"])
        value_max = float(self.grid_info["scalar_max"])
        sentinel = float(self.grid_info["invalid_sentinel"])
        self.color_function.RemoveAllPoints()
        rgb = _palette_samples(settings.palette, 64)
        # Match colour contrast to the same robust intensity interval as the
        # opacity ramp. Rare hot pixels remain in the data/diagnostics but can
        # no longer stretch the entire transfer function and flatten the volume.
        color_low, color_high = float(low), float(high)
        if color_high <= color_low:
            color_high = color_low + max(abs(color_low) * 1e-6, 1e-6)
        for i, color in enumerate(rgb):
            value = color_low + (color_high - color_low) * (i / (len(rgb) - 1))
            self.color_function.AddRGBPoint(float(value), *map(float, color))

        self.opacity_function.RemoveAllPoints()
        self.opacity_function.AddPoint(sentinel, 0.0)
        self.opacity_function.AddPoint(value_min, 0.0)
        self.opacity_function.AddPoint(low, 0.0)
        # The same gamma ramp is used by the VTK prototype; thresholds now
        # follow its p99.9-fraction semantics rather than array percentiles.
        for t in np.linspace(0.0, 1.0, 17):
            value = low + (high - low) * float(t)
            alpha = float(settings.max_opacity) * (float(t) ** float(settings.gamma))
            self.opacity_function.AddPoint(value, alpha)
        if value_max > high:
            self.opacity_function.AddPoint(value_max, float(settings.max_opacity))

        self.settings = settings
        self._update_theta_slice_shader_uniforms(low, high)
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
        self.color_function.Modified()
        self.opacity_function.Modified()
        self.volume_property.Modified()
        self.mapper.Modified()
        result = {
            "threshold_min": low,
            "threshold_max": high,
            "threshold_reference": reference,
            "sample_distance": float(sample_distance),
            "opacity_unit_distance": float(opacity_unit_distance),
            "projection_visible": bool(self.projection_actor.GetVisibility()) if self.projection_actor is not None else False,
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
            "theta_slice_extra_actors": 0,
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
        self._update_geometry_axes(force=True)
        self._request_render()

    def set_perspective_view(self) -> None:
        """Set the canonical oblique/isometric camera pose in perspective mode.

        This is a one-shot command for a useful three-axis oblique view. It is
        not a projection-mode checkbox and therefore cannot accidentally switch
        perspective off. The direction is based on this viewer's stable initial
        camera pose used by set_payload.
        """
        if self.payload is None:
            return
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
        self._cursor_line_enabled = False
        self._cursor_line_position = None
        self._cursor_line_radii = None
        self.payload = None
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
        ):
            if prop is not None:
                prop.SetVisibility(0)
        self.projection_grid = None
        self.projection_mapper = None
        self.projection_actor = None
        self.projection_lut = None
        self.projection_info = None
        self._finite_sample = None
        self._percentile_sample_stride = 1
        self._projection_base_z = None
        self._theta_scale = 1.0
        self._last_transfer_info = {}
        # A combo-box signal can run while Qt is processing another GUI event.
        # QVTKRenderWindowInteractor.Render() is the non-blocking update() API;
        # it schedules paintEvent instead of rendering outside the GL context.
        if had_content and render:
            self._request_render()

    def close(self) -> None:
        """Release scene props before QVTK finalizes its native GL window."""
        if self._closed:
            return
        # Remove props while the QVTK render window and its GL context still
        # exist. vtkRenderer owns the single ReleaseGraphicsResources call.
        self.clear(render=False)
        self._closed = True
        try:
            if self._camera_observer_id is not None and self.interactor is not None:
                self.interactor.RemoveObserver(self._camera_observer_id)
                self._camera_observer_id = None
        except Exception:
            pass
        try:
            if self.axes_widget is not None:
                self.axes_widget.SetEnabled(0)
        except Exception:
            pass
        self._render_callback = None
