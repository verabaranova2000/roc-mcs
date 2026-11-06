"""Regression test for the GPU-only theta display preview.

Run from the repository root:
    python tools/verify_vtk_theta_preview.py

The test intentionally does not render or initialize an OpenGL window. It
validates the VTK mapper state and proves that the preview leaves the payload,
scalar array, coordinate grids, XY projection, and camera untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roc_mcs.gui.roi_explorer.vtk_backend.vtk_scene import (  # noqa: E402
    TransferSettings,
    VTKVolumeScene,
)


def _payload(theta_values=(11.0, 11.3, 11.7, 12.2, 13.0, 14.0)):
    theta = np.asarray(theta_values, dtype=np.float64)
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    volume = (
        np.arange(x.size * y.size * theta.size, dtype=np.float32)
        .reshape(x.size, y.size, theta.size)
        + 1.0
    )
    projection = np.sum(volume, axis=2)
    return SimpleNamespace(
        volume=volume,
        projection=projection,
        x=x,
        y=y,
        theta=theta,
        coord_system="xy",
        roi=None,
    )


def _camera_signature(scene):
    camera = scene.renderer.GetActiveCamera()
    return (
        tuple(float(v) for v in camera.GetPosition()),
        tuple(float(v) for v in camera.GetFocalPoint()),
        tuple(float(v) for v in camera.GetViewUp()),
        bool(camera.GetParallelProjection()),
        float(camera.GetParallelScale()),
    )


def _grid_coordinates(scene):
    return tuple(
        vtk_to_numpy(array).copy()
        for array in (
            scene.grid.GetXCoordinates(),
            scene.grid.GetYCoordinates(),
            scene.grid.GetZCoordinates(),
        )
    )


def _assert_close_tuple(actual, expected, *, atol=1e-10):
    if len(actual) != len(expected) or not np.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"Ожидалось {expected}, получено {actual}")


def main() -> int:
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetMultiSamples(0)
    render_window.SetSize(128, 128)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    scene = VTKVolumeScene(render_window, interactor, render_callback=None)
    try:
        payload = _payload()
        source_volume = payload.volume.copy()
        source_projection = payload.projection.copy()

        scene.set_payload(payload, TransferSettings(), axis_scales=(1.0, 1.0, 1.0))
        mapper = scene.mapper
        grid = scene.grid
        projection_grid = scene.projection_grid
        projection_actor = scene.projection_actor
        boundary_actor = scene._theta_boundary_plane_actor
        renderer_actor_count = int(scene.renderer.GetActors().GetNumberOfItems())

        original_camera = _camera_signature(scene)
        original_coordinates = _grid_coordinates(scene)
        original_scalars = vtk_to_numpy(grid.GetPointData().GetScalars()).copy()
        original_projection_bounds = tuple(projection_actor.GetBounds())
        original_projection_matrix = tuple(
            projection_actor.GetUserMatrix().GetElement(row, col)
            for row in range(4)
            for col in range(4)
        )
        original_projection_visibility = bool(projection_actor.GetVisibility())

        if mapper.GetCropping():
            raise AssertionError("Новая сцена должна начинаться с полного диапазона θ.")
        if boundary_actor is None or boundary_actor.GetVisibility():
            raise AssertionError("Граничные плоскости должны существовать, но быть выключены по умолчанию.")

        applied = scene.set_theta_display_range_indices(1, 4, render=False)
        if applied != (1, 4):
            raise AssertionError(f"Неверный диапазон индексов: {applied}")
        if scene.mapper is not mapper or scene.grid is not grid:
            raise AssertionError("Предпросмотр неожиданно пересоздал mapper или сетку.")
        if not mapper.GetCropping():
            raise AssertionError("GPU mapper не включил cropping.")

        crop_planes = tuple(float(v) for v in mapper.GetCroppingRegionPlanes())
        _assert_close_tuple(crop_planes, (0.0, 4.0, -1.0, 2.0, 11.15, 13.5))

        # The optional translucent caps follow the exact selected interval,
        # remain hidden by default, and do not alter the mapper crop itself.
        if scene.set_theta_boundary_planes_visible(True, render=False) is not True:
            raise AssertionError("Не удалось включить граничные плоскости θ.")
        if not boundary_actor.GetVisibility():
            raise AssertionError("Включённые граничные плоскости не видны.")
        _assert_close_tuple(
            tuple(float(v) for v in boundary_actor.GetBounds()),
            (0.0, 4.0, -1.0, 2.0, 11.159, 13.47),
            atol=1e-8,
        )
        if not np.isclose(boundary_actor.GetProperty().GetOpacity(), 0.12):
            raise AssertionError("Начальная прозрачность граничных плоскостей должна быть 12%.")
        scene.set_theta_boundary_plane_opacity(0.20, render=False)
        if not np.isclose(boundary_actor.GetProperty().GetOpacity(), 0.20):
            raise AssertionError("Изменение прозрачности не применилось к actor.")
        crop_planes_before_marker_adjustment = tuple(
            float(v) for v in mapper.GetCroppingRegionPlanes()
        )
        try:
            scene.set_theta_boundary_plane_opacity(0.50, render=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Недопустимая прозрачность граничных плоскостей была принята.")
        if not np.isclose(boundary_actor.GetProperty().GetOpacity(), 0.20):
            raise AssertionError("Отклонённая прозрачность повредила предыдущее значение.")
        dark_plane_color = tuple(
            float(v) for v in boundary_actor.GetProperty().GetColor()
        )
        scene.set_theme("light")
        light_plane_color = tuple(
            float(v) for v in boundary_actor.GetProperty().GetColor()
        )
        if dark_plane_color == light_plane_color:
            raise AssertionError("Цвет граничных плоскостей не адаптируется к теме сцены.")
        scene.set_theme("dark")
        if not np.isclose(boundary_actor.GetProperty().GetOpacity(), 0.20):
            raise AssertionError("Переключение темы сбросило пользовательскую прозрачность.")
        scene.set_theta_boundary_planes_visible(False, render=False)
        if boundary_actor.GetVisibility() or not mapper.GetCropping():
            raise AssertionError("Скрытие маркеров изменило crop или не скрыло плоскости.")
        scene.set_theta_boundary_planes_visible(True, render=False)
        if tuple(float(v) for v in mapper.GetCroppingRegionPlanes()) != crop_planes_before_marker_adjustment:
            raise AssertionError("Включение маркеров изменило clipping planes GPU-маппера.")

        # The preview is a view-only clipping operation: nothing in the data
        # pipeline or the independent projection actor is modified.
        if not np.array_equal(payload.volume, source_volume):
            raise AssertionError("Предпросмотр изменил payload.volume.")
        if not np.array_equal(payload.projection, source_projection):
            raise AssertionError("Предпросмотр изменил payload.projection.")
        for actual, expected in zip(_grid_coordinates(scene), original_coordinates):
            if not np.array_equal(actual, expected):
                raise AssertionError("Предпросмотр изменил координатную сетку.")
        if not np.array_equal(vtk_to_numpy(grid.GetPointData().GetScalars()), original_scalars):
            raise AssertionError("Предпросмотр изменил массив интенсивностей.")
        if scene.projection_grid is not projection_grid or scene.projection_actor is not projection_actor:
            raise AssertionError("Предпросмотр пересоздал XY-проекцию.")
        if tuple(projection_actor.GetBounds()) != original_projection_bounds:
            raise AssertionError("Предпросмотр изменил геометрию XY-проекции.")
        if tuple(
            projection_actor.GetUserMatrix().GetElement(row, col)
            for row in range(4)
            for col in range(4)
        ) != original_projection_matrix:
            raise AssertionError("Предпросмотр изменил матрицу XY-проекции.")
        if bool(projection_actor.GetVisibility()) != original_projection_visibility:
            raise AssertionError("Предпросмотр изменил видимость XY-проекции.")
        if _camera_signature(scene) != original_camera:
            raise AssertionError("Предпросмотр изменил положение или режим камеры.")

        # Invalid requests must fail without changing the active crop.
        try:
            scene.set_theta_display_range_indices(4, 4, render=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Недопустимый диапазон θ был принят.")
        if not mapper.GetCropping() or scene.theta_display_range_indices != (1, 4):
            raise AssertionError("Ошибка валидации повредила ранее установленный диапазон.")

        # Display-scale changes keep the same physical sample indices and
        # recompute mapper planes from the new displayed theta coordinates.
        scene.set_axis_scales(2.0, 0.5, 1.5, render=False)
        if scene.theta_display_range_indices != (1, 4) or not mapper.GetCropping():
            raise AssertionError("Масштабирование потеряло активный диапазон θ.")
        display_theta = np.asarray(scene._display_theta, dtype=np.float64)
        expected_low = 0.5 * (display_theta[0] + display_theta[1])
        expected_high = 0.5 * (display_theta[4] + display_theta[5])
        scaled_planes = tuple(float(v) for v in mapper.GetCroppingRegionPlanes())
        if not (
            np.isclose(scaled_planes[4], expected_low)
            and np.isclose(scaled_planes[5], expected_high)
        ):
            raise AssertionError(
                "Clipping planes не пересчитались после масштаба θ: "
                f"{scaled_planes[4:6]} вместо {(expected_low, expected_high)}"
            )
        if not boundary_actor.GetVisibility():
            raise AssertionError("Граничные плоскости выключились после изменения масштаба.")
        if int(scene.renderer.GetActors().GetNumberOfItems()) != renderer_actor_count:
            raise AssertionError("Перетаскивание диапазона/масштаба создало лишние VTK actors.")
        scaled_bounds = tuple(float(v) for v in boundary_actor.GetBounds())
        _assert_close_tuple(
            scaled_bounds,
            (-2.0, 6.0, -0.25, 1.25, 10.4885, 13.955),
            atol=1e-8,
        )
        actor_matrix = boundary_actor.GetUserMatrix()
        if any(
            not np.isclose(
                actor_matrix.GetElement(row, col),
                scene._scene_user_matrix.GetElement(row, col),
            )
            for row in range(4)
            for col in range(4)
        ):
            raise AssertionError("Граничные плоскости потеряли общую матрицу ROI-преобразования.")
        if _camera_signature(scene) != original_camera:
            raise AssertionError("Изменение диапазона/масштаба неожиданно изменило камеру.")

        scene.set_theta_display_range_indices(0, len(payload.theta) - 1, render=False)
        if mapper.GetCropping() or scene.theta_display_range_indices is not None:
            raise AssertionError("Сброс полного диапазона не выключил GPU cropping.")

        scene.set_theta_boundary_planes_visible(False, render=False)
        if boundary_actor.GetVisibility() or scene.theta_boundary_planes_visible:
            raise AssertionError("Граничные плоскости не выключились по запросу.")

        # Loading another payload must start from its full extent, never reuse
        # stale indices from the previous scan.
        scene.set_theta_display_range_indices(1, 4, render=False)
        next_payload = _payload((20.0, 20.5, 21.5, 23.0, 25.0))
        scene.set_payload(next_payload, TransferSettings(), axis_scales=(2.0, 0.5, 1.5))
        if scene.mapper.GetCropping() or scene.theta_display_range_indices is not None:
            raise AssertionError("Новый payload унаследовал предпросмотр предыдущего scan.")
        if scene.theta_boundary_planes_visible or boundary_actor.GetVisibility():
            raise AssertionError("Новый payload унаследовал включённые граничные плоскости.")
        if not np.isclose(boundary_actor.GetProperty().GetOpacity(), 0.12):
            raise AssertionError("Новый payload не сбросил прозрачность граничных плоскостей к 12%.")

        print("PASS: VTK theta display-preview regression")
        print(f"  selected sample indices: (1, 4)")
        print(f"  initial GPU crop planes: {crop_planes}")
        print(f"  scaled GPU crop planes: {scaled_planes}")
        print("  payload / intensity grid / XY projection / camera: unchanged during preview")
        print("  optional boundary planes: geometry, opacity, scale-following, reset: PASS")
        print("  reset and new-payload full-range behavior: PASS")
        return 0
    finally:
        scene.close()


if __name__ == "__main__":
    raise SystemExit(main())
