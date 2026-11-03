#!/usr/bin/env python3
"""Read-only VTK OpenGL object-factory preflight; does not create a GUI or GL context."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
        from vtkmodules.vtkRenderingCore import vtkRenderWindow
        from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
        from vtkmodules.vtkCommonCore import vtkVersion
    except Exception as exc:
        print(f"IMPORT_FAILED: {type(exc).__name__}: {exc}")
        return 10

    window = vtkRenderWindow()
    mapper = vtkGPUVolumeRayCastMapper()
    window_class = window.GetClassName()
    mapper_class = mapper.GetClassName()
    window_ok = bool(window.IsA("vtkOpenGLRenderWindow"))
    mapper_ok = bool(mapper.IsA("vtkOpenGLGPUVolumeRayCastMapper"))

    print(f"VTK_VERSION={vtkVersion.GetVTKVersion()}")
    print(f"RENDER_WINDOW_CLASS={window_class}")
    print(f"RENDER_WINDOW_IS_OPENGL={window_ok}")
    print(f"VOLUME_MAPPER_CLASS={mapper_class}")
    print(f"VOLUME_MAPPER_IS_OPENGL_GPU={mapper_ok}")

    if not window_ok:
        print("FAIL: vtkRenderingOpenGL2 factory did not create an OpenGL render window.")
        return 20
    if not mapper_ok:
        print("FAIL: vtkRenderingVolumeOpenGL2 factory did not create an OpenGL GPU mapper.")
        return 21

    print("PASS: VTK rendering factories are registered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
