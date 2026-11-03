"""Regression tests for QVTK startup without requiring a live Qt/OpenGL server.

The fake widget deliberately reports only one pre-initialization paint event.
That is a valid Qt lifecycle and must not prevent QVTK Initialize().
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from roc_mcs.gui.roi_explorer.vtk_backend import desktop_viewer


class _Signal:
    def connect(self, *_args, **_kwargs):
        pass


class _QObject:
    pass


class _QMainWindow:
    pass


class _DummyQVTKBase:
    pass


class _FakeInteractor:
    def __init__(self, initialized: bool = False):
        self.initialized = initialized

    def GetInitialized(self):
        return int(self.initialized)

    def GetEnabled(self):
        return int(self.initialized)

    def GetClassName(self):
        return "vtkGenericRenderWindowInteractor"


class _FakeRenderWindow:
    def __init__(self, interactor):
        self._interactor = interactor
        self.multisamples = 4

    def SetMultiSamples(self, value):
        self.multisamples = int(value)

    def GetMultiSamples(self):
        return self.multisamples

    def GetInteractor(self):
        return self._interactor

    def GetClassName(self):
        return "vtkWin32OpenGLRenderWindow"

    def IsA(self, name):
        return int(name == "vtkOpenGLRenderWindow")


class _FakeWidget:
    def __init__(self, interactor):
        self._interactor = interactor
        self._render_window = _FakeRenderWindow(interactor)
        self.paint_count = 1  # This is the exact lifecycle that previously failed.
        self.paint_stage = "pre-init"
        self._preinit_refresh_requested = False
        self._preinit_refresh_requested = False
        self.initialize_calls = 0
        self.repaint_calls = 0

    def testAttribute(self, _attribute):
        return True

    def isVisible(self):
        return True

    def width(self):
        return 1007

    def height(self):
        return 860

    def GetRenderWindow(self):
        return self._render_window

    def Initialize(self):
        self.initialize_calls += 1
        self._interactor.initialized = True

    def Render(self):
        pass

    def repaint(self):
        self.repaint_calls += 1
        self.paint_count += 1


class _FakeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _FakeLabel:
    def __init__(self):
        self.text = None
        self.hidden = False

    def setText(self, text):
        self.text = str(text)

    def hide(self):
        self.hidden = True


class _FakeTimer:
    calls = []

    @classmethod
    def singleShot(cls, delay, callback):
        cls.calls.append((delay, callback))


class TestQVTKStartupLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qt_core = SimpleNamespace(
            QObject=_QObject,
            pyqtSignal=lambda *_args, **_kwargs: _Signal(),
            pyqtSlot=lambda *_args, **_kwargs: (lambda function: function),
            QTimer=_FakeTimer,
            Qt=SimpleNamespace(
                WidgetAttribute=SimpleNamespace(WA_WState_Created=object())
            ),
        )
        qt_widgets = SimpleNamespace(QMainWindow=_QMainWindow)
        cls.window_class = desktop_viewer._create_window_class(
            qt_core, qt_widgets, _DummyQVTKBase
        )

    def _make_window(self, interactor_initialized=False):
        interactor = _FakeInteractor(interactor_initialized)
        widget = _FakeWidget(interactor)
        window = object.__new__(self.window_class)
        window.scene = None
        window._vtk_init_pending = True
        window._vtk_init_attempts = 0
        window.vtk_widget = widget
        window.render_window = widget.GetRenderWindow()
        window.interactor = None
        window.loader = None
        window.payload = None
        window._auto_load_pending = False
        window._busy = False
        window._refresh_controls = lambda: None
        window.open_button = _FakeButton()
        window.load_button = _FakeButton()
        window.apply_display_button = _FakeButton()
        window.placeholder_label = _FakeLabel()
        window.status_label = _FakeLabel()
        window.isVisible = lambda: True
        window._show_error = lambda title, message: self.fail(
            f"Unexpected startup error: {title}: {message}"
        )
        return window, interactor, widget

    def setUp(self):
        _FakeTimer.calls.clear()

    def test_one_preinit_paint_does_not_block_initialize_or_scene_creation(self):
        window, interactor, widget = self._make_window()

        class FakeScene:
            def __init__(self, render_window, actual_interactor, render_callback=None):
                self.render_window = render_window
                self.interactor = actual_interactor
                self.render_callback = render_callback

        with patch.object(desktop_viewer, "VTKVolumeScene", FakeScene), patch.object(
            desktop_viewer, "_trace", lambda *_args, **_kwargs: None
        ), patch.object(desktop_viewer, "_trace_resources", lambda *_args, **_kwargs: None):
            self.window_class._initialize_vtk_widget(window)

        # The best-effort repaint is synchronous, but initialization does not
        # depend on it delivering another paint event.
        self.assertEqual(widget.paint_count, 2)
        self.assertEqual(widget.repaint_calls, 1)
        self.assertEqual(widget.initialize_calls, 1)
        self.assertEqual(interactor.GetInitialized(), 1)
        self.assertEqual(widget.GetRenderWindow().GetMultiSamples(), 0)
        self.assertIsInstance(window.scene, FakeScene)
        self.assertFalse(window._vtk_init_pending)
        self.assertFalse(_FakeTimer.calls)
        self.assertEqual(window.status_label.text, "Выберите HDF5-файл для загрузки объёма.")

    def test_already_initialized_interactor_is_not_initialized_twice(self):
        window, interactor, widget = self._make_window(interactor_initialized=True)

        class FakeScene:
            def __init__(self, *_args, **_kwargs):
                pass

        with patch.object(desktop_viewer, "VTKVolumeScene", FakeScene), patch.object(
            desktop_viewer, "_trace", lambda *_args, **_kwargs: None
        ), patch.object(desktop_viewer, "_trace_resources", lambda *_args, **_kwargs: None):
            self.window_class._initialize_vtk_widget(window)

        self.assertEqual(widget.initialize_calls, 0)
        self.assertEqual(interactor.GetInitialized(), 1)
        self.assertIsInstance(window.scene, FakeScene)


if __name__ == "__main__":
    unittest.main(verbosity=2)
