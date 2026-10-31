"""Безопасная ранняя инициализация Qt WebEngine для всех точек входа ROC-MCS.

Вызовите :func:`prepare_qt_webengine` в основном процессе до создания
QCoreApplication/QApplication и до импорта модулей, создающих GUI.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


def _log_file() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
    folder = base / "roc_mcs" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "qt_webengine_bootstrap.log"


def prepare_qt_webengine(component: str = "ROC-MCS") -> str | None:
    """Подготавливает Qt WebEngine в текущем GUI-процессе.

    Возвращает None при успехе или диагностическое сообщение при ошибке.
    Функция намеренно не создаёт QApplication и не скрывает исключения:
    подробный traceback записывается в startup log, краткая причина
    возвращается вызывающему коду для отображения в 3D-панели.
    """
    stage = "import PyQt6.QtCore"
    log_path = None
    try:
        from PyQt6 import QtCore

        stage = "inspect QCoreApplication state"
        attribute = QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
        app_instance = QtCore.QCoreApplication.instance()
        webengine_already_imported = "PyQt6.QtWebEngineWidgets" in sys.modules
        shared_before = bool(QtCore.QCoreApplication.testAttribute(attribute))

        if app_instance is None:
            # Qt требует выставить этот атрибут до создания QApplication.
            if not shared_before:
                QtCore.QCoreApplication.setAttribute(attribute, True)
            if not QtCore.QCoreApplication.testAttribute(attribute):
                raise RuntimeError(
                    "Qt не подтвердил AA_ShareOpenGLContexts после setAttribute(True)."
                )
        elif not shared_before and not webengine_already_imported:
            # После создания QCoreApplication поздно включать атрибут.
            # Продолжать и делать вид, что 3D запустится, нельзя.
            raise RuntimeError(
                "QCoreApplication уже создан до импорта QtWebEngineWidgets, "
                "а AA_ShareOpenGLContexts выключен. Этот порядок нельзя "
                "исправить задним числом: вызовите prepare_qt_webengine() "
                "до QApplication в самой ранней точке входа. "
                f"Существующий объект: {type(app_instance).__name__}."
            )

        stage = "import PyQt6.QtWebEngineWidgets / QtWebEngineCore / QtWebChannel"
        from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings  # noqa: F401
        from PyQt6.QtWebChannel import QWebChannel  # noqa: F401

        if app_instance is None:
            shared_after = bool(QtCore.QCoreApplication.testAttribute(attribute))
            if not shared_after:
                raise RuntimeError("После импорта WebEngine атрибут AA_ShareOpenGLContexts неожиданно выключен.")
        print(
            f"[{component}] Qt WebEngine bootstrap: OK; "
            f"Python={sys.executable}; "
            f"QCoreApplication existed before import={app_instance is not None}; "
            f"AA_ShareOpenGLContexts={bool(QtCore.QCoreApplication.testAttribute(attribute))}; "
            f"QtWebEngineWidgets={sys.modules['PyQt6.QtWebEngineWidgets'].__file__ if hasattr(sys.modules['PyQt6.QtWebEngineWidgets'], '__file__') else 'loaded'}",
            flush=True,
        )
        return None

    except Exception as exc:
        details = (
            f"[{component}] Qt WebEngine bootstrap FAILED at stage: {stage}\n"
            f"Python: {sys.executable}\n"
            f"Exception: {type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
        )
        try:
            log_path = _log_file()
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n[{datetime.now().isoformat(timespec='seconds')}]\n{details}\n")
        except Exception:
            log_path = None
        message = f"{type(exc).__name__}: {exc} (этап: {stage})"
        if log_path is not None:
            message += f" Полный журнал: {log_path}"
        print(f"[{component}] Qt WebEngine bootstrap FAILED: {message}", file=sys.stderr, flush=True)
        return message
