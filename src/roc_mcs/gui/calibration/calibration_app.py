import os
import sys
from pathlib import Path


# ============================================================
# Автоматическая настройка окружения
# ============================================================

def _find_project_root() -> Path:
    """
    Возвращает корень репозитория roc-mcs.

    Структура:
        roc-mcs/
        └── src/
            └── roc_mcs/
                └── gui/
                    └── calibration/
                        └── calibration_app.py
    """
    return Path(__file__).resolve().parents[4]


def _find_virtualenv(project_root: Path) -> Path | None:
    """
    Ищет виртуальное окружение в корне проекта.

    Поддерживаются:
        .venv
        venv
    """

    for name in (".venv", "venv"):
        env_dir = project_root / name

        if sys.platform == "win32":
            python_exe = env_dir / "Scripts" / "python.exe"
        else:
            python_exe = env_dir / "bin" / "python"

        if python_exe.is_file():
            return python_exe

    return None


def _is_same_python(python_a: Path, python_b: Path) -> bool:
    """Проверяет, являются ли два пути одним и тем же Python."""

    try:
        return python_a.resolve() == python_b.resolve()
    except OSError:
        return os.path.normcase(str(python_a)) == os.path.normcase(str(python_b))


def _restart_in_virtualenv() -> None:
    """
    Если в проекте есть virtual environment и приложение
    запущено не из него — перезапускает приложение через
    Python из virtual environment.
    """

    project_root = _find_project_root()

    # --------------------------------------------------------
    # Делаем src доступным для импорта roc_mcs
    # --------------------------------------------------------

    src_dir = project_root / "src"

    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # --------------------------------------------------------
    # Ищем virtual environment
    # --------------------------------------------------------

    venv_python = _find_virtualenv(project_root)

    # Если virtual environment нет — ничего не меняем.
    if venv_python is None:
        os.chdir(project_root)
        return

    current_python = Path(sys.executable)

    # Мы уже работаем из нужного окружения.
    if _is_same_python(current_python, venv_python):
        os.chdir(project_root)
        return

    # --------------------------------------------------------
    # Нашли virtual environment, но сейчас используется
    # другой Python → перезапускаем приложение через него.
    # --------------------------------------------------------

    script_path = Path(__file__).resolve()

    if sys.platform == "win32":
        pythonw = venv_python.with_name("pythonw.exe")

        if pythonw.is_file():
            executable = pythonw
        else:
            executable = venv_python
    else:
        executable = venv_python

    # Работаем относительно корня проекта.
    os.chdir(project_root)

    # Передаём управление Python из virtual environment.
    os.execv(
        str(executable),
        [str(executable), str(script_path), *sys.argv[1:]],
    )


# ============================================================
# Запускаем bootstrap ДО импортов приложения
# ============================================================

_restart_in_virtualenv()


# ============================================================
# Обычный код приложения
# ============================================================


import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow

from roc_mcs.gui.calibration.calibration_widget import CalibrationWidget
from roc_mcs.io.sardana_h5 import  H51DEntryProvider


APP_TITLE = "Утилита MCS-калибровки v1.0"
TPC_DATA_H5_PATH = Path(r"C:\Users\User\Desktop\For_Sardana\TPC_data.h5")

def create_application():
    """Создаёт QApplication и главное окно приложения."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    entry_provider = None

    if TPC_DATA_H5_PATH.is_file():                             # если по этому пути существует именно файл
        entry_provider = H51DEntryProvider(TPC_DATA_H5_PATH)

    calibration_widget = CalibrationWidget(entry_provider=entry_provider)

    window = QMainWindow()
    window.setWindowTitle(APP_TITLE)
    window.setCentralWidget(calibration_widget)

    window.setMinimumSize(calibration_widget.minimumSizeHint())
    window.adjustSize()

    return app, window


def main():
    """Запускает приложение калибровки."""
    app, window = create_application()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())