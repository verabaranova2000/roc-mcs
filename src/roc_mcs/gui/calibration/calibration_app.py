import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

from roc_mcs.gui.calibration.calibration_widget import CalibrationWidget


APP_TITLE = "Утилита MCS-калибровки v1.0"


def create_application():
    """Создаёт QApplication и главное окно приложения."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    calibration_widget = CalibrationWidget()

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