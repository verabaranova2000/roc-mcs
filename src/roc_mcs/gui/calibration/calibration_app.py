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