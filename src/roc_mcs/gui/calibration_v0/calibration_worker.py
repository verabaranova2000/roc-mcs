# calibration_worker.py

from PyQt6.QtCore import QThread, pyqtSignal
from roc_mcs.processing.calibration import run_calibration

class CalibrationWorker(QThread):
    """
    Фоновый поток для запуска калибровки без блокировки GUI.

    Worker — это «посредник» между интерфейсом и тяжёлым расчётом:
    
        CalibrationWidget                               < --- знает про GUI и подключает сигналы к своим методам
              │ запускает
              ▼
        CalibrationWorker                               < --- знает про Qt, но ничего не знает про внешний вид GUI
              │ run()
              ▼
        run_calibration()                                < --- ничего не знает про Qt
              │
              ├── log_signal       → сообщения в GUI
              ├── progress_signal  → прогресс в GUI
              ├── finished_signal  → результат и график
              └── error_signal     → ошибка в GUI

    Почему нужен Worker:
        run_calibration() выполняет длительные вычисления.
        Если запустить их прямо из GUI, интерфейс «зависнет».
        QThread выполняет run() в отдельном потоке, поэтому GUI остаётся отзывчивым.

    Сигналы:
        log_signal(str)
            Передаёт текстовые сообщения в интерфейс.

        progress_signal(str, int, int)
            Передаёт статус, текущий шаг и общее число шагов.

        finished_signal(object, object)
            Сигнал успешного завершения:
            (result, fig).

        error_signal(str)
            Передаёт текст ошибки.

    Важно:
        Worker не рисует интерфейс и не управляет виджетами напрямую.
        Он только выполняет расчёт и сообщает GUI о происходящем через сигналы.
    """

    # Сигналы для связи фонового потока с GUI
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str, int, int)     # Сигнал принимает (строку, текущий шаг, всего шагов)
    finished_signal = pyqtSignal(object, object)    # result, fig
    error_signal = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        """Запускает калибровку в фоновом потоке и передаёт события в GUI."""
        try:
            # Преобразуем (через лямбды) callback-и расчёта в Qt-сигналы
            self.kwargs['log_callback'] = lambda msg: self.log_signal.emit(str(msg))
            self.kwargs['progress_callback'] = lambda msg, cur, tot: self.progress_signal.emit(str(msg), int(cur), int(tot))  # Лямбда принимает 3 аргумента и пробрасывает их в сигнал
            result, fig = run_calibration(**self.kwargs)
            self.finished_signal.emit(result, fig)                 # Расчёт успешно завершён
        except Exception as e:
            self.error_signal.emit(str(e))                         # Расчёт завершился ошибкой.
