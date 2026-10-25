import time
from pathlib import Path
from PyQt6 import QtCore
import numpy as np

# Импорт твоих готовых функций
from roc_mcs.io.mcs import load_mcs
from roc_mcs.processing.alignment import find_phase, extract_branch
from roc_mcs.gui.live_scans.folder_watcher import BaseFolderWatcher  # backward-compatible import

from pathlib import Path
from PyQt6 import QtCore

class ExperimentalMcsStreamer(QtCore.QObject):
    """
    Фоновый провайдер данных. Фазу определяет по первому MCS текущей серии и фиксирует для следующих сканов.
    """
    # Сигнал передает словарь с распарсенными данными для одного среза
    sigRecordReady = QtCore.pyqtSignal(dict)
    
    def __init__(self, folder: str, branch: str = "up", parent=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self.branch = branch

        self.processed_files = set()
        # Фаза определяется заново по первому MCS каждой live-серии и фиксируется до её конца.
        self.global_phi = None
        self.phase_source_file = None
        self.start_time = None
        
        # Очередь для файлов, которые появились, но еще не дописаны до конца
        # ИСПРАВЛЕНИЕ 1: Используем list вместо set, чтобы сохранять строгий порядок
        self._pending_files = []
        
        self._retry_timer = QtCore.QTimer(self)
        self._retry_timer.setInterval(250) # Проверка каждые 250 мс
        self._retry_timer.timeout.connect(self._process_pending_files)
        
        self.watcher = QtCore.QFileSystemWatcher()
        self.watcher.addPath(str(self.folder))
        self.watcher.directoryChanged.connect(self._on_directory_changed)
        # Резервный опрос: QFileSystemWatcher может пропускать события на Windows.
        self._scan_timer = QtCore.QTimer(self)
        self._scan_timer.setInterval(500)
        self._scan_timer.timeout.connect(lambda: self._on_directory_changed(str(self.folder)))
        self._scan_timer.start()
        # При запуске проверяем уже существующие файлы.
        self._on_directory_changed(str(self.folder))

    def _on_directory_changed(self, path: str):
        """ Срабатывает, когда в папке появляются новые файлы """
        current_files = set(Path(path).glob("*.mcs"))
        # Находим новые файлы, которых еще нет ни в обработанных, ни в ожидающих
        new_files = current_files - self.processed_files - set(self._pending_files)        
        # new_files = current_files - self.processed_files - self._pending_files
        
        if new_files:
            # ИСПРАВЛЕНИЕ 2: Сортируем строго по имени файла (001.mcs, 002.mcs...)
            sorted_new = sorted(list(new_files), key=lambda x: x.name)            
            # Сортируем новые файлы по времени модификации или имени, чтобы соблюсти порядок
            # sorted_new = sorted(list(new_files), key=lambda x: x.stat().st_mtime)
            for file_path in sorted_new:
                if file_path not in self._pending_files:
                    self._pending_files.append(file_path)
            
            if not self._retry_timer.isActive():
                self._retry_timer.start()

    def _process_pending_files(self):
        """ Попытка прочитать незавершенные файлы """
        if not self._pending_files:
            self._retry_timer.stop()
            return
            
        # Копируем сет, чтобы можно было удалять успешные файлы во время итерации
        # Итерируемся по копии списка
        for file_path in list(self._pending_files):  
            try:
                # Пытаемся прочитать файл моей функцией
                mcs_data = load_mcs(file_path)
                
                # Если файл успешно прочитан и распарсен, обрабатываем его
                self._process_valid_mcs(mcs_data)
                
                # Убираем из очереди ожидания и добавляем в обработанные
                self._pending_files.remove(file_path)
                self.processed_files.add(file_path)
                
            except (ValueError, IndexError, OSError, EOFError):
                # Файл еще записывается детектором. Оставляем в очереди до следующего тика таймера.
                # continue
                # ИСПРАВЛЕНИЕ 3: Если файл (например, 002.mcs) недописан, 
                # мы ОБЯЗАНЫ прервать цикл (break). Мы не имеем права читать 003.mcs, 
                # пока не дочитаем 002.mcs, иначе нарушим последовательность времени на графике!
                break                

    def _process_valid_mcs(self, mcs_data: dict):
        """ Обработка успешно прочитанного файла с использованием твоей логики """
        counts = mcs_data["counts"]
        n_channels = mcs_data["n_channels"]
        file_time = mcs_data["header"]["file_info"]["datetime"]
        
        # Фаза серии определяется только её первым MCS; фаза калибровочного MCS здесь не используется.
        # Временной ноль также задаётся первым файлом текущей серии.
        if self.start_time is None:
            self.start_time = file_time
        if self.global_phi is None:
            phi, _ = find_phase(counts, n_channels)
            self.global_phi = float(phi)
            self.phase_source_file = str(mcs_data.get("file_name", ""))
            
        # Извлекаем нужную ветвь (ось Y и интенсивность Z)
        s_branch, I_branch = extract_branch(counts, self.global_phi, n_channels, branch=self.branch)
        
        # Рассчитываем время в секундах от начала эксперимента
        time_s = (file_time - self.start_time).total_seconds()
        
        # Формируем пакет данных для UI
        record_packet = {
            "file_name": mcs_data["file_name"],
            "datetime": file_time,
            "time_s": time_s,
            "s_axis": s_branch,
            "intensity": I_branch,
            "phase_phi": self.global_phi,
            "phase_source_file": self.phase_source_file,
            "raw_mcs": mcs_data # Сохраняем сырые данные, если понадобятся для UI (например, каналы)
        }
        
        # Отправляем пакет в графический интерфейс
        self.sigRecordReady.emit(record_packet)

    def stop(self):
            """ Корректная остановка всех процессов стримера """
            # 1. Останавливаем таймер проверки файлов
            if self._retry_timer.isActive():
                self._retry_timer.stop()
            if hasattr(self, "_scan_timer") and self._scan_timer.isActive():
                self._scan_timer.stop()
            
            # 2. Отключаем файловый наблюдатель
            if self.watcher:
                # Сначала удаляем все пути, за которыми следим
                paths = self.watcher.files() + self.watcher.directories()
                if paths:
                    self.watcher.removePaths(paths)
                # Отключаем сигнал, если он был подключен
                try:
                    self.watcher.directoryChanged.disconnect(self._on_directory_changed)
                except (TypeError, RuntimeError):
                    pass
                self.watcher = None        



from .live_reader import LiveCSVReader
import time

class CsvStreamer(QtCore.QThread):
    """
    Фоновый поток для чтения лога нагрузки. 
    Эмитит сигнал с новыми значениями времени, силы и давления.
    """
    # Сигнал: time_s, force_kg, pressure_mpa
    sigDataReady = QtCore.pyqtSignal(float, float, float)

    def __init__(self, folder: str, parent=None):
        super().__init__(parent)
        self.folder = Path(folder)
        self._is_running = True

    def run(self):
        # 1. Ждем появления лог-файла в папке
        log_file = None
        patterns = ("periodic_control_log_*.csv", "cycle_control_log_*.csv", "control_log_*.csv")
        while self._is_running and not log_file:
            files = [p for pattern in patterns for p in self.folder.glob(pattern)]
            if files:
                # В новой папке ожидается один журнал управления; выбираем самый свежий.
                log_file = max(files, key=lambda p: p.stat().st_mtime)
            else:
                time.sleep(0.25)
                
        if not self._is_running or not log_file:
            return

        # 2. Запускаем наш надежный reader
        reader = LiveCSVReader(str(log_file))
        for row in reader.follow():
            if not self._is_running:
                break
                
            if row is None:
                time.sleep(0.05)
                continue
                
            try:
                t = float(row["elapsed"])
                f = float(row["force"])
                p = float(row["pressure_mpa"])
                self.sigDataReady.emit(t, f, p)
            except (ValueError, KeyError):
                pass

    def stop(self):
        self._is_running = False
        self.wait() # Ждем корректного завершения потока                