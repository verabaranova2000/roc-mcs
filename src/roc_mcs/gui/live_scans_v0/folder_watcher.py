"""Наблюдение за появлением новой папки эксперимента."""
from __future__ import annotations

from pathlib import Path
from PyQt6 import QtCore


class BaseFolderWatcher(QtCore.QObject):
    """Обнаруживает новый дочерний каталог; существующие папки игнорируются."""
    sigNewExperimentFound = QtCore.pyqtSignal(str)

    def __init__(self, base_folder: str | Path, parent=None, poll_interval_ms: int = 500):
        super().__init__(parent)
        self.base_folder = Path(base_folder).expanduser().resolve()
        if not self.base_folder.is_dir():
            raise NotADirectoryError(f"Каталог наблюдения не существует: {self.base_folder}")
        self._known_dirs = {p.resolve() for p in self.base_folder.iterdir() if p.is_dir()}
        self.watcher = QtCore.QFileSystemWatcher(self)
        if not self.watcher.addPath(str(self.base_folder)):
            # Polling остаётся резервным механизмом, даже если watcher не смог подписаться.
            pass
        self.watcher.directoryChanged.connect(self._on_base_dir_changed)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(max(100, int(poll_interval_ms)))
        self._poll_timer.timeout.connect(self._check_for_new_dirs)
        self._poll_timer.start()
        self._stopped = False

    @QtCore.pyqtSlot(str)
    def _on_base_dir_changed(self, _path: str) -> None:
        self._check_for_new_dirs()

    @QtCore.pyqtSlot()
    def _check_for_new_dirs(self) -> None:
        if self._stopped:
            return
        try:
            current = {p.resolve() for p in self.base_folder.iterdir() if p.is_dir()}
        except OSError:
            return
        new_dirs = current - self._known_dirs
        if not new_dirs:
            return
        self._known_dirs.update(new_dirs)
        def created_key(path: Path) -> tuple[int, str]:
            try:
                return path.stat().st_ctime_ns, path.name
            except OSError:
                return 0, path.name
        self.sigNewExperimentFound.emit(str(max(new_dirs, key=created_key)))

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._poll_timer.stop()
        paths = self.watcher.directories()
        if paths:
            self.watcher.removePaths(paths)
        try:
            self.watcher.directoryChanged.disconnect(self._on_base_dir_changed)
        except (TypeError, RuntimeError):
            pass
