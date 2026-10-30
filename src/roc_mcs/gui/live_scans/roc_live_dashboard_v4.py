"""Оперативный дашборд РОК-карты и потоковая обработка папки опыта.

- При запуске без записей окно позволяет выбрать базовую папку и ждёт новую папку эксперимента.
- После обнаружения папки читает новые .mcs-файлы и CSV управления по мере записи.
- Встроенный синтетический режим доступен отдельно для проверки визуализации.

Зависимости: PyQt6, pyqtgraph, numpy, pandas и проектный пакет roc_mcs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import tempfile
import shutil

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .streamer import ExperimentalMcsStreamer, CsvStreamer
from roc_mcs.gui.live_scans.folder_watcher import BaseFolderWatcher
from roc_mcs.gui.components.collapsible_section import CollapsibleSection
from roc_mcs.gui.components.icon_rail import IconRail
from roc_mcs.gui.components.icon_provider import resource_svg_icon, set_icon_theme
from roc_mcs.gui.app_settings import load_settings, save_settings, resolve_setting_path
from .demo import make_demo_param_table, make_synthetic_mcs_records
from .axis_calibration import load_calibration, save_calibration, effective_scale
from .calibration_dialog import CalibrationDialog
from .dashboard_widgets import RocMapWidget, FilmstripWidget, MetricsPlotWidget, LiveMcsAdapter
from .press_panel import PressControlPanel
from .experiment_config import ExperimentConfigWidget
from .export_dialog import ExportOptionsDialog
from .export_worker import PipelineExportWorker
from .export_contract import build_pipeline_calibration_args

WATCH_STOP_ICON = "watch_stop.svg"  # Выбранный файл; варианты watch_stop_1..3.svg в ресурсах.


# -----------------------------------------------------------------------------
# Main dashboard (СТРОГИЙ КОНТРОЛЛЕР С ПОДДЕРЖКОЙ LIVE FOLDER)
# -----------------------------------------------------------------------------

class LiveRocDashboard(QtWidgets.QMainWindow):
    def __init__(self, records: list | None = None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Оперативный анализ РОК-кривых")
        pg.setConfigOptions(antialias=True)

        # --- АРХИТЕКТУРНОЕ РАЗДЕЛЕНИЕ РЕЖИМОВ ---
        self.is_synthetic = records is not None
        self.records = records or []
        
        self._current_records = []
        
        # ЕДИНСТВЕННЫЙ ИСТОЧНИК ПРАВДЫ (State)
        self.pinned_index = 0
        self.hover_state = None  # None или dict {'idx': int, 'x': float}
        self.follow_latest = True
        
        self.time_s = np.array([], dtype=float)
        self.roc_map_live = {}

        # Массивы для данных нагрузки
        self.load_t = []
        self.load_f = []
        self.load_p = []
        self.load_metric = "force"
        
        # Объекты контроллеров потока данных
        self.streamer = None  # Для реального режима
        self.base_watcher = None  # Координатор уровня 1
        self.csv_streamer = None  # Переменная для стримера нагрузки   
        self._timer = QtCore.QTimer(self) # Для синтетики
        self._timer.setInterval(850)
        self._timer.timeout.connect(self._append_synthetic_record)
        self._stream_index = 0
        self.calibration_path = self._user_data_dir() / "axis_calibration.json"
        legacy_calibration = Path(__file__).resolve().parent / "config" / "axis_calibration.json"
        if not self.calibration_path.is_file() and legacy_calibration.is_file():
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            try: shutil.copy2(legacy_calibration, self.calibration_path)
            except OSError: pass
        self.axis_calibration = None if self.is_synthetic else load_calibration(self.calibration_path)
        self._sidebar_expanded = True
        self._sidebar_width = 250
        self._active_sidebar_section = "sample"
        self._calibration_dialog = None
        configured_experiments_dir = None if self.is_synthetic else resolve_setting_path("experiments_dir", require_exists=True)
        self._base_folder_path: Path | None = configured_experiments_dir if configured_experiments_dir is not None and configured_experiments_dir.is_dir() else None
        self._active_experiment_folder: Path | None = None
        self._export_worker: PipelineExportWorker | None = None
        self._run_experiment_amplitude_mVpp = 0.0
        
        app = QtWidgets.QApplication.instance()
        self._build_ui()
        if self._base_folder_path is not None and hasattr(self, "edit_base_folder"):
            self.edit_base_folder.setText(str(self._base_folder_path)); self.edit_base_folder.setToolTip(str(self._base_folder_path))
        self._apply_theme()
        self._update_axis_ui()
        
        # Инициализация старта в зависимости от режима
        if self.is_synthetic:
            self._reset_synthetic_stream()
        elif self._base_folder_path is not None:
            self.lbl_status.setText("Сохранённый каталог указан; нажмите кнопку папки, чтобы начать ожидание")
        else:
            self.lbl_status.setText("Выберите базовую папку эксперимента")

        # Явно ограничиваем начальную геометрию доступной областью экрана.
        # Без этого QSplitter мог запросить слишком большое окно на Windows с масштабированием DPI.
        self._fit_initial_window()
        self._update_export_button()

    @staticmethod
    def _user_data_dir() -> Path:
        """Каталог настроек пользователя; пакет roc_mcs может быть только для чтения."""
        root = Path(os.environ.get("LOCALAPPDATA", Path.home())) if os.name == "nt" else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return root / "roc_mcs" / "live_scans"

    def _fit_initial_window(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1000, 700)
            return
        area = screen.availableGeometry()
        width = max(600, min(1180, area.width() - 28))
        height = max(420, min(820, area.height() - 40))
        self.setMinimumSize(min(760, width), min(480, height))
        self.resize(width, height)
        side_width = min(max(self._sidebar_width, 260), max(180, width // 3))
        self.body_splitter.setSizes([side_width, max(250, width - side_width - 44)])

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self); central.setObjectName("appSurface"); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Встроенная верхняя полоса без отдельной светлой плашки.
        toolbar_box = QtWidgets.QWidget(central); toolbar_box.setObjectName("topToolbar")
        toolbar = QtWidgets.QHBoxLayout(toolbar_box); toolbar.setContentsMargins(6, 5, 7, 5); toolbar.setSpacing(5)
        self.btn_sidebar = QtWidgets.QToolButton(); self.btn_sidebar.setObjectName("toolbarIconButton")
        self.btn_sidebar.setFixedSize(32, 30); self.btn_sidebar.setIconSize(QtCore.QSize(18, 18))
        self.btn_sidebar.setToolTip("Скрыть панель раздела"); self.btn_sidebar.setAccessibleName("Свернуть или развернуть панель")
        self.btn_sidebar.setIcon(self._pane_icon(True)); self.btn_sidebar.clicked.connect(self._toggle_sidebar); toolbar.addWidget(self.btn_sidebar)

        self.btn_open = QtWidgets.QToolButton(); self.btn_open.setObjectName("toolbarIconButton")
        self.btn_open.setFixedSize(32, 30); self.btn_open.setIconSize(QtCore.QSize(18, 18))
        self.btn_open.setIcon(resource_svg_icon("folder_open.svg", widget=self.btn_open, size=18))
        self.btn_open.setToolTip("Выбрать базовый каталог и ждать новую папку эксперимента")
        self.btn_open.setAccessibleName("Выбрать базовый каталог"); self.btn_open.clicked.connect(self._start_watching_folder); toolbar.addWidget(self.btn_open)

        self.btn_stop = QtWidgets.QToolButton(); self.btn_stop.setObjectName("toolbarIconButton")
        self.btn_stop.setFixedSize(32, 30); self.btn_stop.setIconSize(QtCore.QSize(18, 18))
        self.btn_stop.setIcon(resource_svg_icon(WATCH_STOP_ICON, widget=self.btn_stop, size=18))
        self.btn_stop.setToolTip("Остановить наблюдение и чтение файлов текущей серии")
        self.btn_stop.setAccessibleName("Остановить наблюдение"); self.btn_stop.clicked.connect(self._stop_watching); self.btn_stop.setEnabled(False); toolbar.addWidget(self.btn_stop)

        self.edit_base_folder = QtWidgets.QLineEdit(); self.edit_base_folder.setObjectName("baseFolderField")
        self.edit_base_folder.setReadOnly(True); self.edit_base_folder.setPlaceholderText("Базовый каталог не выбран")
        self.edit_base_folder.setToolTip("Каталог, в котором ожидается создание новой папки эксперимента")
        self.edit_base_folder.setMinimumWidth(120); self.edit_base_folder.setMaximumWidth(240); self.edit_base_folder.setFixedHeight(28)
        toolbar.addWidget(self.edit_base_folder)
        self.btn_export = QtWidgets.QToolButton(); self.btn_export.setObjectName("toolbarIconButton")
        self.btn_export.setFixedSize(32, 30); self.btn_export.setIconSize(QtCore.QSize(18, 18))
        self.btn_export.setIcon(resource_svg_icon("save_icon.svg", widget=self.btn_export, size=18))
        self.btn_export.setToolTip("Экспортировать завершённый эксперимент через pipeline/build.py")
        self.btn_export.setAccessibleName("Экспорт результатов"); self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._open_export_dialog); toolbar.addWidget(self.btn_export)

        divider = QtWidgets.QFrame(); divider.setObjectName("toolbarDivider"); divider.setFrameShape(QtWidgets.QFrame.Shape.VLine); divider.setFixedHeight(22); toolbar.addWidget(divider)
        self.chk_follow = QtWidgets.QCheckBox("Следовать"); self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("Автоматически выбирать последний полученный срез")
        self.chk_follow.toggled.connect(self._set_follow_latest); toolbar.addWidget(self.chk_follow)
        self.cmb_load_metric = QtWidgets.QComboBox(); self.cmb_load_metric.addItem("Сила, кг", "force"); self.cmb_load_metric.addItem("Давление, МПа", "pressure")
        self.cmb_load_metric.currentIndexChanged.connect(self._on_load_metric_changed); self.cmb_load_metric.setMinimumWidth(105); self.cmb_load_metric.setMaximumWidth(125); toolbar.addWidget(self.cmb_load_metric)
        self.btn_auto_levels = QtWidgets.QPushButton("Уровни I"); self.btn_auto_levels.setObjectName("autoLevelsButton")
        self.btn_auto_levels.setToolTip("Пересчитать пределы интенсивности и зафиксировать их до следующего нажатия")
        self.btn_auto_levels.clicked.connect(lambda: self.map_plot.auto_levels()); toolbar.addWidget(self.btn_auto_levels)
        self.color_scale = QtWidgets.QFrame(); self.color_scale.setFixedSize(62, 10); self.color_scale.setToolTip("Шкала интенсивности: Inferno")
        self.color_scale.setStyleSheet("QFrame { border:1px solid #536173; border-radius:2px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #000004, stop:0.25 #57106e, stop:0.5 #bc3754, stop:0.75 #f98e09, stop:1 #fcffa4); }"); toolbar.addWidget(self.color_scale)
        self.lbl_color_levels = QtWidgets.QLabel("I: —"); self.lbl_color_levels.setObjectName("statusText"); self.lbl_color_levels.setMinimumWidth(74)
        self.lbl_color_levels.setToolTip("Зафиксированные пределы шкалы интенсивности"); toolbar.addWidget(self.lbl_color_levels)
        toolbar.addStretch(1)
        self.lbl_status = QtWidgets.QLabel("Готов к работе"); self.lbl_status.setObjectName("statusText")
        self.lbl_status.setMinimumWidth(110); self.lbl_status.setMaximumWidth(230); toolbar.addWidget(self.lbl_status)
        root.addWidget(toolbar_box)

        body_row = QtWidgets.QHBoxLayout(); body_row.setContentsMargins(0, 0, 0, 0); body_row.setSpacing(0)
        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal); self.body_splitter.setChildrenCollapsible(False); self.body_splitter.setHandleWidth(4)
        self.sidebar = QtWidgets.QWidget(); self.sidebar.setObjectName("sidebar"); self.sidebar.setMinimumWidth(204); self.sidebar.setMaximumWidth(310)
        sidebar_outer = QtWidgets.QVBoxLayout(self.sidebar); sidebar_outer.setContentsMargins(0, 0, 0, 0); sidebar_outer.setSpacing(0)
        self.sidebar_scroll = QtWidgets.QScrollArea(); self.sidebar_scroll.setWidgetResizable(True); self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.sidebar_content = QtWidgets.QWidget(); side = QtWidgets.QVBoxLayout(self.sidebar_content); side.setContentsMargins(5, 5, 6, 5); side.setSpacing(5)
        self.sidebar_scroll.setWidget(self.sidebar_content); sidebar_outer.addWidget(self.sidebar_scroll)

        self.section_experiment = CollapsibleSection("ОБРАЗЕЦ И УСЛОВИЯ", expanded=True)
        self.experiment_config = ExperimentConfigWidget(); self.experiment_config.saveProfileRequested.connect(self._save_experiment_profile)
        self.experiment_config.loadProfileRequested.connect(self._load_experiment_profile); self.section_experiment.content_layout.addWidget(self.experiment_config)
        side.addWidget(self.section_experiment)

        self.section_calibration = CollapsibleSection("КАЛИБРОВКА ОСИ θ", expanded=False); cal = self.section_calibration.content_layout
        self.lbl_calibration = QtWidgets.QLabel(); self.lbl_calibration.setObjectName("calibrationStatus"); self.lbl_calibration.setWordWrap(True); cal.addWidget(self.lbl_calibration)
        self.btn_calibration = QtWidgets.QPushButton("Открыть калибровку…"); self.btn_calibration.clicked.connect(self._open_calibration_dialog); cal.addWidget(self.btn_calibration)
        amp_row = QtWidgets.QWidget(); amp_layout = QtWidgets.QHBoxLayout(amp_row); amp_layout.setContentsMargins(0, 0, 0, 0); amp_layout.setSpacing(4)
        amp_layout.addWidget(QtWidgets.QLabel("U серии")); self.spin_experiment_amplitude = QtWidgets.QDoubleSpinBox()
        self.spin_experiment_amplitude.setRange(0, 10000); self.spin_experiment_amplitude.setDecimals(2); self.spin_experiment_amplitude.setSingleStep(50)
        self.spin_experiment_amplitude.setSuffix(" mVpp"); self.spin_experiment_amplitude.setKeyboardTracking(False)
        self.spin_experiment_amplitude.setToolTip("Амплитуда генератора адаптивного элемента для текущей серии; не напряжение трубки или пресса.")
        amp_layout.addWidget(self.spin_experiment_amplitude, 1); cal.addWidget(amp_row)
        self.lbl_axis_scale_details = QtWidgets.QLabel(); self.lbl_axis_scale_details.setObjectName("axisScaleDetails"); self.lbl_axis_scale_details.setWordWrap(True); cal.addWidget(self.lbl_axis_scale_details)
        side.addWidget(self.section_calibration); self.spin_experiment_amplitude.valueChanged.connect(self._update_axis_ui)

        self.section_press = CollapsibleSection("ПРЕСС · МАКЕТ", expanded=False)
        self.press_panel = PressControlPanel(); self.section_press.content_layout.addWidget(self.press_panel); side.addWidget(self.section_press)

        self.section_analysis = CollapsibleSection("АНАЛИЗ ДАННЫХ", expanded=False)
        for caption in ("Маркеры и диапазоны", "Сегментация", "Аппроксимация", "Экспорт результатов"):
            button = QtWidgets.QPushButton(caption); button.setEnabled(False); button.setToolTip("Раздел находится в разработке"); self.section_analysis.content_layout.addWidget(button)
        side.addWidget(self.section_analysis); side.addStretch(1)
        note = QtWidgets.QLabel("Потоковая обработка · тёмная тема"); note.setObjectName("sideFootnote"); note.setWordWrap(True); side.addWidget(note)
        self.body_splitter.addWidget(self.sidebar)

        content = QtWidgets.QWidget(); content_layout = QtWidgets.QVBoxLayout(content); content_layout.setContentsMargins(0, 0, 0, 0); content_layout.setSpacing(4)
        self.visual_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical); self.visual_splitter.setChildrenCollapsible(False); self.visual_splitter.setHandleWidth(4); content_layout.addWidget(self.visual_splitter, 1)
        self.map_plot = RocMapWidget(); self.map_plot.sigHoverX.connect(self._on_map_hover); self.map_plot.sigClickedX.connect(self._on_map_click)
        self.map_plot.sigHoverLeft.connect(self._on_hover_left); self.map_plot.sigColorLevelsChanged.connect(self._update_color_levels_label); self.visual_splitter.addWidget(self.map_plot)
        self.filmstrip = FilmstripWidget(); self.filmstrip.sigIndexHovered.connect(self._on_strip_hover); self.filmstrip.sigIndexClicked.connect(self._on_strip_click)
        self.filmstrip.sigHoverLeft.connect(self._on_hover_left); self.visual_splitter.addWidget(self.filmstrip)
        self.metrics_plot = MetricsPlotWidget(); self.visual_splitter.addWidget(self.metrics_plot); self.visual_splitter.setSizes([440, 225, 220])
        self.body_splitter.addWidget(content); self.body_splitter.setStretchFactor(0, 0); self.body_splitter.setStretchFactor(1, 1)

        self.nav_rail = IconRail([
            ("sample", "sample_config.svg", "Образец и условия эксперимента", True),
            ("calibration", "theta_axis.svg", "Калибровка угловой оси θ", True),
            ("press", "press.svg", "Режимы и параметры пресса", True),
            ("analysis", "analysis.svg", "Анализ данных", True),
            ("archive", "archive.svg", "Архив экспериментов — следующий этап", False),
        ], parent=central)
        self.nav_rail.sectionSelected.connect(self._select_sidebar_section); self.nav_rail.set_current("sample")
        body_row.addWidget(self.nav_rail); body_row.addWidget(self.body_splitter, 1); root.addLayout(body_row, 1)
        self.body_splitter.setSizes([self._sidebar_width, max(250, self.width() - self._sidebar_width - 44)])
        self._update_axis_ui()

    def _toggle_sidebar(self) -> None:
        self._sidebar_expanded = not self._sidebar_expanded
        if not self._sidebar_expanded:
            sizes = self.body_splitter.sizes()
            if sizes and sizes[0] > 0: self._sidebar_width = sizes[0]
            self.sidebar.hide(); self.btn_sidebar.setIcon(self._pane_icon(False)); self.btn_sidebar.setToolTip("Показать панель раздела")
        else:
            self.sidebar.show(); self.sidebar.setMinimumWidth(204); self.btn_sidebar.setIcon(self._pane_icon(True)); self.btn_sidebar.setToolTip("Скрыть панель раздела")
            self.body_splitter.setSizes([self._sidebar_width, max(250, self.width() - self._sidebar_width - 44)])

    def _select_sidebar_section(self, key: str) -> None:
        """Показывает один контекстный раздел; навигация не запускает оборудование."""
        sections = {"sample": self.section_experiment, "calibration": self.section_calibration,
                    "press": self.section_press, "analysis": self.section_analysis}
        target = sections.get(key)
        if target is None: return
        if not self._sidebar_expanded: self._toggle_sidebar()
        for section in sections.values(): section.setExpanded(section is target)
        self._active_sidebar_section = key; self.nav_rail.set_current(key)
        QtCore.QTimer.singleShot(0, lambda w=target: self.sidebar_scroll.ensureWidgetVisible(w, 8, 8))

    def _pane_icon(self, expanded: bool) -> QtGui.QIcon:
        icon_name = "pane_expanded.svg" if expanded else "pane_collapsed.svg"
        return resource_svg_icon(icon_name, widget=self.btn_sidebar, size=18)

    @QtCore.pyqtSlot(float, float)
    def _update_color_levels_label(self, low: float, high: float) -> None:
        self.lbl_color_levels.setText(f"I: {low:.4g}…{high:.4g}")

    def _current_effective_scale(self):
        if not hasattr(self, "spin_experiment_amplitude") or not hasattr(self, "press_panel"): return None
        return effective_scale(self.axis_calibration, self.spin_experiment_amplitude.value())

    def _update_axis_ui(self, *_args) -> None:
        scale = self._current_effective_scale()
        calibrated = scale is not None
        unit = self.axis_calibration.get("unit", "угл. с") if calibrated and self.axis_calibration else "отн. ед."
        self.map_plot.setLabel("left", "Угол θ" if calibrated else "Параметр s", units=unit)
        if hasattr(self, "metrics_plot"): self.metrics_plot.set_axis_units(unit if calibrated else "отн. ед. s", calibrated)
        self.lbl_calibration.setToolTip(f"Файл параметров калибровки: {self.calibration_path}")
        if self.axis_calibration:
            cal = self.axis_calibration
            branch = {"up": "верхняя (up)", "down": "нижняя (down)"}.get(cal.get("branch"), cal.get("branch"))
            u_cal = cal.get("reference_amplitude_mVpp")
            if u_cal:
                self.lbl_calibration.setText(f"Калибровочная пара: Aк={cal['A']:.6g} угл. с; Bк={cal['B']:.6g} (только совмещение)\nUк={u_cal:g} mVpp; φк найдена автоматически.")
            else:
                self.lbl_calibration.setText(f"Старая калибровка Aк={cal['A']:.6g}; Bк={cal['B']:.6g}.\nАмплитуда Uк не сохранена — повторите калибровку.")
            if calibrated:
                self.lbl_calibration.setStyleSheet("color:#9bd6a6; padding:3px 0;")
                self.lbl_axis_scale_details.setStyleSheet("color:#9bd6a6; font-size:10px; padding:2px 0 4px 0;")
                self.lbl_axis_scale_details.setText(f"Серия: Aс=Aк·Uс/Uк={scale:.7g} угл. с; Uс={self.spin_experiment_amplitude.value():g} mVpp.\nВетвь: {branch}. Фаза φс определяется заново по первому MCS серии.")
            else:
                self.lbl_calibration.setStyleSheet("color:#e3bd7a; padding:3px 0;")
                self.lbl_axis_scale_details.setStyleSheet("color:#e3bd7a; font-size:10px; padding:2px 0 4px 0;")
                if not u_cal: reason = "В калибровке нет Uк. Повторите калибровку с известной амплитудой генератора (mVpp)."
                elif self.spin_experiment_amplitude.value() <= 0: reason = "Введите Uс — амплитуду генератора для текущей серии (mVpp)."
                else: reason = "Не удалось применить угловой масштаб; проверьте Aк и Uк в JSON калибровки."
                self.lbl_axis_scale_details.setText(reason + " Сейчас отображается параметр s.")
        else:
            self.lbl_calibration.setText("Калибровка не задана. До получения валидной шкалы ось отображается как s.")
            self.lbl_calibration.setStyleSheet("color:#e3bd7a; padding:3px 0;")
            self.lbl_axis_scale_details.setText("Сначала откройте калибровку, укажите Uк; перед серией укажите Uс. Фаза каждой MCS-серии определяется автоматически.")
        if hasattr(self, "roc_map_live") and self._current_records and self.streamer is None and not self.is_synthetic:
            amp = self.spin_experiment_amplitude.value()
            for record in self._current_records:
                if isinstance(record, LiveMcsAdapter): record.set_calibration(self.axis_calibration, experiment_amplitude_mVpp=amp)
            self._rebuild_current_map()

    def _open_calibration_dialog(self) -> None:
        self._calibration_dialog = CalibrationDialog(self, self.axis_calibration)
        self._calibration_dialog.calibration_saved.connect(self._on_calibration_saved)
        self._calibration_dialog.show()
        self._calibration_dialog.raise_(); self._calibration_dialog.activateWindow()

    def _collect_experiment_config(self) -> dict:
        calibration = None
        if self.axis_calibration:
            calibration = dict(self.axis_calibration)
        return {"schema_version": 1, "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "application": {"name": "Live ROC Dashboard", "version": "4.3.0"},
                "sample": self.experiment_config.collect_config(), "easy_mcs": self.press_panel.collect_config(),
                "session": {"base_folder": str(self._base_folder_path) if self._base_folder_path else None,
                            "experiment_amplitude_mVpp": self.spin_experiment_amplitude.value()},
                "axis_calibration": calibration}

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try: os.unlink(temp_name)
            except FileNotFoundError: pass

    def _save_experiment_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить конфигурацию эксперимента", "experiment_profile.json", "Профиль эксперимента (*.json)")
        if not path: return
        try:
            target = Path(path)
            if target.suffix.lower() != ".json": target = target.with_suffix(".json")
            self._write_json_atomic(target, self._collect_experiment_config())
            self.lbl_status.setText("Профиль эксперимента сохранён"); self.lbl_status.setToolTip(str(target))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Ошибка сохранения профиля", str(exc))

    def _load_experiment_profile(self) -> None:
        if self.base_watcher is not None or self.streamer is not None:
            QtWidgets.QMessageBox.warning(self, "Идёт сеанс", "Остановите наблюдение перед загрузкой другой конфигурации."); return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Загрузить конфигурацию эксперимента", "", "Профиль эксперимента (*.json)")
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict): raise ValueError("Корень JSON должен быть объектом.")
            press_config = payload.get("easy_mcs", payload.get("press", {}))
            sample_config = payload.get("sample", {})
            if not sample_config and isinstance(press_config, dict):
                sample_config = {key: press_config[key] for key in ("sample_name", "width_mm", "length_mm", "tube_mode", "frequency_tag", "sample_note") if key in press_config}
            self.experiment_config.apply_config(sample_config)
            self.press_panel.apply_config(press_config)
            session = payload.get("session", {})
            if session.get("base_folder"):
                candidate = Path(session["base_folder"])
                if candidate.is_dir(): self._base_folder_path = candidate.resolve(); self.edit_base_folder.setText(str(self._base_folder_path))
            if session.get("experiment_amplitude_mVpp") is not None:
                self.spin_experiment_amplitude.setValue(float(session["experiment_amplitude_mVpp"]))
            self.lbl_status.setText("Профиль эксперимента загружен"); self.lbl_status.setToolTip(str(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Ошибка загрузки профиля", f"Не удалось загрузить конфигурацию:\n{exc}")

    def _write_experiment_snapshot(self, folder: Path) -> Path:
        folder = folder.resolve()
        target = folder / "experiment_config.json"
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = folder / f"experiment_config_dashboard_{stamp}.json"
        payload = self._collect_experiment_config()
        payload["session"]["experiment_folder"] = str(folder)
        self._write_json_atomic(target, payload)
        return target

    def _update_export_button(self) -> None:
        if not hasattr(self, "btn_export"): return
        folder = self._active_experiment_folder
        idle = self.base_watcher is None and self.streamer is None and self.csv_streamer is None
        self.btn_export.setEnabled(bool(not self.is_synthetic and idle and folder and folder.is_dir() and self._export_worker is None))

    def _open_export_dialog(self) -> None:
        folder = self._active_experiment_folder
        if folder is None or not folder.is_dir():
            QtWidgets.QMessageBox.information(self, "Экспорт", "Сначала дождитесь обнаружения папки эксперимента."); return
        if self.base_watcher is not None or self.streamer is not None or self.csv_streamer is not None:
            QtWidgets.QMessageBox.warning(self, "Экспорт ещё рано запускать", "Сначала нажмите «Остановить наблюдение», чтобы MCS/CSV-файлы больше не дописывались."); return
        cal = load_calibration(self.calibration_path)
        u_cal = cal.get("reference_amplitude_mVpp") if cal else None
        u_series = float(getattr(self, "_run_experiment_amplitude_mVpp", self.spin_experiment_amplitude.value()))
        if cal is None or not u_cal or u_series <= 0:
            QtWidgets.QMessageBox.warning(self, "Нет действующей калибровки", "Для научно корректного экспорта нужна действующая калибровка Aк и положительные Uк/Uс. Сначала сохраните калибровку оси θ и проверьте амплитуду серии."); return
        dialog = ExportOptionsDialog(folder, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted: return
        opts = dialog.collect_options()
        output_dir = Path(opts.pop("output_folder")).expanduser().resolve()
        if output_dir.exists() and any(p.is_file() for p in output_dir.rglob("*")):
            answer = QtWidgets.QMessageBox.question(self, "Каталог результатов не пуст", f"В каталоге уже есть файлы:\n{output_dir}\n\nСовпадающие имена могут быть перезаписаны. Продолжить?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
        try: calibration_args = build_pipeline_calibration_args(cal, u_series)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Некорректная калибровка", str(exc)); return
        request = {"input_folder": str(folder), "output_folder": str(output_dir), **calibration_args, **opts}
        self.btn_export.setEnabled(False); self.lbl_status.setText("Выполняется экспорт результатов…")
        self._export_worker = PipelineExportWorker(request, self)
        self._export_worker.succeeded.connect(self._on_export_success)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.start()

    @QtCore.pyqtSlot(str, list)
    def _on_export_success(self, output_dir: str, files: list) -> None:
        self.lbl_status.setText(f"Экспорт завершён · файлов: {len(files)}")
        self.lbl_status.setToolTip("\n".join(files))
        QtWidgets.QMessageBox.information(self, "Экспорт завершён", f"Результаты сохранены в:\n{output_dir}\n\nСоздано файлов: {len(files)}")

    @QtCore.pyqtSlot(str)
    def _on_export_failed(self, details: str) -> None:
        self.lbl_status.setText("Ошибка экспорта")
        QtWidgets.QMessageBox.critical(self, "Ошибка экспорта", details[-6000:])

    @QtCore.pyqtSlot()
    def _on_export_finished(self) -> None:
        worker = self._export_worker; self._export_worker = None
        if worker is not None: worker.deleteLater()
        self._update_export_button()

    @QtCore.pyqtSlot(dict)
    def _on_calibration_saved(self, payload: dict) -> None:
        # Смена φ в середине серии меняет смысл уже прочитанных s. Не применяем её к текущему потоку.
        if self.streamer is not None or self._current_records:
            QtWidgets.QMessageBox.warning(self, "Калибровка не применена",
                "Сначала остановите наблюдение и начните новый сеанс: масштаб θ должен быть неизменным на протяжении одной серии.")
            return
        try:
            saved = save_calibration(self.calibration_path, payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Ошибка сохранения калибровки", str(exc)); return
        self.axis_calibration = saved
        self._update_axis_ui()
        self.lbl_status.setText("Калибровка θ сохранена")
        self.lbl_status.setToolTip(str(self.calibration_path))

    def _rebuild_current_map(self) -> None:
        if not self._current_records: return
        times = self.time_s
        records = self._current_records
        intensity = np.asarray([r.counts for r in records], dtype=float)
        first = records[0]
        self.roc_map_live.update({"time_s": times, "s_axis": getattr(first, "s_axis", first.theta_axis),
            "theta_axis": first.theta_axis, "axis_name": getattr(first, "axis_name", "Параметр s"),
            "axis_unit": getattr(first, "axis_unit", "отн. ед."),
            "axis_calibrated": getattr(first, "axis_calibrated", False), "intensity": intensity,
            "records": records, "model_name": getattr(records[-1], "model_name", "pvoigt")})
        self.metrics_plot.set_axis_units(getattr(first, "axis_unit", "отн. ед."), getattr(first, "axis_calibrated", False))
        self.map_plot.set_roc_map(self.roc_map_live)
        self.filmstrip.render_state(self.roc_map_live, highlight_idx=min(self.pinned_index, len(records)-1))
        self.metrics_plot.reset()
        self.metrics_plot.update_data(self.time_s, records)

    def _apply_theme(self, theme: str | None = None) -> None:
        app = QtWidgets.QApplication.instance(); requested = str(theme or (app.property("rocMcsTheme") if app is not None else "dark")).lower()
        active = requested if requested in ("dark", "light") else "dark"
        dark = active == "dark"; fg = "#e2e8f0" if dark else "#263445"; bg = (16, 20, 28) if dark else (245, 247, 250)
        if app is not None: set_icon_theme(app, fg)
        pg.setConfigOptions(antialias=True, foreground=(226, 232, 240) if dark else (38, 52, 69), background=bg)
        self.map_plot.apply_theme(active); self.metrics_plot.apply_theme(active); self.filmstrip.apply_theme(active)
        qss = """
            QMainWindow, QWidget { background:#10141c; color:#e2e8f0; font-family:'Segoe UI',Arial,sans-serif; font-size:11px; }
            QWidget#appSurface, QWidget#topToolbar { background:#10141c; border:0; }
            QFrame#toolbarDivider { color:#2c3746; background:#2c3746; max-width:1px; }
            QWidget#navRail { background:#10141c; border-right:1px solid #263242; }
            QToolButton#railButton { background:transparent; border:1px solid transparent; border-radius:5px; padding:2px; min-width:28px; min-height:28px; }
            QToolButton#railButton:hover { background:#1b2634; border-color:#303f52; }
            QToolButton#railButton:checked { background:#203349; border-color:#4f83b8; }
            QToolButton#toolbarIconButton { background:transparent; border:1px solid transparent; border-radius:5px; padding:2px; min-height:25px; min-width:27px; }
            QToolButton#toolbarIconButton:hover { background:#1b2634; border-color:#303f52; }
            QToolButton#toolbarIconButton:disabled { color:#637186; background:transparent; border-color:transparent; }
            QLineEdit#baseFolderField { background:#121923; border:1px solid #263445; color:#91a5bb; padding:3px 6px; min-height:20px; }
            QLabel#appTitle { font-size:12px; font-weight:700; letter-spacing:0.5px; color:#dfe8f4; }
            QLabel#statusText { color:#9eafc3; padding:0 5px; }
            QWidget#sidebar { background:#141a24; border-right:1px solid #263242; }
            QToolButton#collapsibleSectionHeader { color:#9fb9d5; background:#141a24; border:0; border-bottom:1px solid #303b4a; border-radius:3px; padding:5px 4px; font-size:10px; font-weight:700; text-align:left; }
            QToolButton#collapsibleSectionHeader:hover { background:#1b2634; }
            QToolButton#iconActionButton, QToolButton#pressActionButton { text-align:center; padding:2px; min-height:22px; min-width:26px; }
            QLabel#calibrationStatus { color:#e3bd7a; padding:3px 0; }
            QLabel#axisScaleDetails { color:#9eafc3; font-size:10px; padding:2px 0 4px 0; }
            QLabel#panelHint { color:#d6b878; font-size:10px; padding:2px; }
            QScrollArea { background:transparent; border:0; }
            QLabel#sideFootnote { color:#78889d; font-size:10px; border-top:1px solid #303b4a; padding-top:6px; }
            QPushButton, QToolButton { background:#1a2330; border:1px solid #354255; border-radius:4px; padding:4px 6px; min-height:20px; text-align:left; }
            QToolButton#toolbarIconButton, QToolButton#railButton { text-align:center; }
            QLineEdit#baseFolderField { border-radius:4px; }
            QPushButton:hover, QToolButton:hover { background:#263448; border-color:#4c6889; }
            QPushButton:pressed, QToolButton:pressed { background:#15202d; }
            QPushButton:disabled { color:#637186; background:#171d27; border-color:#293341; }
            QPushButton#autoLevelsButton { padding:3px 5px; min-height:18px; }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QPlainTextEdit { background:#151d28; color:#e2e8f0; border:1px solid #354255; border-radius:4px; padding:3px 5px; min-height:21px; }
            QCheckBox { spacing:5px; color:#c6d2e1; }
            QCheckBox::indicator { width:13px; height:13px; }
            QSplitter::handle { background:#263242; }
            QSplitter::handle:hover { background:#45617f; }
            QScrollBar:horizontal, QScrollBar:vertical { background:#121823; border:0; width:10px; height:10px; }
            QScrollBar::handle { background:#344255; border-radius:4px; min-width:24px; min-height:24px; }
            QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
        """
        if not dark:
            replacements = {"#10141c":"#f5f7fa", "#e2e8f0":"#263445", "#2c3746":"#cbd5e1", "#263242":"#cbd5e1", "#1b2634":"#e3eaf2", "#303f52":"#bdcad8", "#203349":"#d6e6f5", "#4f83b8":"#527fa8", "#121923":"#ffffff", "#263445":"#526276", "#91a5bb":"#526276", "#dfe8f4":"#263445", "#9eafc3":"#64748b", "#141a24":"#edf1f5", "#9fb9d5":"#475569", "#303b4a":"#cbd5e1", "#d6b878":"#966c18", "#78889d":"#64748b", "#1a2330":"#edf1f5", "#354255":"#cbd5e1", "#15202d":"#dfe7f0", "#637186":"#94a3b8", "#171d27":"#e5eaf0", "#293341":"#cbd5e1", "#151d28":"#ffffff", "#c6d2e1":"#334155", "#45617f":"#89a8c9", "#344255":"#cbd5e1", "#121823":"#f5f7fa", "#2a394c":"#dfe7f0"}
            tokens = {old: f"__THEME_COLOR_{i}__" for i, old in enumerate(replacements)}
            for old, token in tokens.items(): qss = qss.replace(old, token)
            for old, new_color in replacements.items(): qss = qss.replace(tokens[old], new_color)
        self.setStyleSheet(qss)
        size = max(12, min(20, int(app.property("rocMcsIconSize") if app is not None else 16)))
        self.btn_sidebar.setIcon(self._pane_icon(self._sidebar_expanded)); self.btn_open.setIcon(resource_svg_icon("folder_open.svg", widget=self.btn_open, size=size, color=fg)); self.btn_stop.setIcon(resource_svg_icon(WATCH_STOP_ICON, widget=self.btn_stop, size=size, color=fg)); self.btn_export.setIcon(resource_svg_icon("save_icon.svg", widget=self.btn_export, size=size, color=fg))
        self.nav_rail.apply_theme(fg, size)
        for button in self.findChildren(QtWidgets.QAbstractButton):
            if not button.icon().isNull(): button.setIconSize(QtCore.QSize(size, size))
        for label in self.findChildren(QtWidgets.QLabel):
            if label.text().startswith("Потоковая обработка ·"): label.setText("Потоковая обработка · " + ("тёмная" if dark else "светлая") + " тема")

    def apply_global_settings(self, settings: dict | None = None) -> None:
        """Получает глобальные параметры оболочки, не прерывая активный эксперимент."""
        settings = settings or {}; app = QtWidgets.QApplication.instance(); theme = str(app.property("rocMcsTheme") if app is not None else settings.get("theme", "dark"))
        self._apply_theme(theme)
        if self.is_synthetic: return
        configured = str(settings.get("experiments_dir", "") or "").strip(); candidate = Path(configured).expanduser() if configured else None
        active = self.base_watcher is not None or self.streamer is not None or self.csv_streamer is not None
        if not active:
            self._base_folder_path = candidate.resolve() if candidate is not None and candidate.is_dir() else None
            self.edit_base_folder.setText(str(self._base_folder_path) if self._base_folder_path else "")
            self.edit_base_folder.setToolTip(str(self._base_folder_path) if self._base_folder_path else "Базовый каталог не выбран")
            if self._base_folder_path: self.lbl_status.setText("Сохранённый каталог указан; нажмите кнопку папки, чтобы начать ожидание")

    # --- ЛОГИКА МАТЕМАТИКИ И ГРАНИЦ ---
    def _nearest_index(self, x: float) -> int:
        if self.time_s.size == 0: return 0
        return int(np.argmin(np.abs(self.time_s - x)))

    def _slice_bounds(self, idx: int) -> tuple[float, float]:
        n = len(self.time_s)
        if n <= 1: 
            return (self.time_s[0] - 0.5, self.time_s[0] + 0.5) if n == 1 else (0.0, 0.0)
        if idx == 0:
            dt = self.time_s[1] - self.time_s[0]
            return float(self.time_s[0] - dt / 2), float(self.time_s[0] + dt / 2)
        if idx == n - 1:
            dt = self.time_s[-1] - self.time_s[-2]
            return float(self.time_s[-1] - dt / 2), float(self.time_s[-1] + dt / 2)
            
        left = 0.5 * (self.time_s[idx - 1] + self.time_s[idx])
        right = 0.5 * (self.time_s[idx] + self.time_s[idx + 1])
        return float(left), float(right)

    # --- ОБРАБОТЧИКИ СОБЫТИЙ МЫШИ ---
    def _on_map_hover(self, x: float) -> None:
        if len(self.time_s) == 0: return
        idx = self._nearest_index(x)
        self.hover_state = {'idx': idx, 'x': x}
        self._sync_visuals()
        self.filmstrip.scroll_to_index(idx)

    def _on_strip_hover(self, idx: int) -> None:
        if len(self.time_s) == 0: return
        x = float(self.time_s[idx])
        self.hover_state = {'idx': idx, 'x': x}
        self._sync_visuals()

    def _on_hover_left(self) -> None:
        self.hover_state = None
        self._sync_visuals()
        if self.follow_latest:
            self.filmstrip.scroll_to_end()
        else:
            self.filmstrip.scroll_to_index(self.pinned_index)

    def _on_map_click(self, x: float) -> None:
        if len(self.time_s) == 0: return
        self._pin_index(self._nearest_index(x))

    def _on_strip_click(self, idx: int) -> None:
        self._pin_index(idx)

    def _pin_index(self, idx: int) -> None:
        self.pinned_index = idx
        self.follow_latest = False
        self.chk_follow.blockSignals(True)
        self.chk_follow.setChecked(False)
        self.chk_follow.blockSignals(False)
        
        self.hover_state = None
        self._sync_visuals()
        self.filmstrip.scroll_to_index(self.pinned_index)

    def _sync_visuals(self) -> None:
        """Центральный метод отрисовки."""
        if not self._current_records: return

        active_idx = self.hover_state['idx'] if self.hover_state else self.pinned_index

        self.filmstrip.render_state(self.roc_map_live, highlight_idx=active_idx)

        if self.hover_state:
            cursor_x = self.hover_state['x']
            left, right = self._slice_bounds(active_idx)
            self.map_plot.show_cursors(cursor_x, left, right)
            self.metrics_plot.show_cursor(cursor_x)
        else:
            self.map_plot.hide_cursors()
            self.metrics_plot.hide_cursor()

    # --- ЕДИНЫЙ КОНВЕЙЕР ДАННЫХ (SINGLE PIPELINE) ---
    def _process_new_record(self, record) -> None:
        """Общая логика добавления нового фрейма (неважно: стример это или таймер)."""
        self._current_records.append(record)

        times = [r.header["file_info"]["datetime"] for r in self._current_records]
        t0 = times[0]
        self.time_s = np.array([(t - t0).total_seconds() for t in times], dtype=float)
        intensity = np.asarray([r.counts for r in self._current_records], dtype=float)

        first = self._current_records[0]
        self.roc_map_live.update({
            "time_s": self.time_s,
            "s_axis": getattr(first, "s_axis", first.theta_axis),
            "theta_axis": first.theta_axis,
            "axis_name": getattr(first, "axis_name", "Параметр s"),
            "axis_unit": getattr(first, "axis_unit", "отн. ед."),
            "axis_calibrated": getattr(first, "axis_calibrated", False),
            "intensity": intensity,
            "records": self._current_records,
            "model_name": getattr(record, "model_name", "pvoigt")
        })
        self.metrics_plot.set_axis_units(getattr(first, "axis_unit", "отн. ед."), getattr(first, "axis_calibrated", False))

        self.map_plot.set_roc_map(self.roc_map_live)
        self.filmstrip.sync_cards(len(self.time_s))
        self.metrics_plot.update_data(self.time_s, self._current_records)

        if self.follow_latest:
            self.pinned_index = len(self._current_records) - 1

        self._sync_visuals()

        if self.follow_latest and not self.hover_state:
            QtCore.QTimer.singleShot(0, self.filmstrip.scroll_to_end)        
        
        mode = "деморежим" if self.is_synthetic else "эксперимент"
        self.lbl_status.setText(f"{mode}: получено срезов — {len(self._current_records)}")
        if not self.is_synthetic:
            phi = getattr(record, "phase_phi", None)
            s_axis = np.asarray(getattr(record, "s_axis", []), dtype=float)
            scale = getattr(record, "scale_applied", None)
            details = [f"Файл: {getattr(record, 'file_name', '') or '—'}",
                       f"Фазовый эталон: {getattr(record, 'phase_source_file', None) or '—'}",
                       f"Фаза live-серии φ: {phi if phi is not None else 'ещё не определена'}",
                       f"Диапазон s: {np.nanmin(s_axis):.6g} … {np.nanmax(s_axis):.6g}" if s_axis.size else "Диапазон s: —"]
            if scale is not None:
                x = np.asarray(getattr(record, "theta_axis", []), dtype=float)
                details += [f"Применённый масштаб A серии: {scale:.8g} угл. с",
                            f"Диапазон θ: {np.nanmin(x):.6g} … {np.nanmax(x):.6g} угл. с" if x.size else "Диапазон θ: —"]
            else:
                details.append("Угловая калибровка не применена; отображается s.")
            self.lbl_status.setToolTip("\n".join(details))

    # --- УПРАВЛЕНИЕ: REAL FOLDER (STREAMER) ---
    def _start_watching_folder(self) -> None:
        initial = str(self._base_folder_path) if self._base_folder_path else str(Path.home())
        base_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите базовый каталог экспериментов", initial)
        if not base_folder:
            return
        self._stop_watching(); self._active_experiment_folder = None; self._clear_state(); self._update_export_button()
        self._base_folder_path = Path(base_folder).expanduser().resolve()
        self.edit_base_folder.setText(str(self._base_folder_path)); self.edit_base_folder.setToolTip(str(self._base_folder_path))
        try:
            settings = load_settings(); settings["experiments_dir"] = str(self._base_folder_path); save_settings(settings)
        except (OSError, ValueError, TypeError) as exc:
            self.lbl_status.setToolTip(f"Каталог выбран, но не удалось сохранить его в общих настройках: {exc}")
        self._run_experiment_amplitude_mVpp = self.spin_experiment_amplitude.value()
        if self._current_effective_scale() is None:
            answer = QtWidgets.QMessageBox.warning(self, "Угловая шкала не активна", "Нет действующей калибровки или значения U серии. Будет отображаться ось s, не физический угол θ. Продолжить?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.lbl_status.setText("Наблюдение не запущено"); return
        try:
            self.base_watcher = BaseFolderWatcher(base_folder=self._base_folder_path, parent=self)
        except (OSError, ValueError, NotADirectoryError) as exc:
            self.base_watcher = None
            QtWidgets.QMessageBox.critical(self, "Не удалось наблюдать каталог", str(exc)); return
        self.base_watcher.sigNewExperimentFound.connect(self._on_new_experiment_folder)
        self.lbl_status.setText(f"Ожидание новой папки в «{self._base_folder_path.name}»")
        self.btn_stop.setEnabled(True); self.btn_open.setEnabled(False); self.spin_experiment_amplitude.setEnabled(False)
        self.experiment_config.setEnabled(False); self.press_panel.set_acquisition_locked(True)

    def _stop_watching(self) -> None:
        # Останавливаем стример данных (если он уже запущен)
        if self.streamer:
            try:
                self.streamer.sigRecordReady.disconnect(self._on_streamer_record)
            except Exception:
                pass
            self.streamer.stop()
            self.streamer = None
            
        # Останавливаем следилку за папкой (если она ждала)
        if self.base_watcher:
            try:
                self.base_watcher.sigNewExperimentFound.disconnect(self._on_new_experiment_folder)
            except Exception:
                pass
            self.base_watcher.stop()
            self.base_watcher = None
        
        # Останавливаем CSV стример <--- ДОБАВИТЬ ЭТОТ БЛОК
        if self.csv_streamer:
            try:
                self.csv_streamer.sigDataReady.disconnect(self._on_csv_load_record)
            except Exception:
                pass
            self.csv_streamer.stop()
            self.csv_streamer = None            
            
        self.btn_stop.setEnabled(False)
        self.btn_open.setEnabled(True)
        self.spin_experiment_amplitude.setEnabled(True)
        self.press_panel.set_acquisition_locked(False)
        self.experiment_config.setEnabled(True)
        self.lbl_status.setText("Наблюдение остановлено")
        self._update_export_button()
    
    @QtCore.pyqtSlot(dict)
    def _on_streamer_record(self, record_dict: dict) -> None:
        # Пропускаем сырой словарь через Адаптер
        record = LiveMcsAdapter(record_dict, calibration=self.axis_calibration, experiment_amplitude_mVpp=getattr(self, "_run_experiment_amplitude_mVpp", self.spin_experiment_amplitude.value()))
        self._process_new_record(record)

    # Создадим новый слот (обработчик), который сработает, когда папка появится
    @QtCore.pyqtSlot(str)
    def _on_new_experiment_folder(self, new_folder_path: str) -> None:
        # Один сеанс наблюдения обслуживает один эксперимент; не оставляем старые стримеры.
        if self.base_watcher:
            try:
                self.base_watcher.sigNewExperimentFound.disconnect(self._on_new_experiment_folder)
            except Exception:
                pass
            self.base_watcher.stop()
            self.base_watcher = None
        self._active_experiment_folder = Path(new_folder_path).expanduser().resolve()
        self.lbl_status.setText(f"Идёт обработка: {Path(new_folder_path).name}")
        try:
            snapshot_path = self._write_experiment_snapshot(Path(new_folder_path))
            self.lbl_status.setToolTip(f"Конфигурация сеанса: {snapshot_path}")
        except Exception as exc:
            self.lbl_status.setText(f"Идёт обработка: {Path(new_folder_path).name} · конфиг не сохранён")
            self.lbl_status.setToolTip(f"Ошибка записи конфигурации: {exc}")

        # Новая папка обнаружена: запускаем MCS и CSV-потоки именно для неё.
        branch = self.axis_calibration.get("branch", "up") if self.axis_calibration else "up"
        # Стример сам определит фазу по первому MCS этой серии; калибровочная φ не переносится в live.
        self.streamer = ExperimentalMcsStreamer(folder=new_folder_path, branch=branch)
        self.streamer.sigRecordReady.connect(self._on_streamer_record)
        
        # В будущем именно здесь ты добавишь запуск стримера для CSV-логов:
        # self.csv_streamer = CsvStreamer(folder=new_folder_path)
        # self.csv_streamer.sigCsvRowReady.connect(...)   
        # 2. Запускаем CSV стример нагрузки <--- ДОБАВИТЬ ЭТОТ БЛОК
        self.csv_streamer = CsvStreamer(folder=new_folder_path)
        self.csv_streamer.sigDataReady.connect(self._on_csv_load_record)
        self.csv_streamer.start() # Так как это QThread, запускаем через start()
        self._update_export_button()

    # Добавляем слот-обработчик новых точек нагрузки:
    @QtCore.pyqtSlot(float, float, float)
    def _on_csv_load_record(self, t: float, f: float, p: float) -> None:
        self.load_t.append(t)
        self.load_f.append(f)
        self.load_p.append(p)
        self._refresh_load_curve()

    def _on_load_metric_changed(self, _index: int) -> None:
        self.load_metric = self.cmb_load_metric.currentData()
        self._refresh_load_curve()

    def _refresh_load_curve(self) -> None:
        values = self.load_p if self.load_metric == "pressure" else self.load_f
        self.map_plot.set_load_data(np.asarray(self.load_t, dtype=float), np.asarray(values, dtype=float), self.load_metric)

    # --- Синтетический режим ---
    def _set_follow_latest(self, checked: bool) -> None:
        self.follow_latest = checked
        if checked and self._current_records:
            self.pinned_index = len(self._current_records) - 1
            self._sync_visuals()
            self.filmstrip.scroll_to_end()

    def _clear_state(self) -> None:
        """Сброс всех массивов и индексов перед новым запуском."""
        self._current_records = []
        self.time_s = np.array([], dtype=float)
        self.roc_map_live = {}
        self.pinned_index = 0
        self.hover_state = None
        self.load_t.clear()
        self.load_f.clear()
        self.load_p.clear()
        if hasattr(self, "metrics_plot"): self.metrics_plot.reset()
        if hasattr(self, "filmstrip"): self.filmstrip.sync_cards(0)
        self.map_plot.set_load_data(np.array([]), np.array([]), self.load_metric)
        self.map_plot.clear_map()
        if hasattr(self, "lbl_color_levels"): self.lbl_color_levels.setText("I: —")

    def _reset_synthetic_stream(self) -> None:
        self._clear_state()
        self._stream_index = 0
        self._append_synthetic_record() # загружаем первый кадр сразу
        self._timer.start()

    def _append_synthetic_record(self) -> None:
        if self._stream_index >= len(self.records):
            self._timer.stop()
            self.lbl_status.setText("Демонстрация завершена")
            return
        record = self.records[self._stream_index]
        self._stream_index += 1
        self._process_new_record(record)

    def closeEvent(self, event) -> None:
        """Не закрывает окно посреди калибровки/экспорта и останавливает потоки."""
        if self._export_worker is not None and self._export_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Идёт экспорт", "Дождитесь завершения экспорта перед закрытием приложения.")
            event.ignore(); return
        dialog = self._calibration_dialog
        worker = getattr(getattr(dialog, "widget", None), "worker", None) if dialog else None
        if worker is not None and worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Идёт калибровка", "Дождитесь завершения калибровки перед закрытием приложения.")
            event.ignore()
            return
        self._timer.stop()
        self._stop_watching()
        event.accept()


# -----------------------------------------------------------------------------
# Convenience function for notebooks
# -----------------------------------------------------------------------------

def make_demo_dashboard(
    model_name: str = "pvoigt",
    seed: int = 7,
    theta_axis: np.ndarray | None = None,
    noise: str | float | None = "poisson",
) -> LiveRocDashboard:
    """Create the dashboard window preloaded with the demo parameter table."""
    param_df = make_demo_param_table()
    records = make_synthetic_mcs_records(
        param_df,
        model_name=model_name,
        theta_axis=theta_axis,
        noise=noise,
        seed=seed,
        start_datetime=datetime(2026, 6, 25, 12, 0, 0),
    )
    return LiveRocDashboard(records)




    
# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # win = make_demo_dashboard()
    # Просто создаем пустой дашборд без аргументов
    win = LiveRocDashboard(records=None)     
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

