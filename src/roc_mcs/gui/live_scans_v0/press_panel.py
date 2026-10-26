"""Параметры режимов Easy-MCS; команды оборудования намеренно не подключены."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.icon_provider import resource_svg_icon

EMERGENCY_ICON = "emergency_stop.svg"


class PressControlPanel(QtWidgets.QWidget):
    """UI-слой для выбора режима/параметров. Не открывает COM и не запускает MCS-32."""
    configuration_changed = QtCore.pyqtSignal(dict)

    MODE_ITEMS = (
        ("Подготовка / юстировка", "prep"), ("Обычный эксперимент", "ordinary"),
        ("Ступенчатый", "stepped"), ("Периодический", "periodic"),
        ("Циклический", "cycle"), ("Ручное управление", "manual"),
        ("Выдержка под нагрузкой", "hold"), ("Повторные сканирования", "repeat"),
        ("Нагрузка–разгрузка по программе", "load_unload"),
        ("Программируемый протокол", "programmed"), ("Адаптивные ступени", "adaptive"),
        ("Заданная скорость · только симуляция", "ramp"),
    )

    def __init__(self, parent=None):
        super().__init__(parent); self._locked = False; self._fields = {}
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(5)
        self.mode = QtWidgets.QComboBox()
        for text, key in self.MODE_ITEMS: self.mode.addItem(text, key)
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.addWidget(QtWidgets.QLabel("Режим")); row.addWidget(self.mode, 1); root.addLayout(row)
        self.shared_note = QtWidgets.QLabel("Образец, размеры и режим источника — в разделе «Образец и условия».")
        self.shared_note.setWordWrap(True); self.shared_note.setObjectName("panelHint"); root.addWidget(self.shared_note)

        self.meta_toggle, self.meta_box, mf = self._collapsible(root, "Параметры MCS", checked=False)
        self.preset_pass = QtWidgets.QSpinBox(); self.preset_pass.setRange(1, 100_000_000); self.preset_pass.setValue(1000)
        self.preset_pass.setToolTip("Число циклов накопления; в циклическом режиме фиксировано настройкой ядра.")
        mf.addRow("Циклы накопления", self.preset_pass)
        self.cycle_preset_note = QtWidgets.QLabel("Циклический режим использует 1000 циклов по настройке контроллера.")
        self.cycle_preset_note.setWordWrap(True); self.cycle_preset_note.setObjectName("panelHint"); mf.addRow(self.cycle_preset_note)

        self.control_toggle, self.control_box, cf = self._collapsible(root, "Параметры регулирования", checked=False)
        self.control_dt = self._double(0.01, 3600, 2, " с", 0.5); self.control_dt.setToolTip("Период шага регулирования; не период между MCS-сканами.")
        self.use_alignment = QtWidgets.QCheckBox("Использовать сохранённую юстировку"); self.use_alignment.setChecked(True)
        cf.addRow("Шаг регулирования", self.control_dt); cf.addRow(self.use_alignment)

        self.pages = QtWidgets.QStackedWidget(); root.addWidget(self.pages)
        self.page_forms = {}; self.page_widgets = {}
        self.prep_page, pf = self._page("prep")
        self.prep_mode = QtWidgets.QComboBox()
        for text, key in (("Задать давление", "pressure"), ("Автоподвод и ручная доводка", "align"),
                          ("Сбросить сохранённую юстировку", "clear"), ("Ручная юстировка + один MCS", "manual")):
            self.prep_mode.addItem(text, key)
        self.pressure_bar = self._double(0, 10, 2, " бар", 1.0)
        self.manual_step_v = self._double(0.001, 1, 3, " В", 0.02)
        self.prep_hint = QtWidgets.QLabel("Автоподвод: при первом касании ожидается решение оператора."); self.prep_hint.setWordWrap(True); self.prep_hint.setObjectName("panelHint")
        pf.addRow("Операция", self.prep_mode); pf.addRow("Заданное давление*", self.pressure_bar); pf.addRow("Шаг ручного поджима", self.manual_step_v); pf.addRow(self.prep_hint)

        self.ordinary_page, f = self._page("ordinary")
        self.ordinary_target = self._number_field("ordinary_target_kg", 0.1, 700, 1, " кг", 100); f.addRow("Целевая нагрузка", self.ordinary_target)

        self.stepped_page, f = self._page("stepped")
        self.stepped_target = self._number_field("stepped_target_kg", 0.1, 700, 1, " кг", 100)
        self.load_step = self._number_field("load_step_kg", 0.1, 700, 1, " кг", 10)
        f.addRow("Конечная нагрузка", self.stepped_target); f.addRow("Шаг нагрузки", self.load_step)

        self.periodic_page, f = self._page("periodic")
        self.periodic_direction = QtWidgets.QComboBox(); self.periodic_direction.addItem("Нагружение", "forward"); self.periodic_direction.addItem("Разгрузка", "reverse")
        self.periodic_target = self._number_field("periodic_target_kg", 0.1, 700, 1, " кг", 100)
        self.scan_interval = self._number_field("scan_interval_s", 0.1, 86400, 1, " с", 60)
        f.addRow("Направление", self.periodic_direction); f.addRow("Конечная нагрузка", self.periodic_target); f.addRow("Интервал MCS", self.scan_interval)

        self.cycle_page, f = self._page("cycle")
        self.cycle_target = self._number_field("cycle_target_kg", 0.1, 700, 1, " кг", 100)
        self.stabilize_sec = self._number_field("stabilize_sec", 0, 86400, 1, " с", 5)
        self.cycle_info = QtWidgets.QLabel("Во время MCS давление фиксировано; после скана — подкачка/разгрузка и стабилизация.")
        self.cycle_info.setWordWrap(True); self.cycle_info.setObjectName("panelHint")
        f.addRow("Целевая нагрузка", self.cycle_target); f.addRow("Стабилизация", self.stabilize_sec); f.addRow(self.cycle_info)

        self.manual_page, f = self._page("manual")
        self.manual_hint = QtWidgets.QLabel("Ручной режим: цель/давление, пауза, возврат к касанию, мягкий спуск и завершение. Команды пока отключены.")
        self.manual_hint.setWordWrap(True); self.manual_hint.setObjectName("panelHint"); f.addRow(self.manual_hint)

        self.hold_page, f = self._page("hold")
        self.hold_target = self._number_field("hold_target_kg", 0.1, 700, 1, " кг", 50)
        self.hold_duration = self._number_field("hold_duration_s", 0, 86400, 1, " с", 60)
        self.hold_scans = self._int_field("hold_scans", 1, 100000, 6)
        self.hold_interval = self._number_field("hold_interval_s", 0, 86400, 1, " с", 60)
        f.addRow("Нагрузка", self.hold_target); f.addRow("Выдержка до сканов", self.hold_duration); f.addRow("Число сканов", self.hold_scans); f.addRow("Интервал сканов", self.hold_interval)

        self.repeat_page, f = self._page("repeat")
        self.repeat_target = self._number_field("repeat_target_kg", 0.1, 700, 1, " кг", 50)
        self.repeat_scans = self._int_field("repeat_scans", 1, 100000, 10)
        self.repeat_interval = self._number_field("repeat_interval_s", 0, 86400, 1, " с", 3)
        f.addRow("Нагрузка", self.repeat_target); f.addRow("Число сканов", self.repeat_scans); f.addRow("Интервал", self.repeat_interval)

        self.load_unload_page, f = self._page("load_unload")
        self.load_route = QtWidgets.QLineEdit("10;20;40;20;10"); self.load_route.setToolTip("Целевые нагрузки в кг, через точку с запятой.")
        self.route_hold = self._number_field("route_hold_s", 0, 86400, 1, " с", 7)
        self.route_scans = self._int_field("route_scans", 1, 100000, 1)
        f.addRow("Нагрузки, кг", self.load_route); f.addRow("Выдержка на ступени", self.route_hold); f.addRow("Сканов на ступень", self.route_scans)

        self.programmed_page, f = self._page("programmed")
        self.program_text = QtWidgets.QPlainTextEdit("10|7|1|0;20|10|2|5;10|7|1|0")
        self.program_text.setFixedHeight(76); self.program_text.setPlaceholderText("цель кг | выдержка с | сканов | интервал с; …")
        f.addRow("Ступени (кг | с | шт. | с)", self.program_text)

        self.adaptive_page, f = self._page("adaptive")
        self.adaptive_start = self._number_field("adaptive_start_kg", 0, 700, 1, " кг", 0)
        self.adaptive_final = self._number_field("adaptive_final_kg", 0.1, 700, 1, " кг", 60)
        self.adaptive_step = self._number_field("adaptive_step_kg", 0.1, 700, 1, " кг", 20)
        self.adaptive_focus_start = self._number_field("adaptive_focus_start_kg", 0, 700, 1, " кг", 20)
        self.adaptive_focus_end = self._number_field("adaptive_focus_end_kg", 0, 700, 1, " кг", 40)
        self.adaptive_fine_step = self._number_field("adaptive_fine_step_kg", 0.1, 700, 1, " кг", 5)
        self.adaptive_hold = self._number_field("adaptive_hold_s", 0, 86400, 1, " с", 7)
        self.adaptive_scans = self._int_field("adaptive_scans", 1, 100000, 1)
        for label, widget in (("Начальная нагрузка", self.adaptive_start), ("Конечная нагрузка", self.adaptive_final), ("Основной шаг", self.adaptive_step),
                              ("Начало зоны мелких шагов", self.adaptive_focus_start), ("Конец зоны мелких шагов", self.adaptive_focus_end),
                              ("Малый шаг", self.adaptive_fine_step), ("Выдержка", self.adaptive_hold), ("Сканов на ступень", self.adaptive_scans)):
            f.addRow(label, widget)

        self.ramp_page, f = self._page("ramp")
        self.ramp_final = self._number_field("ramp_final_kg", 0.1, 700, 1, " кг", 30)
        self.ramp_rate = self._number_field("ramp_rate_kg_min", 0.1, 700, 1, " кг/мин", 30)
        self.ramp_scan_every = self._number_field("ramp_scan_every_kg", 0.1, 700, 1, " кг", 10)
        self.ramp_warning = QtWidgets.QLabel("Только симуляция в версии коллеги; на реальное оборудование не выводить без аттестации режима.")
        self.ramp_warning.setWordWrap(True); self.ramp_warning.setObjectName("panelHint")
        f.addRow("Конечная нагрузка", self.ramp_final); f.addRow("Желаемая скорость", self.ramp_rate); f.addRow("Скан каждые", self.ramp_scan_every); f.addRow(self.ramp_warning)

        for key, page in self.page_widgets.items(): self.pages.addWidget(page)
        self.mode.currentIndexChanged.connect(self._update_mode); self.prep_mode.currentIndexChanged.connect(self._update_mode)
        self.meta_toggle.toggled.connect(self.meta_box.setVisible); self.control_toggle.toggled.connect(self.control_box.setVisible)
        controls_row = QtWidgets.QHBoxLayout(); controls_row.setContentsMargins(0, 2, 0, 0); controls_row.setSpacing(4)
        self.btn_start = self._icon_button("play_icon.svg", "Запуск", "Макет: запуск оборудования не подключён.")
        self.btn_pause = self._icon_button("pause_icon.svg", "Пауза", "Макет: пауза оборудования не подключена.")
        self.btn_resume = self._icon_button("resume_icon.svg", "Продолжить", "Макет: продолжение не подключено.")
        self.btn_stop = self._icon_button("stop_icon.svg", "Штатно завершить", "Макет: штатное завершение не подключено.")
        self.btn_emergency = self._icon_button(EMERGENCY_ICON, "Аварийная остановка", "МАКЕТ: не является аппаратной аварийной остановкой и не отправляет команды.", color="#ef4444")
        for button in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop, self.btn_emergency):
            button.setEnabled(False); controls_row.addWidget(button)
        controls_row.addStretch(1); root.addLayout(controls_row)
        self.status = QtWidgets.QLabel("МАКЕТ: контроллер TRS_UMLM_CS и оборудование не подключены.")
        self.status.setObjectName("panelHint"); self.status.setWordWrap(True); root.addWidget(self.status)

        self.runtime_box = QtWidgets.QGroupBox("Команды во время опыта · макет")
        rf = QtWidgets.QGridLayout(self.runtime_box); rf.setContentsMargins(6, 5, 6, 5); rf.setHorizontalSpacing(4); rf.setVerticalSpacing(3)
        self.runtime_target = self._double(0.1, 700, 1, " кг", 50); self.runtime_pressure = self._double(0, 10, 2, " бар", 1)
        self.runtime_step_v = self._double(0.001, 1, 3, " В", 0.02)
        self.btn_runtime_target = QtWidgets.QPushButton("n · Новая цель"); self.btn_runtime_pressure = QtWidgets.QPushButton("g · Давление")
        self.btn_runtime_step_down = QtWidgets.QPushButton("− шаг"); self.btn_runtime_step_up = QtWidgets.QPushButton("+ шаг")
        self.btn_runtime_return = QtWidgets.QPushButton("b · К точке касания"); self.btn_runtime_softdown = QtWidgets.QPushButton("s · Мягкий спуск"); self.btn_runtime_exit = QtWidgets.QPushButton("q · Штатное завершение")
        self.btn_runtime_target.setToolTip("Изменить цель в поддерживаемом режиме; пока отключено."); self.btn_runtime_pressure.setToolTip("Задать давление; пока отключено.")
        self.runtime_step_v.setToolTip("Шаг напряжения для команды ручного управления; пока отключено.")
        self.btn_runtime_step_down.setToolTip("Уменьшить напряжение на заданный шаг; пока отключено."); self.btn_runtime_step_up.setToolTip("Увеличить напряжение на заданный шаг; пока отключено.")
        self.btn_runtime_return.setToolTip("Вернуться к сохранённой точке первого касания; пока отключено.")
        self.btn_runtime_softdown.setToolTip("Мягкий спуск, если поддерживается режимом; пока отключено.")
        self.btn_runtime_exit.setToolTip("Штатное меню завершения; не аварийный выключатель.")
        for widget in (self.runtime_target, self.runtime_pressure, self.runtime_step_v, self.btn_runtime_target, self.btn_runtime_pressure,
                       self.btn_runtime_step_down, self.btn_runtime_step_up, self.btn_runtime_return, self.btn_runtime_softdown, self.btn_runtime_exit): widget.setEnabled(False)
        rf.addWidget(QtWidgets.QLabel("Новая цель"), 0, 0); rf.addWidget(self.runtime_target, 0, 1); rf.addWidget(self.btn_runtime_target, 0, 2, 1, 2)
        rf.addWidget(QtWidgets.QLabel("Задать"), 1, 0); rf.addWidget(self.runtime_pressure, 1, 1); rf.addWidget(self.btn_runtime_pressure, 1, 2, 1, 2)
        rf.addWidget(QtWidgets.QLabel("Шаг, В"), 2, 0); rf.addWidget(self.runtime_step_v, 2, 1); rf.addWidget(self.btn_runtime_step_down, 2, 2); rf.addWidget(self.btn_runtime_step_up, 2, 3)
        rf.addWidget(self.btn_runtime_return, 3, 0, 1, 2); rf.addWidget(self.btn_runtime_softdown, 3, 2, 1, 2); rf.addWidget(self.btn_runtime_exit, 4, 0, 1, 4)
        root.addWidget(self.runtime_box)

        self._config_controls = [self.mode, self.meta_toggle, self.preset_pass, self.control_toggle, self.control_dt, self.use_alignment,
            self.prep_mode, self.pressure_bar, self.manual_step_v, self.ordinary_target, self.stepped_target, self.load_step,
            self.periodic_direction, self.periodic_target, self.scan_interval, self.cycle_target, self.stabilize_sec,
            self.hold_target, self.hold_duration, self.hold_scans, self.hold_interval, self.repeat_target, self.repeat_scans,
            self.repeat_interval, self.load_route, self.route_hold, self.route_scans, self.program_text, self.adaptive_start,
            self.adaptive_final, self.adaptive_step, self.adaptive_focus_start, self.adaptive_focus_end, self.adaptive_fine_step,
            self.adaptive_hold, self.adaptive_scans, self.ramp_final, self.ramp_rate, self.ramp_scan_every]
        self._update_mode()

    def _page(self, key: str):
        page = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(page); form.setContentsMargins(3, 4, 3, 4); form.setHorizontalSpacing(5); form.setVerticalSpacing(4)
        self.page_widgets[key] = page; self.page_forms[key] = form
        return page, form

    def _number_field(self, key: str, minimum: float, maximum: float, decimals: int, suffix: str, value: float):
        field = self._double(minimum, maximum, decimals, suffix, value); self._fields[key] = field; return field

    def _int_field(self, key: str, minimum: int, maximum: int, value: int):
        field = QtWidgets.QSpinBox(); field.setRange(minimum, maximum); field.setValue(value); field.setKeyboardTracking(False); self._fields[key] = field; return field

    @staticmethod
    def _double(minimum, maximum, decimals, suffix, value=0.0):
        widget = QtWidgets.QDoubleSpinBox(); widget.setRange(minimum, maximum); widget.setDecimals(decimals); widget.setSingleStep(0.1)
        widget.setSuffix(suffix); widget.setKeyboardTracking(False); widget.setValue(value); return widget

    @staticmethod
    def _collapsible(root, title: str, checked=False):
        button = QtWidgets.QToolButton(); button.setText(title); button.setCheckable(True); button.setChecked(checked)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly); button.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        box = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(box); form.setContentsMargins(6, 4, 4, 4); form.setHorizontalSpacing(5); form.setVerticalSpacing(3)
        root.addWidget(button); root.addWidget(box); box.setVisible(checked)
        button.toggled.connect(box.setVisible); button.toggled.connect(lambda on, b=button: b.setArrowType(QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow))
        return button, box, form

    @staticmethod
    def _icon_button(icon_name: str, accessible_name: str, tooltip: str, color: str | None = None) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(); button.setObjectName("pressActionButton"); button.setFixedSize(32, 32); button.setIconSize(QtCore.QSize(17, 17))
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly); button.setIcon(resource_svg_icon(icon_name, widget=button, color=color, size=17))
        button.setToolTip(tooltip); button.setAccessibleName(accessible_name); button.setAutoRaise(False); return button

    def _update_mode(self, *_):
        key = self.mode.currentData(); self.pages.setCurrentWidget(self.page_widgets.get(key, self.prep_page))
        needs_preset = key in {"stepped", "periodic", "cycle", "hold", "repeat", "load_unload", "programmed", "adaptive", "ramp"}
        needs_control = key not in {"prep"}
        self.meta_toggle.setVisible(needs_preset); self.meta_box.setVisible(needs_preset and self.meta_toggle.isChecked())
        self.control_toggle.setVisible(needs_control); self.control_box.setVisible(needs_control and self.control_toggle.isChecked())
        self.cycle_preset_note.setVisible(key == "cycle")
        if key == "cycle": self.preset_pass.setValue(1000); self.preset_pass.setEnabled(False)
        else: self.preset_pass.setEnabled(not self._locked)
        prep = key == "prep"; prep_mode = self.prep_mode.currentData()
        self.pressure_bar.setVisible(prep and prep_mode == "pressure"); self.manual_step_v.setVisible(prep and prep_mode in {"align", "manual"})
        self.prep_hint.setVisible(prep and prep_mode == "align")
        if key == "ramp": self.status.setText("МАКЕТ: режим скорости у коллеги помечен как симуляционный.")
        else: self.status.setText("МАКЕТ: контроллер TRS_UMLM_CS и оборудование не подключены.")
        if not self._locked:
            self.pressure_bar.setEnabled(prep and prep_mode == "pressure"); self.manual_step_v.setEnabled(prep and prep_mode in {"align", "manual"})

    def set_acquisition_locked(self, locked: bool):
        """Блокирует параметры во время наблюдения; команды оборудования не выполняются."""
        self._locked = bool(locked)
        for widget in self._config_controls: widget.setEnabled(not self._locked)
        self.btn_start.setEnabled(False); self.btn_pause.setEnabled(False); self.btn_resume.setEnabled(False); self.btn_stop.setEnabled(False); self.btn_emergency.setEnabled(False)
        if not self._locked: self._update_mode()

    def collect_config(self) -> dict:
        result = {"mode": self.mode.currentData(), "prep_mode": self.prep_mode.currentData(), "mcs_passes": self.preset_pass.value(),
                  "control_dt_s": self.control_dt.value(), "use_alignment": self.use_alignment.isChecked(), "pressure_bar": self.pressure_bar.value(),
                  "manual_step_v": self.manual_step_v.value(), "ordinary_target_kg": self.ordinary_target.value(), "stepped_target_kg": self.stepped_target.value(),
                  "load_step_kg": self.load_step.value(), "periodic_direction": self.periodic_direction.currentData(),
                  "periodic_target_kg": self.periodic_target.value(), "scan_interval_s": self.scan_interval.value(), "cycle_target_kg": self.cycle_target.value(),
                  "stabilize_sec": self.stabilize_sec.value(), "hold_target_kg": self.hold_target.value(), "hold_duration_s": self.hold_duration.value(),
                  "hold_scans": self.hold_scans.value(), "hold_interval_s": self.hold_interval.value(), "repeat_target_kg": self.repeat_target.value(),
                  "repeat_scans": self.repeat_scans.value(), "repeat_interval_s": self.repeat_interval.value(), "load_route_kg": self.load_route.text().strip(),
                  "route_hold_s": self.route_hold.value(), "route_scans": self.route_scans.value(), "program_steps": self.program_text.toPlainText().strip(), "runtime_step_v": self.runtime_step_v.value(),
                  "adaptive_start_kg": self.adaptive_start.value(), "adaptive_final_kg": self.adaptive_final.value(), "adaptive_step_kg": self.adaptive_step.value(),
                  "adaptive_focus_start_kg": self.adaptive_focus_start.value(), "adaptive_focus_end_kg": self.adaptive_focus_end.value(),
                  "adaptive_fine_step_kg": self.adaptive_fine_step.value(), "adaptive_hold_s": self.adaptive_hold.value(), "adaptive_scans": self.adaptive_scans.value(),
                  "ramp_final_kg": self.ramp_final.value(), "ramp_rate_kg_min": self.ramp_rate.value(), "ramp_scan_every_kg": self.ramp_scan_every.value()}
        return result

    def apply_config(self, config: dict) -> None:
        if not isinstance(config, dict): raise TypeError("Раздел параметров пресса в JSON должен быть объектом.")
        index = self.mode.findData(config.get("mode"))
        if index >= 0: self.mode.setCurrentIndex(index)
        index = self.prep_mode.findData(config.get("prep_mode"))
        if index >= 0: self.prep_mode.setCurrentIndex(index)
        index = self.periodic_direction.findData(config.get("periodic_direction"))
        if index >= 0: self.periodic_direction.setCurrentIndex(index)
        field_map = {"mcs_passes": self.preset_pass, "control_dt_s": self.control_dt, "pressure_bar": self.pressure_bar,
                     "manual_step_v": self.manual_step_v, "ordinary_target_kg": self.ordinary_target, "stepped_target_kg": self.stepped_target,
                     "load_step_kg": self.load_step, "periodic_target_kg": self.periodic_target, "scan_interval_s": self.scan_interval,
                     "cycle_target_kg": self.cycle_target, "stabilize_sec": self.stabilize_sec, "hold_target_kg": self.hold_target,
                     "hold_duration_s": self.hold_duration, "hold_scans": self.hold_scans, "hold_interval_s": self.hold_interval,
                     "repeat_target_kg": self.repeat_target, "repeat_scans": self.repeat_scans, "repeat_interval_s": self.repeat_interval,
                     "route_hold_s": self.route_hold, "route_scans": self.route_scans, "runtime_step_v": self.runtime_step_v, "adaptive_start_kg": self.adaptive_start,
                     "adaptive_final_kg": self.adaptive_final, "adaptive_step_kg": self.adaptive_step, "adaptive_focus_start_kg": self.adaptive_focus_start,
                     "adaptive_focus_end_kg": self.adaptive_focus_end, "adaptive_fine_step_kg": self.adaptive_fine_step,
                     "adaptive_hold_s": self.adaptive_hold, "adaptive_scans": self.adaptive_scans, "ramp_final_kg": self.ramp_final,
                     "ramp_rate_kg_min": self.ramp_rate, "ramp_scan_every_kg": self.ramp_scan_every}
        for key, widget in field_map.items():
            if key in config:
                try: widget.setValue(float(config[key]))
                except (TypeError, ValueError): pass
        if "load_route_kg" in config: self.load_route.setText(str(config["load_route_kg"]))
        if "program_steps" in config: self.program_text.setPlainText(str(config["program_steps"]))
        if "use_alignment" in config: self.use_alignment.setChecked(bool(config["use_alignment"]))
        self._update_mode()
