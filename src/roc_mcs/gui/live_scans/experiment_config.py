"""Компактные метаданные образца и источника; без управления приборами."""
from __future__ import annotations
import re
from PyQt6 import QtCore, QtWidgets
from roc_mcs.gui.components.icon_provider import resource_svg_icon

class ExperimentConfigWidget(QtWidgets.QWidget):
    saveProfileRequested = QtCore.pyqtSignal()
    loadProfileRequested = QtCore.pyqtSignal()
    configurationChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(5)
        form = QtWidgets.QFormLayout(); form.setContentsMargins(0, 0, 0, 0); form.setHorizontalSpacing(6); form.setVerticalSpacing(5)
        self.sample_name = QtWidgets.QLineEdit(); self.sample_name.setPlaceholderText("Название / ID образца")
        self.sample_name.setToolTip("Уникальный идентификатор или короткое название образца.")
        dims = QtWidgets.QWidget(); dr = QtWidgets.QHBoxLayout(dims); dr.setContentsMargins(0, 0, 0, 0); dr.setSpacing(4)
        self.width_mm = self._number(0, 1000, 3, 76, "Ширина образца в миллиметрах.")
        self.length_mm = self._number(0, 1000, 3, 76, "Длина образца в миллиметрах.")
        for label, field in (("Ш", self.width_mm), ("Д", self.length_mm)):
            dr.addWidget(QtWidgets.QLabel(label)); dr.addWidget(field); dr.addWidget(QtWidgets.QLabel("мм"))
        dr.addStretch(1)
        tube = QtWidgets.QWidget(); tr = QtWidgets.QHBoxLayout(tube); tr.setContentsMargins(0, 0, 0, 0); tr.setSpacing(4)
        self.tube_voltage_kv = self._number(0, 200, 1, 66, "Напряжение рентгеновской трубки, кВ.")
        self.tube_current_ma = self._number(0, 5000, 2, 66, "Ток рентгеновской трубки, мА.")
        tr.addWidget(QtWidgets.QLabel("U")); tr.addWidget(self.tube_voltage_kv); tr.addWidget(QtWidgets.QLabel("кВ"))
        tr.addSpacing(5); tr.addWidget(QtWidgets.QLabel("I")); tr.addWidget(self.tube_current_ma); tr.addWidget(QtWidgets.QLabel("мА")); tr.addStretch(1)
        self.excitation_tag = QtWidgets.QLineEdit(); self.excitation_tag.setPlaceholderText("Доп. метка режима, если нужна")
        self.excitation_tag.setToolTip("Свободная метка для протокола. Не влияет на фазу MCS и масштаб θ.")
        self.sample_note = QtWidgets.QPlainTextEdit(); self.sample_note.setPlaceholderText("Заметка оператора: трещина, хруст, контакт…")
        self.sample_note.setMaximumBlockCount(30); self.sample_note.setFixedHeight(48); self.sample_note.setToolTip("Наблюдения экспериментатора о состоянии образца во время опыта.")
        form.addRow("Образец", self.sample_name); form.addRow("Размеры", dims); form.addRow("Источник X-ray", tube)
        form.addRow("Метка", self.excitation_tag); form.addRow("Заметка", self.sample_note); root.addLayout(form)
        actions = QtWidgets.QHBoxLayout(); actions.setContentsMargins(0, 0, 0, 0); actions.setSpacing(4)
        self.btn_save_profile = QtWidgets.QPushButton("Сохранить профиль"); self.btn_load_profile = QtWidgets.QPushButton("Загрузить профиль")
        self.btn_save_profile.setIcon(resource_svg_icon("save_icon.svg", widget=self.btn_save_profile, size=14))
        self.btn_load_profile.setIcon(resource_svg_icon("folder_open.svg", widget=self.btn_load_profile, size=14))
        self.btn_save_profile.clicked.connect(self.saveProfileRequested); self.btn_load_profile.clicked.connect(self.loadProfileRequested)
        actions.addWidget(self.btn_save_profile); actions.addWidget(self.btn_load_profile); root.addLayout(actions)
        for signal in (self.sample_name.textChanged, self.tube_voltage_kv.valueChanged, self.tube_current_ma.valueChanged,
                       self.excitation_tag.textChanged, self.sample_note.textChanged, self.width_mm.valueChanged, self.length_mm.valueChanged):
            signal.connect(lambda *_: self.configurationChanged.emit())

    @staticmethod
    def _number(minimum: float, maximum: float, decimals: int, width: int, tooltip: str) -> QtWidgets.QDoubleSpinBox:
        field = QtWidgets.QDoubleSpinBox(); field.setRange(minimum, maximum); field.setDecimals(decimals)
        field.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons); field.setSpecialValueText(""); field.setFixedWidth(width)
        field.setKeyboardTracking(False); field.setSingleStep(0.1); field.setToolTip(tooltip); field.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        return field

    def collect_config(self) -> dict:
        voltage, current = self.tube_voltage_kv.value(), self.tube_current_ma.value()
        voltage_value = voltage if voltage > 0 else None; current_value = current if current > 0 else None
        source = f"{voltage:g}kV_{current:g}mA" if voltage_value is not None and current_value is not None else ""
        return {"sample_name": self.sample_name.text().strip(), "width_mm": self.width_mm.value(), "length_mm": self.length_mm.value(),
                "tube_voltage_kv": voltage_value, "tube_current_ma": current_value,
                "xray_source": source, "excitation_tag": self.excitation_tag.text().strip(),
                "sample_note": self.sample_note.toPlainText().strip()}

    def apply_config(self, config: dict) -> None:
        if not isinstance(config, dict): raise TypeError("Раздел образца в JSON должен быть объектом.")
        self.sample_name.setText(str(config.get("sample_name", "")))
        self.width_mm.setValue(float(config.get("width_mm", 0) or 0)); self.length_mm.setValue(float(config.get("length_mm", 0) or 0))
        source = str(config.get("xray_source", config.get("tube_mode", "")) or "")
        match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*kV.*?([0-9]+(?:[.,][0-9]+)?)\s*mA", source, re.I)
        if "tube_voltage_kv" in config: self.tube_voltage_kv.setValue(float(config["tube_voltage_kv"] or 0))
        elif match: self.tube_voltage_kv.setValue(float(match.group(1).replace(",", ".")))
        if "tube_current_ma" in config: self.tube_current_ma.setValue(float(config["tube_current_ma"] or 0))
        elif match: self.tube_current_ma.setValue(float(match.group(2).replace(",", ".")))
        self.excitation_tag.setText(str(config.get("excitation_tag", config.get("frequency_tag", "")) or ""))
        self.sample_note.setPlainText(str(config.get("sample_note", "") or ""))
