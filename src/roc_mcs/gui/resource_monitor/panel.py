"""Компактная всплывающая панель показаний ресурсов компьютера."""
from __future__ import annotations
from PyQt6 import QtCore, QtWidgets
from .telemetry import GPUSnapshot, ResourceSnapshot, format_gib, format_pct
from .service import ResourceMonitorService

class ResourceMonitorPanel(QtWidgets.QDialog):
    def __init__(self, service: ResourceMonitorService, parent=None):
        super().__init__(parent); self._service = service; self._colors = None
        self.setObjectName("resourceMonitorPanel"); self.setWindowFlags(QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setFixedWidth(360); self.setMinimumHeight(230)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(13, 11, 13, 11); root.setSpacing(8)
        header = QtWidgets.QHBoxLayout(); title = QtWidgets.QLabel("СИСТЕМНЫЕ РЕСУРСЫ", self); title.setObjectName("resourceTitle"); header.addWidget(title); header.addStretch(1)
        self.state_label = QtWidgets.QLabel("Нет данных", self); self.state_label.setObjectName("resourceState"); header.addWidget(self.state_label); root.addLayout(header)
        self.ram_label = self._add_metric(root, "Доступная физическая RAM")
        self.ram_bar = QtWidgets.QProgressBar(self); self.ram_bar.setRange(0, 100); self.ram_bar.setTextVisible(False); self.ram_bar.setFixedHeight(6); self.ram_bar.setObjectName("resourceBar"); root.addWidget(self.ram_bar)
        self.commit_label = self._add_metric(root, "Запас system commit")
        self.app_label = self._add_metric(root, "TRS Apps · RSS процесса")
        self.private_label = self._add_metric(root, "Private commit процесса")
        self.cpu_label = self._add_metric(root, "CPU · система / процесс")
        self.gui_label = self._add_metric(root, "Задержка Qt event loop · сейчас / пик за сеанс")
        self.process_counts_label = self._add_metric(root, "Потоки / handles процесса")
        self.gpu_label = self._add_metric(root, "GPU · общий адаптер"); self.gpu_label.setWordWrap(True); self.gpu_label.setMaximumWidth(205)
        self.gpu_note = QtWidgets.QLabel("VRAM относится ко всему адаптеру; учёт Windows WDDM может быть неполным.", self); self.gpu_note.setObjectName("resourceFooter"); self.gpu_note.setWordWrap(True); root.addWidget(self.gpu_note)
        self.footer = QtWidgets.QLabel("RAM/CPU: 2 с · GPU: 5 с только при открытой панели. Порог ориентировочный.", self); self.footer.setObjectName("resourceFooter"); self.footer.setWordWrap(True); root.addWidget(self.footer)
        self._gpu_timer = QtCore.QTimer(self); self._gpu_timer.setInterval(5000); self._gpu_timer.timeout.connect(self._service.request_gpu_sample)
        self._service.snapshotChanged.connect(self.update_snapshot); self._service.gpuChanged.connect(self.update_gpu)
        self._service.sample_now()
        self.apply_theme({"window": "#11151b", "surface": "#171c24", "text": "#d8e0ec", "muted": "#98a5b6", "border": "#303744", "accent": "#8da9c7"})

    def _add_metric(self, layout, name: str):
        row = QtWidgets.QHBoxLayout(); left = QtWidgets.QLabel(name, self); left.setObjectName("resourceMetricName"); right = QtWidgets.QLabel("н/д", self); right.setObjectName("resourceMetricValue"); right.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter); row.addWidget(left, 1); row.addWidget(right); layout.addLayout(row); return right

    def apply_theme(self, colors: dict):
        self._colors = colors
        self.setStyleSheet(f"""
        QDialog#resourceMonitorPanel {{ background:{colors['surface']}; color:{colors['text']}; border:1px solid {colors['border']}; border-radius:6px; }}
        QLabel#resourceTitle {{ color:{colors['text']}; font-size:9pt; font-weight:700; letter-spacing:0.5px; }}
        QLabel#resourceMetricName {{ color:{colors['muted']}; font-size:8.5pt; }}
        QLabel#resourceMetricValue {{ color:{colors['text']}; font-size:8.5pt; font-weight:600; }}
        QLabel#resourceState {{ color:{colors['accent']}; font-size:8pt; font-weight:700; }}
        QLabel#resourceFooter {{ color:{colors['muted']}; font-size:7.5pt; }}
        QProgressBar#resourceBar {{ border:0; border-radius:3px; background:{colors['border']}; }}
        QProgressBar#resourceBar::chunk {{ border-radius:3px; background:{colors['accent']}; }}
        """)

    @QtCore.pyqtSlot(object)
    def update_snapshot(self, sample: ResourceSnapshot):
        states = {"normal": ("ЗАПАС ЕСТЬ", "#32966f"), "warning": ("ЗАПАС СНИЖЕН", "#d69b35"), "critical": ("КРИТИЧЕСКИ МАЛО", "#d34f55"), "unknown": ("НЕТ ДАННЫХ", "#8792a2")}
        label, color = states.get(sample.state, states["unknown"]); self.state_label.setText(label); self.state_label.setStyleSheet(f"color:{color};font-weight:700;")
        if sample.total_ram and sample.available_ram is not None:
            used = max(0, min(100, round(100 * (sample.total_ram - sample.available_ram) / sample.total_ram)))
            self.ram_label.setText(f"доступно {format_gib(sample.available_ram)} / {format_gib(sample.total_ram)}")
            self.ram_bar.setValue(used)
        else:
            self.ram_label.setText("н/д"); self.ram_bar.setValue(0)
        self.commit_label.setText(f"доступно {format_gib(sample.commit_available)} / {format_gib(sample.commit_limit)}")
        if sample.process_rss is not None:
            self.app_label.setText(f"RSS {format_gib(sample.process_rss)} · пик {format_gib(sample.process_peak_rss)}")
        else: self.app_label.setText(sample.error or "н/д")
        self.private_label.setText(format_gib(sample.process_private_commit))
        self.cpu_label.setText(f"{format_pct(sample.system_cpu_pct)} / {format_pct(sample.process_cpu_pct)}")
        self.gui_label.setText(f"{sample.gui_event_delay_ms:.0f} / {sample.gui_event_delay_peak_ms:.0f} мс" if sample.gui_event_delay_ms is not None and sample.gui_event_delay_peak_ms is not None else "н/д")
        self.process_counts_label.setText(f"{sample.process_threads if sample.process_threads is not None else 'н/д'} / {sample.process_handles if sample.process_handles is not None else 'н/д'}")
        self.footer.setText(f"Обновлено {sample.sampled_at} · CPU процесса может быть выше 100% на нескольких ядрах. Монитор наблюдает, но не ограничивает ресурсы.")

    @QtCore.pyqtSlot(object)
    def update_gpu(self, sample: GPUSnapshot):
        if not sample.devices:
            self.gpu_label.setText(sample.message or "нет данных"); return
        parts = []
        for gpu in sample.devices:
            util = format_pct(gpu.utilization_pct); used = format_gib(gpu.memory_used); total = format_gib(gpu.memory_total)
            parts.append(f"GPU {gpu.index}: {util} · {used}/{total}")
        self.gpu_label.setText(" | ".join(parts))
        self.gpu_note.setText("GPU/VRAM — показатели всего адаптера, не только TRS Apps. В Windows WDDM учёт памяти отдельных процессов может быть недоступен.")

    def showEvent(self, event):
        super().showEvent(event); self._service.request_gpu_sample(); self._gpu_timer.start()

    def hideEvent(self, event):
        self._gpu_timer.stop(); super().hideEvent(event)
