"""Компактная закрепляемая панель показаний ресурсов компьютера."""
from __future__ import annotations
from PyQt6 import QtCore, QtWidgets
from .activity import ActivityStatus
from .telemetry import GPUSnapshot, ResourceSnapshot, format_gib, format_pct
from .service import ResourceMonitorService

class _DragHeader(QtWidgets.QFrame):
    def __init__(self, panel):
        super().__init__(panel); self._panel = panel; self._offset = None
        self.setObjectName("resourceHeader"); self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._offset = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor); event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._offset is not None and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._offset); self._panel._user_moved = True; event.accept(); return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        self._offset = None; self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor); super().mouseReleaseEvent(event)

class ResourceMonitorPanel(QtWidgets.QDialog):
    def __init__(self, service: ResourceMonitorService, parent=None):
        super().__init__(parent); self._service = service; self._colors = None; self._user_moved = False
        self.setObjectName("resourceMonitorPanel"); self.setWindowFlags(QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setModal(False); self.setWindowModality(QtCore.Qt.WindowModality.NonModal); self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False); self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedWidth(370); self.setMinimumHeight(245)
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(12, 9, 12, 9); root.setSpacing(7)
        header = _DragHeader(self); header_layout = QtWidgets.QHBoxLayout(header); header_layout.setContentsMargins(1, 0, 0, 0); header_layout.setSpacing(6)
        title = QtWidgets.QLabel("СИСТЕМНЫЕ РЕСУРСЫ", header); title.setObjectName("resourceTitle"); title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True); header_layout.addWidget(title); header_layout.addStretch(1)
        self.state_label = QtWidgets.QLabel("Нет данных", header); self.state_label.setObjectName("resourceState"); self.state_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True); header_layout.addWidget(self.state_label)
        self.close_button = QtWidgets.QToolButton(header); self.close_button.setObjectName("resourceCloseButton"); self.close_button.setText("×"); self.close_button.setFixedSize(20, 20); self.close_button.setToolTip("Закрыть монитор"); self.close_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus); self.close_button.clicked.connect(self.close); header_layout.addWidget(self.close_button); root.addWidget(header)
        self.ram_label = self._add_metric(root, "Доступная физическая RAM")
        self.ram_bar = QtWidgets.QProgressBar(self); self.ram_bar.setRange(0, 100); self.ram_bar.setTextVisible(False); self.ram_bar.setFixedHeight(6); self.ram_bar.setObjectName("resourceBar"); root.addWidget(self.ram_bar)
        self.commit_label = self._add_metric(root, "Запас system commit")
        self.app_label = self._add_metric(root, "TRS Apps · RSS процесса")
        self.private_label = self._add_metric(root, "Private commit процесса")
        self.cpu_label = self._add_metric(root, "CPU · система / процесс")
        self.gui_label = self._add_metric(root, "Задержка Qt event loop · сейчас / пик")
        self.process_counts_label = self._add_metric(root, "Потоки / handles процесса")
        self.gpu_label = self._add_metric(root, "GPU / VRAM · NVIDIA (если доступно)")
        self.gpu_label.setWordWrap(True)
        self.activity_label = QtWidgets.QLabel("Нет активных операций", self); self.activity_label.setObjectName("resourceActivity"); self.activity_label.setWordWrap(True); root.addWidget(self.activity_label)
        self.gpu_note = QtWidgets.QLabel("NVIDIA NVSMI — необязательный источник. Его отсутствие не означает отсутствие GPU; AMD/Intel пока не измеряются.", self); self.gpu_note.setObjectName("resourceFooter"); self.gpu_note.setWordWrap(True); root.addWidget(self.gpu_note)
        self.footer = QtWidgets.QLabel("RAM/CPU: 2 с · GPU: 5 с только при открытой панели. Порог ориентировочный.", self); self.footer.setObjectName("resourceFooter"); self.footer.setWordWrap(True); root.addWidget(self.footer)
        self._gpu_timer = QtCore.QTimer(self); self._gpu_timer.setInterval(5000); self._gpu_timer.timeout.connect(self._service.request_gpu_sample)
        self._service.snapshotChanged.connect(self.update_snapshot); self._service.gpuChanged.connect(self.update_gpu); self._service.activityChanged.connect(self.update_activities)
        self._service.sample_now()
        self.apply_theme({"window": "#11151b", "surface": "#171c24", "text": "#d8e0ec", "muted": "#98a5b6", "border": "#303744", "accent": "#8da9c7"})

    @property
    def user_moved(self) -> bool: return self._user_moved

    def place_near(self, anchor: QtCore.QPoint):
        """Ставит окно у кнопки только до ручного перемещения пользователем."""
        self.adjustSize(); screen = QtWidgets.QApplication.screenAt(anchor) or QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry(); x = min(max(area.left(), anchor.x()), max(area.left(), area.right() - self.width() + 1)); y = anchor.y()
            if y + self.height() > area.bottom() + 1: y = anchor.y() - self.height() - 38
            y = min(max(area.top(), y), max(area.top(), area.bottom() - self.height() + 1)); anchor = QtCore.QPoint(x, y)
        self.move(anchor)

    def _add_metric(self, layout, name: str):
        row = QtWidgets.QHBoxLayout(); left = QtWidgets.QLabel(name, self); left.setObjectName("resourceMetricName"); right = QtWidgets.QLabel("н/д", self); right.setObjectName("resourceMetricValue"); right.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter); row.addWidget(left, 1); row.addWidget(right); layout.addLayout(row); return right

    def apply_theme(self, colors: dict):
        self._colors = colors
        self.setStyleSheet(f"""
        QDialog#resourceMonitorPanel {{ background:{colors['surface']}; color:{colors['text']}; border:1px solid {colors['border']}; border-radius:6px; }}
        QFrame#resourceHeader {{ background:transparent; }}
        QLabel#resourceTitle {{ color:{colors['text']}; font-size:9pt; font-weight:700; letter-spacing:0.5px; }}
        QLabel#resourceMetricName {{ color:{colors['muted']}; font-size:8.5pt; }}
        QLabel#resourceMetricValue {{ color:{colors['text']}; font-size:8.5pt; font-weight:600; }}
        QLabel#resourceState {{ color:{colors['accent']}; font-size:8pt; font-weight:700; }}
        QLabel#resourceActivity {{ color:{colors['text']}; font-size:8pt; padding:5px 6px; border:1px solid {colors['border']}; border-radius:3px; }}
        QLabel#resourceFooter {{ color:{colors['muted']}; font-size:7.5pt; }}
        QToolButton#resourceCloseButton {{ color:{colors['muted']}; background:transparent; border:0; border-radius:3px; font-size:15px; }}
        QToolButton#resourceCloseButton:hover {{ color:{colors['text']}; background:{colors['border']}; }}
        QProgressBar#resourceBar {{ border:0; border-radius:3px; background:{colors['border']}; }}
        QProgressBar#resourceBar::chunk {{ border-radius:3px; background:{colors['accent']}; }}
        """)

    @QtCore.pyqtSlot(object)
    def update_snapshot(self, sample: ResourceSnapshot):
        states = {"normal": ("ЗАПАС ЕСТЬ", "#32966f"), "warning": ("ЗАПАС СНИЖЕН", "#d69b35"), "critical": ("КРИТИЧЕСКИ МАЛО", "#d34f55"), "unknown": ("НЕТ ДАННЫХ", "#8792a2")}
        label, color = states.get(sample.state, states["unknown"]); self.state_label.setText(label); self.state_label.setStyleSheet(f"color:{color};font-weight:700;")
        if sample.total_ram and sample.available_ram is not None:
            used = max(0, min(100, round(100 * (sample.total_ram - sample.available_ram) / sample.total_ram)))
            self.ram_label.setText(f"доступно {format_gib(sample.available_ram)} / {format_gib(sample.total_ram)}"); self.ram_bar.setValue(used)
        else: self.ram_label.setText("н/д"); self.ram_bar.setValue(0)
        self.commit_label.setText(f"доступно {format_gib(sample.commit_available)} / {format_gib(sample.commit_limit)}")
        self.app_label.setText(f"RSS {format_gib(sample.process_rss)} · пик {format_gib(sample.process_peak_rss)}" if sample.process_rss is not None else sample.error or "н/д")
        self.private_label.setText(format_gib(sample.process_private_commit)); self.cpu_label.setText(f"{format_pct(sample.system_cpu_pct)} / {format_pct(sample.process_cpu_pct)}")
        self.gui_label.setText(f"{sample.gui_event_delay_ms:.0f} / {sample.gui_event_delay_peak_ms:.0f} мс" if sample.gui_event_delay_ms is not None and sample.gui_event_delay_peak_ms is not None else "н/д")
        self.process_counts_label.setText(f"{sample.process_threads if sample.process_threads is not None else 'н/д'} / {sample.process_handles if sample.process_handles is not None else 'н/д'}")
        self.footer.setText(f"Обновлено {sample.sampled_at} · CPU процесса может быть выше 100% на нескольких ядрах. Монитор наблюдает, но не ограничивает ресурсы.")

    @QtCore.pyqtSlot(object)
    def update_activities(self, status: ActivityStatus):
        if status.active:
            lines = [f"Сейчас: {item.name} · {item.elapsed_seconds:.1f} с" for item in status.active[:2]]
            if len(status.active) > 2: lines.append(f"Ещё операций: {len(status.active) - 2}")
        elif status.last_name:
            lines = [f"Последняя: {status.last_name} · {status.last_duration_seconds:.2f} с"]
        else: lines = ["Нет активных операций"]
        self.activity_label.setText("\n".join(lines))

    @QtCore.pyqtSlot(object)
    def update_gpu(self, sample: GPUSnapshot):
        if not sample.devices:
            self.gpu_label.setText("н/д · NVIDIA telemetry unavailable")
            self.gpu_note.setText(sample.message or "NVIDIA NVSMI недоступен. Это не подтверждает отсутствие GPU; RAM/CPU измеряются независимо.")
            return
        parts = []
        for gpu in sample.devices:
            util = format_pct(gpu.utilization_pct); used = format_gib(gpu.memory_used); total = format_gib(gpu.memory_total)
            parts.append(f"GPU {gpu.index} · {gpu.name}: {util} · {used}/{total}")
        self.gpu_label.setText(" | ".join(parts)); self.gpu_note.setText("GPU/VRAM — показатели всего адаптера, не только TRS Apps. В Windows WDDM память отдельного процесса может быть недоступна.")

    def showEvent(self, event):
        super().showEvent(event); self._service.request_gpu_sample(); self._gpu_timer.start()

    def hideEvent(self, event):
        self._gpu_timer.stop(); super().hideEvent(event)
