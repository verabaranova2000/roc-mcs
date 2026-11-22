"""Small, live schematic of the exact spatial and spectral peak-detector stages."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QSize, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QPalette, QPolygonF
from PyQt6.QtWidgets import QWidget, QSizePolicy


class PeakDetectionPreview(QWidget):
    """A deterministic toy neighbourhood passed through the production detector.

    The preview is intentionally labelled schematic. It uses the same local
    neighbourhood averaging, background-noise estimator and peak detector as
    the scientific operation, but its synthetic data and fixed geometry are
    not a prediction for any selected experimental pixel.
    """

    _N_SAMPLES = 256
    _GRID_SIZE = 15
    _CENTER = _GRID_SIZE // 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = {
            "window_size": 3,
            "smooth_window": 11,
            "smooth_polyorder": 2,
            "min_distance": 6,
            "prominence_mode": "sigma",
            "prominence_value": 3.0,
            "min_relative_prominence": 0.07,
        }
        self._demo_data = self._make_demo_volume()
        self._raw = np.empty(self._N_SAMPLES, dtype=float)
        self._smooth = self._raw.copy()
        self._peaks = np.empty(0, dtype=int)
        self._valid = np.empty(0, dtype=bool)
        self._prominences = np.empty(0, dtype=float)
        self._threshold = 0.0
        self._noise_sigma = float("nan")
        self._relative_cutoff = float("nan")
        self._error = ""
        self._recalculate()
        self.setMinimumSize(300, 142)
        self.setMaximumHeight(168)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(
            "Схематическая синтетическая ROC-кривая. Локальная кривая строится "
            "из пространственной окрестности N×N; затем применяются тот же "
            "Savitzky–Golay и тот же detect_peaks, что в научном этапе. "
            "Иллюстрация не является прогнозом для выбранного пикселя скана."
        )

    @classmethod
    def _make_demo_volume(cls) -> np.ndarray:
        """Build a small reproducible Y×X×theta field with spatially varying peaks."""
        theta_index = np.arange(cls._N_SAMPLES, dtype=float)
        volume = np.empty(
            (cls._GRID_SIZE, cls._GRID_SIZE, cls._N_SAMPLES), dtype=np.float64
        )
        peak_specs = (
            (34.0, 12.0, 5.5),
            (82.0, 20.0, 9.5),
            (145.0, 8.0, 4.5),
            (178.0, 15.0, 7.0),
            (187.0, 6.2, 3.4),
            (224.0, 10.0, 5.0),
        )
        for y in range(cls._GRID_SIZE):
            dy = y - cls._CENTER
            for x in range(cls._GRID_SIZE):
                dx = x - cls._CENTER
                signal = 2.0 + 0.0015 * theta_index
                spatial_shift = 0.80 * dx + 0.56 * dy
                spatial_scale = 1.0 + 0.012 * dx - 0.009 * dy
                for center, height, width in peak_specs:
                    local_center = center + spatial_shift * (1.0 if center < 160 else 0.65)
                    local_height = height * spatial_scale
                    local_width = width * (1.0 + 0.004 * (dx + dy))
                    signal += local_height * np.exp(
                        -0.5 * ((theta_index - local_center) / local_width) ** 2
                    )
                # Deterministic spatially varying texture, not random noise.
                phase = 0.035 * dx + 0.027 * dy
                signal += 0.28 * np.sin(0.71 * theta_index + phase)
                signal += 0.17 * np.sin(1.83 * theta_index + 0.4 - phase)
                signal += 0.025 * np.sin(0.13 * theta_index + 0.11 * dx)
                volume[y, x] = signal
        return volume

    def sizeHint(self) -> QSize:
        return QSize(420, 150)

    def set_parameters(self, **parameters) -> None:
        changed = False
        for key in self._config:
            if key in parameters and self._config[key] != parameters[key]:
                self._config[key] = parameters[key]
                changed = True
        if changed:
            self._recalculate()
            self.update()

    def _recalculate(self) -> None:
        try:
            from roc_mcs.peaks.detection import PeakDetectionConfig, detect_peaks
            from roc_mcs.peaks.peak_signal import local_neighbourhood_mean_curve
            from roc_mcs.processing.background import estimate_background_stats

            # Match the production path: background sigma from theta×Y×X, then
            # exact local-neighbourhood averaging, then Savitzky–Golay/find_peaks.
            data_tyx = np.moveaxis(self._demo_data, -1, 0)
            background = estimate_background_stats(data_tyx)
            self._noise_sigma = float(background.sigma)
            self._raw = local_neighbourhood_mean_curve(
                self._demo_data,
                self._CENTER,
                self._CENTER,
                int(self._config["window_size"]),
            )

            mode = str(self._config["prominence_mode"])
            prominence_value = float(self._config["prominence_value"])
            config = PeakDetectionConfig(
                smooth_window=int(self._config["smooth_window"]),
                smooth_polyorder=int(self._config["smooth_polyorder"]),
                min_distance=int(self._config["min_distance"]),
                prominence_sigma=prominence_value if mode == "sigma" else None,
                prominence_fraction=prominence_value if mode == "fraction" else None,
                min_relative_prominence=float(self._config["min_relative_prominence"]),
            )
            smooth, peaks, properties, valid = detect_peaks(
                self._raw, config, noise_sigma=self._noise_sigma
            )
            self._smooth = np.asarray(smooth, dtype=float)
            self._peaks = np.asarray(peaks, dtype=int)
            self._valid = np.asarray(valid, dtype=bool)
            self._prominences = np.asarray(properties.get("prominences", ()), dtype=float)
            self._threshold = (
                prominence_value * self._noise_sigma
                if mode == "sigma"
                else prominence_value * float(np.ptp(self._smooth))
            )
            self._relative_cutoff = (
                float(self._config["min_relative_prominence"])
                * float(np.max(self._prominences))
                if self._prominences.size else 0.0
            )
            self._error = ""
        except Exception as exc:
            self._raw = self._demo_data[self._CENTER, self._CENTER].copy()
            self._smooth = self._raw.copy()
            self._peaks = np.empty(0, dtype=int)
            self._valid = np.empty(0, dtype=bool)
            self._prominences = np.empty(0, dtype=float)
            self._threshold = 0.0
            self._relative_cutoff = 0.0
            self._error = str(exc)

    @staticmethod
    def _value_y(value: float, rect: QRectF, low: float, high: float) -> float:
        span = max(1e-12, high - low)
        return rect.bottom() - (float(value) - low) / span * rect.height()

    @classmethod
    def _polyline(
        cls, painter: QPainter, values: np.ndarray, rect: QRectF,
        low: float, high: float, pen: QPen,
    ) -> None:
        if values.size < 2:
            return
        points = QPolygonF([
            QPointF(
                rect.left() + index / (values.size - 1) * rect.width(),
                cls._value_y(float(value), rect, low, high),
            )
            for index, value in enumerate(values)
        ])
        painter.setPen(pen)
        painter.drawPolyline(points)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        text = palette.color(QPalette.ColorRole.Text)
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        grid = palette.color(QPalette.ColorRole.Mid)
        accent = palette.color(QPalette.ColorRole.Highlight)
        accepted = QColor("#36b37e")
        rejected = QColor("#d69e36")
        painter.fillRect(self.rect(), background)

        width, height = self.width(), self.height()
        right_panel_width = max(96, min(132, int(width * 0.30)))
        plot = QRectF(
            28, 24, max(125, width - right_panel_width - 45), max(64, height - 52)
        )
        all_values = np.concatenate((self._raw, self._smooth))
        finite_values = all_values[np.isfinite(all_values)]
        low = float(np.min(finite_values)) if finite_values.size else 0.0
        high = float(np.max(finite_values)) if finite_values.size else 1.0
        span = max(1e-9, high - low)
        low -= 0.08 * span
        high += 0.12 * span

        painter.setPen(text)
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSizeF(max(7.0, title_font.pointSizeF() - 0.5))
        painter.setFont(title_font)
        painter.drawText(
            QRectF(8, 3, width - 16, 17),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "СХЕМА ПОИСКА · синтетическая ROC",
        )

        painter.setPen(QPen(grid, 0.6, Qt.PenStyle.DotLine))
        for fraction in (0.25, 0.5, 0.75):
            y = plot.top() + fraction * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        for fraction in (0.25, 0.5, 0.75):
            x = plot.left() + fraction * plot.width()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.setPen(QPen(grid, 0.8))
        painter.drawRect(plot)

        self._polyline(
            painter, self._raw, plot, low, high,
            QPen(muted, 1.0, Qt.PenStyle.SolidLine),
        )
        self._polyline(
            painter, self._smooth, plot, low, high,
            QPen(accent, 1.8, Qt.PenStyle.SolidLine),
        )

        # Mark the vertical prominence of the strongest candidate: prominence
        # is measured relative to its local contour, not from y=0.
        if self._peaks.size and self._prominences.size:
            best = int(np.argmax(self._prominences))
            peak_index = int(self._peaks[best])
            x_peak = plot.left() + peak_index / (self._N_SAMPLES - 1) * plot.width()
            y_peak = self._value_y(self._smooth[peak_index], plot, low, high)
            y_base = self._value_y(
                self._smooth[peak_index] - self._prominences[best], plot, low, high
            )
            painter.setPen(QPen(rejected, 1.1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x_peak, y_peak), QPointF(x_peak, y_base))
            painter.drawLine(QPointF(x_peak - 3, y_base), QPointF(x_peak + 3, y_base))

        # Show a local theta interval illustrating min_distance.
        if self._peaks.size:
            center = int(self._peaks[len(self._peaks) // 2])
            distance = max(1, int(self._config["min_distance"]))
            left = plot.left() + max(0, center - distance / 2) / (self._N_SAMPLES - 1) * plot.width()
            right = plot.left() + min(self._N_SAMPLES - 1, center + distance / 2) / (self._N_SAMPLES - 1) * plot.width()
            painter.setPen(QPen(rejected, 0.8, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(left, plot.bottom() - 2), QPointF(right, plot.bottom() - 2))

        y_span = max(1e-12, high - low)
        for index, peak in enumerate(self._peaks):
            px = plot.left() + int(peak) / (self._N_SAMPLES - 1) * plot.width()
            py = plot.bottom() - (float(self._smooth[peak]) - low) / y_span * plot.height()
            is_accepted = index < self._valid.size and bool(self._valid[index])
            color = accepted if is_accepted else rejected
            painter.setPen(QPen(color, 1.3))
            painter.setBrush(color if is_accepted else background)
            painter.drawEllipse(QPointF(px, py), 3.4, 3.4)

        painter.setPen(muted)
        label_font = painter.font()
        label_font.setPointSizeF(max(6.5, label_font.pointSizeF() - 1.0))
        painter.setFont(label_font)
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 3, plot.width(), 15),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "θ-отсчёты →",
        )

        info_x = plot.right() + 12
        info_w = max(72, width - info_x - 7)
        painter.setPen(text)
        painter.drawText(
            QRectF(info_x, 24, info_w, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._config['window_size']}×{self._config['window_size']} px",
        )
        painter.setPen(muted)
        painter.drawText(
            QRectF(info_x, 40, info_w, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"локальная ROC · σn={self._noise_sigma:.2g}",
        )
        painter.setPen(text)
        painter.drawText(
            QRectF(info_x, 56, info_w, 15),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"SG: {self._config['smooth_window']} / p{self._config['smooth_polyorder']}",
        )
        painter.drawText(
            QRectF(info_x, 72, info_w, 15),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Δmin: {self._config['min_distance']} θ",
        )
        painter.drawText(
            QRectF(info_x, 88, info_w, 15),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Pmin: {self._threshold:.2g}",
        )
        n_candidates = int(self._peaks.size)
        n_accepted = int(np.count_nonzero(self._valid))
        painter.setPen(accepted)
        painter.drawText(
            QRectF(info_x, 102, info_w, 15),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Относит.: ≥{100.0 * self._config['min_relative_prominence']:.0f}%",
        )
        painter.setPen(accepted)
        painter.drawText(
            QRectF(info_x, 118, info_w, 15),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Пиков: {n_accepted}/{n_candidates}",
        )
        if self._error:
            painter.setPen(QColor("#c65d5d"))
            painter.drawText(
                QRectF(8, height - 17, width - 16, 14),
                Qt.AlignmentFlag.AlignLeft,
                "Схема недоступна: " + self._error[:90],
            )
        else:
            painter.setPen(muted)
            painter.drawText(
                QRectF(8, height - 16, width - 16, 13),
                Qt.AlignmentFlag.AlignLeft,
                "серый ROC · цветной SG · ● принят / ○ отсечён",
            )
        painter.end()
