"""Compact, theme-aware result view for Kα doublet / HKL identification.

This is strictly a presentation layer. Histogram binning, zoom and scrolling
never modify the scientific artifact or rerun the HKL calculation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from roc_mcs.gui.roi_explorer.hkl_plot_utils import (
    build_histogram_edges,
    histogram_view_bounds,
    suggest_histogram_bin_width,
    comparison_axis_parameters,
)
from roc_mcs.gui.roi_explorer.theme import theme_colors
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR
from roc_mcs.gui.components.icon_provider import svg_icon


class HklResultsPanel(QWidget):
    """Render an immutable HKL preview using bounded, reusable plot geometry."""

    _DEFAULT_BIN_WIDTH_ARCSEC = 0.25
    _MAX_HISTOGRAM_BINS = 1200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hklResultsPanel")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._result = None
        self._theme = "light"
        self._measurement_values_by_pair: dict[str, np.ndarray] = {}
        self._reflection_x = np.empty(0, dtype=np.float64)
        self._ranking_by_pair: dict[str, pd.DataFrame] = {}
        self._histogram_cache: dict[str, Any] | None = None
        self._histogram_axes = None
        self._histogram_full_xlim: tuple[float, float] | None = None
        self._histogram_full_ylim: tuple[float, float] | None = None
        self._histogram_view_limits: tuple[float, float, float, float] | None = None
        self._histogram_drag_state: dict[str, Any] | None = None
        self._selected_reflection_row = -1
        self._reflection_catalog = pd.DataFrame()
        self._rank_label_artists = []
        self._selected_reflection_artist = None
        self._selected_reflection_marker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        self.summary_label = QLabel("Результат появится после выполнения этапа.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_label.setObjectName("hklSummary")
        root.addWidget(self.summary_label)

        pair_row = QHBoxLayout()
        pair_row.setContentsMargins(0, 0, 0, 0)
        pair_row.setSpacing(6)
        pair_label = QLabel("Пара ветвей:", self)
        pair_label.setObjectName("hklControlLabel")
        pair_row.addWidget(pair_label)

        self.pair_combo = QComboBox(self)
        self.pair_combo.setObjectName("hklPairSelector")
        self.pair_combo.setMinimumWidth(120)
        self.pair_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.pair_combo.setToolTip(
            "Выберите пару ветвей для просмотра экспериментального распределения "
            "и теоретических рефлексов."
        )
        self.pair_combo.currentIndexChanged.connect(self._on_pair_changed)
        pair_row.addWidget(self.pair_combo, 1)

        bin_label = QLabel("Шаг бина:", self)
        bin_label.setObjectName("hklControlLabel")
        pair_row.addWidget(bin_label)
        self.bin_width_spin = QDoubleSpinBox(self)
        self.bin_width_spin.setObjectName("hklHistogramBinWidth")
        self.bin_width_spin.setRange(0.01, 100.0)
        self.bin_width_spin.setDecimals(2)
        self.bin_width_spin.setSingleStep(0.05)
        self.bin_width_spin.setValue(self._DEFAULT_BIN_WIDTH_ARCSEC)
        self.bin_width_spin.setSuffix(" ″")
        self.bin_width_spin.setMinimumWidth(76)
        self.bin_width_spin.setMaximumWidth(88)
        self.bin_width_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.bin_width_spin.setKeyboardTracking(False)
        self.bin_width_spin.setToolTip(
            "Ширина интервалов только для отображения. Не меняет измерения, "
            "теоретические отражения, ранжирование или сохранённый артефакт."
        )
        self.bin_width_spin.valueChanged.connect(self._on_bin_width_changed)
        pair_row.addWidget(self.bin_width_spin)
        self.bin_width_effective_label = QLabel("", self)
        self.bin_width_effective_label.setObjectName("hklControlLabel")
        self.bin_width_effective_label.setToolTip(
            "Фактическая ширина бина после применения ограничения на число интервалов."
        )
        pair_row.addWidget(self.bin_width_effective_label)
        root.addLayout(pair_row)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("hklResultTabs")
        self.tabs.setMinimumHeight(0)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root.addWidget(self.tabs)

        self.pair_tab = QWidget(self.tabs)
        pair_layout = QVBoxLayout(self.pair_tab)
        pair_layout.setContentsMargins(1, 1, 1, 1)
        pair_layout.setSpacing(2)
        plot_toolbar = QHBoxLayout()
        plot_toolbar.setContentsMargins(2, 0, 2, 0)
        plot_toolbar.setSpacing(4)
        self.plot_hint_label = QLabel(
            "Колесо — масштаб · перетаскивание — сдвиг · Ctrl — θ · Shift — Y",
            self.pair_tab,
        )
        self.plot_hint_label.setObjectName("hklPlotHint")
        self.plot_hint_label.setToolTip(
            "Колесо мыши: приблизить/отдалить обе оси вокруг курсора. "
            "Ctrl + колесо или перетаскивание: только θ; Shift: только частота Y. "
            "Перетаскивание левой кнопкой перемещает диапазон по обеим осям."
        )
        plot_toolbar.addWidget(self.plot_hint_label, 1)
        self.reset_view_button = QToolButton(self.pair_tab)
        self.reset_view_button.setObjectName("hklResetViewButton")
        self.reset_view_button.setAutoRaise(True)
        self.reset_view_button.setFixedSize(23, 21)
        self.reset_view_button.setIcon(svg_icon(ICON_DIR / "zoom_out_map.svg", widget=self.reset_view_button))
        self.reset_view_button.setIconSize(QSize(14, 14))
        self.reset_view_button.setToolTip("Сбросить масштаб и положение обеих осей")
        self.reset_view_button.setAccessibleName("Сбросить масштаб гистограммы HKL")
        self.reset_view_button.clicked.connect(self._reset_histogram_view)
        plot_toolbar.addWidget(self.reset_view_button, 0, Qt.AlignmentFlag.AlignRight)
        pair_layout.addLayout(plot_toolbar)

        self.pair_chart_figure = Figure(figsize=(5.2, 2.42), dpi=96)
        self.pair_canvas = FigureCanvasQTAgg(self.pair_chart_figure)
        self.pair_canvas.setMinimumHeight(220)
        self.pair_canvas.setMaximumHeight(258)
        self.pair_canvas.setMinimumWidth(0)
        self.pair_canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        pair_layout.addWidget(self.pair_canvas)
        self._histogram_event_ids = [
            self.pair_canvas.mpl_connect("scroll_event", self._on_histogram_scroll),
            self.pair_canvas.mpl_connect("button_press_event", self._on_histogram_press),
            self.pair_canvas.mpl_connect("button_release_event", self._on_histogram_release),
            self.pair_canvas.mpl_connect("motion_notify_event", self._on_histogram_motion),
        ]

        self.pair_stats_label = QLabel("", self.pair_tab)
        self.pair_stats_label.setWordWrap(True)
        self.pair_stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.pair_stats_label.setObjectName("hklPairStatistics")
        pair_layout.addWidget(self.pair_stats_label)

        self.reflection_table = self._make_table(
            ["#", "HKL", "d, Å", "θ₁, °", "Δθтеор, ″", "|Δ−med|, ″", "|F_H|²", "Darwin σ/π, ″"],
            [34, 64, 70, 70, 82, 82, 74, 96],
            max_height=202,
        )
        self.reflection_table.currentCellChanged.connect(self._on_reflection_row_changed)
        pair_layout.addWidget(self.reflection_table)
        self.tabs.addTab(self.pair_tab, "Пара и рефлексы")

        self.all_pairs_tab = QWidget(self.tabs)
        all_layout = QVBoxLayout(self.all_pairs_tab)
        all_layout.setContentsMargins(1, 1, 1, 1)
        all_layout.setSpacing(2)
        self.all_chart_figure = Figure(figsize=(5.2, 2.22), dpi=96)
        self.all_canvas = FigureCanvasQTAgg(self.all_chart_figure)
        self.all_canvas.setMinimumHeight(210)
        self.all_canvas.setMaximumHeight(238)
        self.all_canvas.setMinimumWidth(0)
        self.all_canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        all_layout.addWidget(self.all_canvas)
        self.pair_table = self._make_table(
            ["Пара", "Назнач. / uniq.", "Медиана, ″", "Q10–Q90, ″", "HKL", "Ошибка, ″", "Статус"],
            [68, 96, 84, 96, 58, 80, 98],
            max_height=176,
        )
        all_layout.addWidget(self.pair_table)
        self.tabs.addTab(self.all_pairs_tab, "Сравнение пар")

        self.method_note = QLabel(
            "Интервал — Q10–Q90 измерений; круг — медиана; ромб — лучшее теоретическое Δθ. "
            "Ранг в графике рефлексов соответствует первому столбцу таблицы. "
            "Ранжирование — геометрическое совпадение, не вероятность и не оценка интенсивности.",
            self,
        )
        self.method_note.setWordWrap(True)
        self.method_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.method_note.setObjectName("hklMethodNote")
        root.addWidget(self.method_note)

        self.apply_theme(self._theme)
        self.set_result(None)

    @staticmethod
    def _make_table(headers: list[str], widths: list[int], *, max_height: int) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)
        table.setWordWrap(False)
        table.setShowGrid(False)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(19)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setMinimumSectionSize(34)
        header.setSectionsMovable(False)
        header.setStretchLastSection(False)
        for index, width in enumerate(widths):
            header.resizeSection(index, int(width))
        table.setMinimumWidth(0)
        table.setMinimumHeight(72)
        table.setMaximumHeight(max_height)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        font = QFont(table.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        table.setFont(font)
        header_font = QFont(font)
        header_font.setWeight(QFont.Weight.DemiBold)
        header.setFont(header_font)
        return table

    @property
    def has_result(self) -> bool:
        """Whether a validated HKL result is currently attached to the panel."""
        return self._result is not None

    @staticmethod
    def _fmt(value: Any, digits: int = 2) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return "—"
        return f"{number:.{digits}f}" if np.isfinite(number) else "—"

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return float("nan")
        return number if np.isfinite(number) else float("nan")

    @staticmethod
    def _count_int(value: Any, default: int = 0) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return int(default)
        return int(number) if np.isfinite(number) else int(default)

    @staticmethod
    def _set_table(table: QTableWidget, rows: list[list[str]]) -> None:
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(rows))
            for row_index, values in enumerate(rows):
                for column_index, value in enumerate(values):
                    text = str(value)
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if column_index == 0:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if text not in {"", "—"}:
                        item.setToolTip(text)
                    table.setItem(row_index, column_index, item)
                table.setRowHeight(row_index, 19)
        finally:
            table.setUpdatesEnabled(True)

    def apply_theme(self, theme: str | None = None) -> None:
        self._theme = str(theme or self._theme).lower()
        colors = theme_colors(self._theme)
        self.setStyleSheet(
            f"QLabel#hklSummary {{ color:{colors['text']}; font-size:9px; font-weight:600; }}"
            f"QLabel#hklControlLabel, QLabel#hklMethodNote {{ color:{colors['muted']}; font-size:8px; }}"
            f"QLabel#hklPairStatistics {{ color:{colors['text']}; font-size:8px; }}"
            f"QComboBox#hklPairSelector, "
            f"QDoubleSpinBox#hklHistogramBinWidth {{ color:{colors['text']}; background:{colors['base']}; "
            f"border:1px solid {colors['border']}; min-height:20px; max-height:22px; "
            f"padding:0px 3px; font-size:9px; }}"
            f"QComboBox#hklPairSelector QAbstractItemView {{ "
            f"background:{colors['base']}; color:{colors['text']}; selection-background-color:{colors['highlight']}; }}"
            f"QTableWidget {{ background:{colors['base']}; color:{colors['text']}; "
            f"alternate-background-color:{colors['alternate']}; gridline-color:{colors['border']}; "
            f"selection-background-color:{colors['highlight']}; selection-color:{colors['highlight_text']}; "
            f"border:1px solid {colors['border']}; font-size:8px; }}"
            f"QHeaderView::section {{ background:{colors['button']}; color:{colors['text']}; "
            f"border:0px; border-right:1px solid {colors['border']}; "
            f"border-bottom:1px solid {colors['border']}; padding:2px 3px; font-size:8px; font-weight:600; }}"
            f"QTabWidget::pane {{ border:1px solid {colors['border']}; top:-1px; }}"
            f"QTabBar::tab {{ background:{colors['button']}; color:{colors['text']}; "
            f"border:1px solid {colors['border']}; padding:3px 8px; font-size:8px; }}"
            f"QTabBar::tab:selected {{ background:{colors['base']}; border-bottom-color:{colors['base']}; }}"
            f"QLabel#hklPlotHint {{ color:{colors['muted']}; font-size:7px; }}"
            f"QToolButton#hklResetViewButton {{ border:1px solid transparent; border-radius:2px; padding:0px; }}"
            f"QToolButton#hklResetViewButton:hover {{ background:{colors['button']}; border:1px solid {colors['border']}; }}"
        )
        self.reset_view_button.setIcon(svg_icon(
            ICON_DIR / "zoom_out_map.svg", color=colors["text"], widget=self.reset_view_button
        ))
        if self._result is None:
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Ожидается результат HKL")
            self._draw_empty(self.all_chart_figure, self.all_canvas, "Нет данных для сравнения пар")
        else:
            self._render_all_pairs()
            if self.pair_combo.currentIndex() >= 0:
                self._render_selected_pair(reset_view=False)
            else:
                self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет измеренных пар")

    def set_result(self, result: Any | None) -> None:
        self._result = result
        self._histogram_cache = None
        self._measurement_values_by_pair.clear()
        self._ranking_by_pair.clear()
        self._reflection_x = np.empty(0, dtype=np.float64)

        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        if result is None:
            self.summary_label.setText("Результат появится после выполнения этапа.")
            self.pair_stats_label.setText("")
            self._set_table(self.reflection_table, [])
            self._set_table(self.pair_table, [])
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет результата HKL")
            self._draw_empty(self.all_chart_figure, self.all_canvas, "Нет результата HKL")
            self.pair_combo.setEnabled(False)
            self.bin_width_spin.setEnabled(False)
            self.bin_width_effective_label.setText("")
            self._histogram_view_limits = None
            self._histogram_axes = None
            self._histogram_full_xlim = None
            self._histogram_full_ylim = None
            self._selected_reflection_row = -1
            self._reflection_catalog = pd.DataFrame()
            self._rank_label_artists = []
            self._selected_reflection_artist = None
            self._selected_reflection_marker = None
            self.pair_combo.blockSignals(False)
            return

        metadata = result.metadata or {}
        stats = result.pair_statistics
        measurements = result.measurements
        ranking = result.pair_ranking
        reflections = result.theoretical_reflections

        # Cache only the small view arrays once. The scientific DataFrames stay
        # owned by the read-only preview model; no display code mutates them.
        for pair_id, frame in measurements.groupby("pair_id", sort=False):
            values = pd.to_numeric(
                frame["delta_theta_abs_arcsec"], errors="coerce"
            ).to_numpy(dtype=np.float64, na_value=np.nan)
            self._measurement_values_by_pair[str(pair_id)] = values[np.isfinite(values)]
        # Ranking rows are already sorted by ascending rank in the immutable
        # artifact. The UI only displays the first 15; retaining the full
        # pair×reflection product in the panel needlessly multiplies memory.
        for pair_id, frame in ranking.groupby("pair_id", sort=False):
            ordered = frame.sort_values("rank", ascending=True, kind="mergesort")
            self._ranking_by_pair[str(pair_id)] = ordered.head(15)

        raw_reflection_x = pd.to_numeric(
            reflections["delta_theta_arcsec"], errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        self._reflection_x = np.unique(raw_reflection_x[np.isfinite(raw_reflection_x)])

        for row in stats.itertuples(index=False):
            count = int(getattr(row, "matched_assignment_count", 0))
            self.pair_combo.addItem(f"{row.pair_id} · N={count}", str(row.pair_id))
        self.pair_combo.setEnabled(self.pair_combo.count() > 0)
        self.bin_width_spin.setEnabled(self.pair_combo.count() > 0)
        self._histogram_view_limits = None
        self._selected_reflection_row = -1
        self._reflection_catalog = reflections.loc[:, [
            name for name in ("hkl", "delta_theta_arcsec", "theta_ka1_deg", "d_A")
            if name in reflections.columns
        ]].copy(deep=True)
        all_measurements = (
            np.concatenate(tuple(self._measurement_values_by_pair.values()))
            if self._measurement_values_by_pair else np.empty(0, dtype=np.float64)
        )
        suggested_width = suggest_histogram_bin_width(
            all_measurements, fallback_width=self._DEFAULT_BIN_WIDTH_ARCSEC
        )
        bin_signals_blocked = self.bin_width_spin.blockSignals(True)
        self.bin_width_spin.setValue(round(suggested_width, 2))
        self.bin_width_spin.blockSignals(bin_signals_blocked)

        cell_description = metadata.get("unit_cell_description")
        if not cell_description:
            cell_parameters = metadata.get("unit_cell_parameters_A") or {}
            cell_angles = metadata.get("unit_cell_angles_deg") or {}
            if all(key in cell_parameters for key in ("a", "b", "c")) and all(
                key in cell_angles for key in ("alpha", "beta", "gamma")
            ):
                cell_description = (
                    f"a={self._fmt(cell_parameters['a'], 5)}, "
                    f"b={self._fmt(cell_parameters['b'], 5)}, "
                    f"c={self._fmt(cell_parameters['c'], 5)} Å; "
                    f"α={self._fmt(cell_angles['alpha'], 3)}°, "
                    f"β={self._fmt(cell_angles['beta'], 3)}°, "
                    f"γ={self._fmt(cell_angles['gamma'], 3)}°"
                )
            else:
                cell_description = f"a={self._fmt(metadata.get('lattice_a_A'), 5)} Å"
        self.summary_label.setText(
            f"{metadata.get('crystal_name', 'Кристалл')} · {cell_description} · "
            f"λ(Kα₁)={self._fmt(metadata.get('lambda_ka1_A'), 7)} Å; "
            f"λ(Kα₂)={self._fmt(metadata.get('lambda_ka2_A'), 7)} Å\n"
            f"Рефлексов: {len(reflections)} · отброшено: {len(result.rejected_reflections)} · "
            f"измерений: {len(measurements)} · пар: {len(stats)}"
        )
        self.pair_combo.blockSignals(False)
        self._render_all_pairs()
        if self.pair_combo.count():
            self._render_selected_pair(reset_view=True)
        else:
            self.pair_stats_label.setText("Нет пар с принятыми соответствиями.")
            self._set_table(self.reflection_table, [])
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет измеренных пар")

    def _draw_empty(self, figure: Figure, canvas: FigureCanvasQTAgg, message: str) -> None:
        if figure is self.pair_chart_figure:
            self._histogram_axes = None
            self._histogram_full_xlim = None
            self._histogram_full_ylim = None
            self._histogram_view_limits = None
            self._histogram_drag_state = None
            self._rank_label_artists = []
            self._selected_reflection_artist = None
            self._selected_reflection_marker = None
            self._selected_reflection_row = -1
            self.plot_hint_label.setText(
                "Колесо — масштаб · перетаскивание — сдвиг · Ctrl — θ · Shift — Y"
            )
        figure.clear()
        ax = figure.add_subplot(111)
        colors = theme_colors(self._theme)
        figure.patch.set_facecolor(colors["canvas"])
        ax.set_facecolor(colors["axes"])
        ax.text(
            0.5, 0.5, message, ha="center", va="center",
            color=colors["muted"], fontsize=8, transform=ax.transAxes,
        )
        ax.set_axis_off()
        canvas.draw_idle()

    def _on_pair_changed(self, _index: int) -> None:
        self._selected_reflection_row = -1
        self._render_selected_pair(reset_view=True)

    def _on_bin_width_changed(self, _value: float) -> None:
        self._render_selected_pair(reset_view=True)

    def _render_selected_pair(self, _value: Any = None, *, reset_view: bool = False) -> None:
        if self._result is None or self.pair_combo.currentIndex() < 0:
            return
        pair_id = str(self.pair_combo.currentData())
        stats_rows = self._result.pair_statistics
        selected = stats_rows.loc[stats_rows["pair_id"].astype(str) == pair_id]
        if selected.empty:
            return
        stats = selected.iloc[0]
        values = self._measurement_values_by_pair.get(pair_id, np.empty(0, dtype=np.float64))
        ranking = self._ranking_by_pair.get(pair_id, pd.DataFrame())
        self._prepare_pair_histogram(pair_id, stats, values, ranking)

        if str(stats["identification_status"]) == "ready" and not ranking.empty:
            best = ranking.iloc[0]
            assignment_count = int(stats["matched_assignment_count"])
            unique_partner_count = int(stats.get("unique_partner_peak_count", assignment_count))
            reused_partner_count = int(stats.get("reused_partner_assignment_count", 0))
            self.pair_stats_label.setText(
                f"Назначений: {assignment_count:,} · уникальных source/partner peak_id: "
                f"{int(stats.get('unique_source_peak_count', assignment_count))}/{unique_partner_count} · "
                f"повторных partner: {reused_partner_count} · "
                f"median |Δθ|={self._fmt(stats['delta_theta_abs_median_arcsec'], 2)}″ · "
                f"Q10–Q90={self._fmt(stats.get('delta_theta_abs_q10_arcsec', np.nan), 2)}–"
                f"{self._fmt(stats.get('delta_theta_abs_q90_arcsec', np.nan), 2)}″ · "
                f"MAD={self._fmt(stats.get('delta_theta_abs_mad_arcsec', np.nan), 2)}″ · "
                f"лучшее геометрическое совпадение: {best['hkl']} · "
                f"Δθтеор={self._fmt(best['delta_theta_arcsec'], 2)}″ · "
                f"|ошибка|={self._fmt(best['difference_from_exp_median_arcsec'], 2)}″"
            )
        else:
            self.pair_stats_label.setText(
                f"Для пары {pair_id} найдено {int(stats['matched_assignment_count'])} соответствий; "
                "для ранжирования требуется не менее двух назначений."
            )

        previous_selected_row = (
            self.reflection_table.currentRow() if not reset_view else -1
        )
        rows: list[list[str]] = []
        for row in ranking.head(15).itertuples(index=False):
            structure_factor_sq = getattr(row, "FH_sq", np.nan)
            if not np.isfinite(pd.to_numeric(structure_factor_sq, errors="coerce")):
                structure_factor_sq = getattr(row, "F_H_sq_at_ka1", np.nan)
            rows.append([
                str(int(row.rank)), str(row.hkl), self._fmt(row.d_A, 4),
                self._fmt(row.theta_ka1_deg, 3), self._fmt(row.delta_theta_arcsec, 2),
                self._fmt(row.difference_from_exp_median_arcsec, 2),
                self._fmt(structure_factor_sq, 3),
                f"{self._fmt(getattr(row, 'darwin_sigma_arcsec', np.nan), 3)} / "
                f"{self._fmt(getattr(row, 'darwin_pi_arcsec', np.nan), 3)}",
            ])
        self._set_table(self.reflection_table, rows)
        self.reflection_table.clearSelection()
        self.reflection_table.setCurrentCell(-1, -1)
        self._selected_reflection_row = -1
        if reset_view:
            self._histogram_view_limits = None
        elif 0 <= previous_selected_row < self.reflection_table.rowCount():
            self.reflection_table.setCurrentCell(previous_selected_row, 0)
        self._draw_histogram_view()

    def _prepare_pair_histogram(
        self, pair_id: str, stats: pd.Series, values: np.ndarray, ranking: pd.DataFrame
    ) -> None:
        if values.size == 0:
            self._histogram_cache = None
            self.bin_width_effective_label.setText("")
            self._histogram_axes = None
            self._histogram_full_xlim = None
            self._histogram_full_ylim = None
            self._histogram_view_limits = None
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет конечных измеренных Δθ")
            return

        edges = build_histogram_edges(
            values,
            float(self.bin_width_spin.value()),
            max_bins=self._MAX_HISTOGRAM_BINS,
        )
        counts, edges = np.histogram(values, bins=edges)
        actual_widths = np.diff(edges)
        requested_width = float(self.bin_width_spin.value())
        effective_width = float(np.median(actual_widths))
        widened_for_display_cap = (
            actual_widths.size >= self._MAX_HISTOGRAM_BINS
            and effective_width > requested_width * (1.0 + 1e-9)
        )
        if widened_for_display_cap:
            self.bin_width_effective_label.setText(
                f"факт. {self._fmt(effective_width, 2)}″ · лимит бинов"
            )
            self.bin_width_effective_label.setToolTip(
                f"Запрошено {requested_width:g}″. Для ограничения числа интервалов "
                f"до {self._MAX_HISTOGRAM_BINS} фактическая ширина ≈ {effective_width:g}″. "
                "Измерения и научный результат не изменяются."
            )
        else:
            self.bin_width_effective_label.setText("")
            self.bin_width_effective_label.setToolTip(
                "Фактическая ширина совпадает с запрошенной (с точностью округления)."
            )
        full_bounds = histogram_view_bounds(values, edges, nonnegative=True)
        ranking_x = pd.to_numeric(
            ranking.get("delta_theta_arcsec", pd.Series(dtype=float)), errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        ranking_ranks = pd.to_numeric(
            ranking.get("rank", pd.Series(dtype=float)), errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        best_x = float(ranking_x[0]) if ranking_x.size and np.isfinite(ranking_x[0]) else float("nan")

        self._histogram_cache = {
            "pair_id": pair_id,
            "counts": counts,
            "edges": edges,
            "full_bounds": full_bounds,
            "max_count": max(1.0, float(counts.max(initial=0))),
            "reflection_x": self._reflection_x,
            "ranking_x": ranking_x,
            "ranking_ranks": ranking_ranks,
            "best_x": best_x,
            "median": self._as_float(stats.get("delta_theta_abs_median_arcsec")),
            "mean": self._as_float(stats.get("delta_theta_abs_mean_arcsec")),
        }
        self._histogram_full_xlim = tuple(map(float, full_bounds))
        self._histogram_full_ylim = (0.0, max(1.0, float(counts.max(initial=0))) * 1.48)

    def _draw_histogram_view(self, _value: int | None = None) -> None:
        cache = self._histogram_cache
        if cache is None:
            return

        figure = self.pair_chart_figure
        figure.clear()
        self._rank_label_artists = []
        self._selected_reflection_artist = None
        self._selected_reflection_marker = None
        ax = figure.add_subplot(111)
        self._histogram_axes = ax
        self._histogram_drag_state = None
        colors = theme_colors(self._theme)
        figure.patch.set_facecolor(colors["canvas"])
        ax.set_facecolor(colors["axes"])

        counts = cache["counts"]
        edges = cache["edges"]
        max_count = float(cache["max_count"])
        histogram_color = "#6689A6" if self._theme == "dark" else "#7896AE"
        theory_color = "#E08A74"
        ax.stairs(
            counts, edges, baseline=0,
            fill=True, color=histogram_color,
            edgecolor=colors["border"], linewidth=0.45, alpha=0.78, zorder=2,
            label="Измеренные назначения",
        )

        median = float(cache["median"])
        mean = float(cache["mean"])
        if np.isfinite(median):
            ax.axvline(
                median, color=colors["roi"], linestyle="--", linewidth=1.15,
                label=f"Медиана {median:.2f}″", zorder=5,
            )
        if np.isfinite(mean):
            ax.axvline(
                mean, color="#4D9A7A", linestyle="-.", linewidth=0.95,
                label=f"Среднее {mean:.2f}″", zorder=4,
            )

        full_low, full_high = cache["full_bounds"]
        full_y_low, full_y_high = self._histogram_full_ylim or (0.0, max_count * 1.48)
        limits = self._histogram_view_limits
        if limits is None:
            view_low, view_high, view_y_low, view_y_high = full_low, full_high, full_y_low, full_y_high
        else:
            view_low, view_high, view_y_low, view_y_high = limits
            view_low, view_high = self._clamp_interval(view_low, view_high, full_low, full_high)
            view_y_low, view_y_high = self._clamp_interval(view_y_low, view_y_high, full_y_low, full_y_high)
            self._histogram_view_limits = (view_low, view_high, view_y_low, view_y_high)

        reflection_x = np.asarray(cache["reflection_x"], dtype=np.float64)
        left_index = int(np.searchsorted(reflection_x, view_low, side="left"))
        right_index = int(np.searchsorted(reflection_x, view_high, side="right"))
        visible_reflections = reflection_x[left_index:right_index]
        if visible_reflections.size:
            visible_reflections = np.unique(visible_reflections)
            axes_width_px = max(1, int(ax.get_position().width * figure.get_figwidth() * figure.dpi))
            if visible_reflections.size > axes_width_px * 2:
                pixel_index = np.floor(
                    (visible_reflections - view_low) / max(view_high - view_low, np.finfo(float).eps) * axes_width_px
                ).astype(np.int64)
                _, first_index = np.unique(pixel_index, return_index=True)
                visible_reflections = visible_reflections[np.sort(first_index)]
            ax.vlines(
                visible_reflections, 0.0, max_count * 1.30,
                color=colors["muted"], linewidth=0.48, alpha=0.56, zorder=3,
                rasterized=True, label="Теоретические рефлексы",
            )

        best_x = float(cache["best_x"])
        if np.isfinite(best_x) and view_low <= best_x <= view_high:
            ax.axvline(
                best_x, color=theory_color, linestyle="-", linewidth=1.35,
                label=f"Лучший HKL {best_x:.2f}″", zorder=6,
            )

        ax.set_ylim(view_y_low, view_y_high)
        ax.set_xlim(view_low, view_high)
        ax.set_xlabel("Абсолютное экспериментальное разделение |Δθ|, ″", fontsize=7.5, labelpad=2)
        ax.set_ylabel("Частота", fontsize=7.5, labelpad=2)
        ax.set_title("Δθ: измерения и теоретические положения", fontsize=8, pad=3)
        ax.grid(axis="y", color=colors["grid"], linewidth=0.5, alpha=0.32, zorder=0)
        ax.set_axisbelow(True)
        legend = ax.legend(
            frameon=False, fontsize=5.9, loc="upper center",
            bbox_to_anchor=(0.5, -0.29), borderaxespad=0.0,
            ncol=3, handlelength=1.45, columnspacing=0.9, labelspacing=0.35,
        )
        if legend is not None:
            for label in legend.get_texts():
                label.set_color(colors["text"])

        y_formatter = ScalarFormatter(useMathText=True)
        y_formatter.set_scientific(True)
        y_formatter.set_powerlimits((0, 0))
        y_formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.yaxis.get_offset_text().set_color(colors["text"])
        ax.tick_params(axis="both", labelsize=6.8, colors=colors["text"], length=2.2, pad=1.8)
        for spine in ax.spines.values():
            spine.set_color(colors["border"])
            spine.set_linewidth(0.6)
        ax.xaxis.label.set_color(colors["text"])
        ax.yaxis.label.set_color(colors["text"])
        ax.title.set_color(colors["text"])
        figure.subplots_adjust(left=0.085, right=0.992, bottom=0.27, top=0.81)
        self._rank_label_artists = self._add_rank_labels(
            ax, cache["ranking_x"], cache["ranking_ranks"],
            view_low, view_high, label_color=colors["text"],
        )
        self._draw_selected_reflection_overlay()
        self.pair_canvas.draw_idle()

    @staticmethod
    def _clamp_interval(low: float, high: float, bound_low: float, bound_high: float) -> tuple[float, float]:
        """Clamp a finite viewport while preserving its span where possible."""
        low, high = float(low), float(high)
        if not (np.isfinite(low) and np.isfinite(high) and high > low):
            return float(bound_low), float(bound_high)
        bound_span = float(bound_high - bound_low)
        span = min(high - low, bound_span)
        if low < bound_low:
            low = bound_low
            high = low + span
        if high > bound_high:
            high = bound_high
            low = high - span
        return float(low), float(high)

    @staticmethod
    def _event_modifier(event, modifier: str) -> bool:
        key = str(getattr(event, "key", "") or "").lower()
        return modifier in key

    def _on_histogram_scroll(self, event) -> None:
        ax = self._histogram_axes
        if ax is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        x_only = self._event_modifier(event, "control") or self._event_modifier(event, "ctrl")
        y_only = self._event_modifier(event, "shift")
        if x_only and y_only:
            return
        scale = 1.22 if getattr(event, "step", 0) > 0 else (1.0 / 1.22)
        xlim = tuple(map(float, ax.get_xlim()))
        ylim = tuple(map(float, ax.get_ylim()))
        new_xlim, new_ylim = xlim, ylim
        if not y_only:
            new_xlim = self._zoom_interval(xlim, float(event.xdata), scale, self._histogram_full_xlim)
        if not x_only:
            new_ylim = self._zoom_interval(ylim, float(event.ydata), scale, self._histogram_full_ylim)
        self._apply_histogram_limits(new_xlim, new_ylim)

    @staticmethod
    def _zoom_interval(limits, anchor: float, scale: float, bounds) -> tuple[float, float]:
        low, high = map(float, limits)
        span = (high - low) / float(scale)
        if not np.isfinite(span) or span <= 0.0:
            return low, high
        fraction = float(np.clip((anchor - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0))
        new_low = anchor - fraction * span
        new_high = new_low + span
        if bounds is not None:
            new_low, new_high = HklResultsPanel._clamp_interval(new_low, new_high, *bounds)
        return float(new_low), float(new_high)

    def _on_histogram_press(self, event) -> None:
        ax = self._histogram_axes
        if ax is None or event.inaxes is not ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x_only = self._event_modifier(event, "control") or self._event_modifier(event, "ctrl")
        y_only = self._event_modifier(event, "shift")
        xlim = tuple(map(float, ax.get_xlim()))
        ylim = tuple(map(float, ax.get_ylim()))
        bbox = ax.get_window_extent()
        self._histogram_drag_state = {
            "pixel_x": float(event.x), "pixel_y": float(event.y),
            "xlim": xlim, "ylim": ylim,
            "x_per_pixel": (xlim[1] - xlim[0]) / max(float(bbox.width), 1.0),
            "y_per_pixel": (ylim[1] - ylim[0]) / max(float(bbox.height), 1.0),
            "x_only": x_only, "y_only": y_only,
        }
        self.pair_canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_histogram_release(self, _event) -> None:
        if self._histogram_drag_state is not None:
            self._histogram_drag_state = None
            self.pair_canvas.unsetCursor()

    def _on_histogram_motion(self, event) -> None:
        ax = self._histogram_axes
        if ax is None:
            return
        state = self._histogram_drag_state
        if state is not None and event.inaxes is ax and event.x is not None and event.y is not None:
            dx = (float(event.x) - state["pixel_x"]) * state["x_per_pixel"]
            dy = (float(event.y) - state["pixel_y"]) * state["y_per_pixel"]
            xlim = state["xlim"]
            ylim = state["ylim"]
            new_xlim = xlim if state["y_only"] else (xlim[0] - dx, xlim[1] - dx)
            new_ylim = ylim if state["x_only"] else (ylim[0] - dy, ylim[1] - dy)
            if self._histogram_full_xlim is not None:
                new_xlim = self._clamp_interval(*new_xlim, *self._histogram_full_xlim)
            if self._histogram_full_ylim is not None:
                new_ylim = self._clamp_interval(*new_ylim, *self._histogram_full_ylim)
            self._apply_histogram_limits(new_xlim, new_ylim)
            return
        if event.inaxes is ax and event.x is not None and event.y is not None:
            self._update_reflection_hover(float(event.x))
        elif self.plot_hint_label is not None:
            self.plot_hint_label.setText("Колесо — масштаб · перетаскивание — сдвиг · Ctrl — θ · Shift — Y")

    def _apply_histogram_limits(self, xlim, ylim) -> None:
        ax = self._histogram_axes
        if ax is None:
            return
        xlim = self._clamp_interval(*xlim, *(self._histogram_full_xlim or xlim))
        ylim = self._clamp_interval(*ylim, *(self._histogram_full_ylim or ylim))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        self._histogram_view_limits = (*xlim, *ylim)
        self._refresh_rank_labels()
        self._draw_selected_reflection_overlay()
        self.pair_canvas.draw_idle()

    def _reset_histogram_view(self) -> None:
        if self._histogram_cache is None:
            return
        self._histogram_view_limits = None
        self._draw_histogram_view()

    def _on_reflection_row_changed(self, row: int, _column: int, _previous_row: int, _previous_column: int) -> None:
        self._selected_reflection_row = int(row) if row >= 0 else -1
        self._draw_selected_reflection_overlay()
        if self._histogram_axes is not None:
            self.pair_canvas.draw_idle()

    def _draw_selected_reflection_overlay(self) -> None:
        ax = self._histogram_axes
        if ax is None or self._histogram_cache is None:
            return
        for old in (
            getattr(self, "_selected_reflection_artist", None),
            getattr(self, "_selected_reflection_marker", None),
        ):
            if old is not None:
                try:
                    old.remove()
                except (ValueError, AttributeError):
                    pass
        self._selected_reflection_artist = None
        self._selected_reflection_marker = None
        row = self._selected_reflection_row
        ranking_x = np.asarray(self._histogram_cache.get("ranking_x", ()), dtype=np.float64)
        if row < 0 or row >= ranking_x.size or not np.isfinite(ranking_x[row]):
            return
        x_value = float(ranking_x[row])
        low, high = ax.get_xlim()
        if not low <= x_value <= high:
            return
        colors = theme_colors(self._theme)
        self._selected_reflection_artist = ax.axvline(
            x_value, color=colors["roi"], linewidth=2.0, linestyle=(0, (3, 1)),
            alpha=1.0, zorder=12,
        )
        marker = ax.plot(
            [x_value], [0.985], marker="v", markersize=4.2,
            markerfacecolor=colors["roi"], markeredgecolor=colors["canvas"],
            markeredgewidth=0.45, transform=ax.get_xaxis_transform(),
            clip_on=True, zorder=13,
        )
        self._selected_reflection_marker = marker[0]

    def _refresh_rank_labels(self) -> None:
        ax = self._histogram_axes
        cache = self._histogram_cache
        if ax is None or cache is None:
            return
        for artist in self._rank_label_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        colors = theme_colors(self._theme)
        low, high = map(float, ax.get_xlim())
        self._rank_label_artists = self._add_rank_labels(
            ax, cache["ranking_x"], cache["ranking_ranks"], low, high,
            label_color=colors["text"],
        )

    @staticmethod
    def _add_rank_labels(
        ax, ranking_x: np.ndarray, ranking_ranks: np.ndarray,
        view_low: float, view_high: float, *, label_color: str,
    ) -> None:
        """Place sparse rank tags vertically, anchored to their exact x-line.

        The blended data/axes transform keeps every tag attached to its
        reflection in θ while holding the text row just inside the top frame.
        Labels that would collide at the current zoom are omitted; hover still
        exposes the full reflection record.
        """
        if ranking_x.size == 0 or ranking_ranks.size == 0 or not view_high > view_low:
            return []
        artists = []
        axes_width_px = max(
            1.0, float(ax.get_position().width * ax.figure.get_figwidth() * ax.figure.dpi)
        )
        minimum_spacing_px = 12.5
        accepted_x_px: list[float] = []
        for x_value, rank_value in zip(ranking_x[:15], ranking_ranks[:15]):
            if not np.isfinite(x_value) or not np.isfinite(rank_value):
                continue
            x_value = float(x_value)
            if not view_low <= x_value <= view_high:
                continue
            x_px = (x_value - view_low) / (view_high - view_low) * axes_width_px
            if any(abs(x_px - previous) < minimum_spacing_px for previous in accepted_x_px):
                continue
            accepted_x_px.append(x_px)
            artists.append(ax.text(
                x_value, 0.985, f"#{int(rank_value)}",
                transform=ax.get_xaxis_transform(), rotation=90,
                rotation_mode="anchor", ha="center", va="top",
                fontsize=5.6, fontweight="medium", color=label_color,
                clip_on=True, zorder=8,
            ))
        return artists

    def _update_reflection_hover(self, mouse_x_px: float) -> None:
        ax = self._histogram_axes
        cache = self._histogram_cache
        if ax is None or cache is None or self._reflection_catalog.empty:
            return
        x_min, x_max = map(float, ax.get_xlim())
        bbox = ax.get_window_extent()
        if bbox.width <= 0:
            return
        x_value = x_min + (mouse_x_px - bbox.x0) / bbox.width * (x_max - x_min)
        values = self._reflection_x
        if values.size == 0:
            return
        index = int(np.searchsorted(values, x_value))
        candidates = [i for i in (index - 1, index) if 0 <= i < values.size]
        if not candidates:
            return
        nearest_index = min(candidates, key=lambda i: abs(float(values[i]) - x_value))
        nearest = float(values[nearest_index])
        distance_px = abs(nearest - x_value) / max(x_max - x_min, np.finfo(float).eps) * bbox.width
        if distance_px > 8.5:
            self.plot_hint_label.setText("Колесо — масштаб · перетаскивание — сдвиг · Ctrl — θ · Shift — Y")
            return
        catalog = self._reflection_catalog
        positions = pd.to_numeric(catalog["delta_theta_arcsec"], errors="coerce")
        matched = catalog.loc[np.isclose(positions.to_numpy(dtype=float), nearest, rtol=0.0, atol=max(abs(nearest), 1.0) * 1e-10)]
        hkl_names = [str(value) for value in matched.get("hkl", pd.Series(dtype=str)).tolist()]
        hkl_text = ", ".join(hkl_names[:3]) + (f" +{len(hkl_names) - 3}" if len(hkl_names) > 3 else "")
        rank_text = ""
        ranking_x = np.asarray(cache.get("ranking_x", ()), dtype=float)
        ranking_ranks = np.asarray(cache.get("ranking_ranks", ()), dtype=float)
        for rx, rr in zip(ranking_x, ranking_ranks):
            if np.isfinite(rx) and np.isclose(rx, nearest, rtol=0.0, atol=max(abs(nearest), 1.0) * 1e-10):
                rank_text = f"#{int(rr)} · "
                break
        self.plot_hint_label.setText(
            f"{rank_text}HKL {hkl_text or '—'} · Δθтеор={nearest:.4f}″"
        )

    def _render_all_pairs(self) -> None:
        """Render measured ranges and theoretical candidates with distinct marks.

        The Q10–Q90 interval and measured median share one y-level. The best
        theoretical HKL is deliberately offset vertically so it cannot be
        mistaken for an endpoint of the experimental interval.
        """
        if self._result is None:
            return
        stats = self._result.pair_statistics
        figure = self.all_chart_figure
        figure.clear()
        ax = figure.add_subplot(111)
        colors = theme_colors(self._theme)
        figure.patch.set_facecolor(colors["canvas"])
        ax.set_facecolor(colors["axes"])

        median_values = pd.to_numeric(
            stats["delta_theta_abs_median_arcsec"], errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        finite_stats = stats.loc[np.isfinite(median_values)].copy(deep=False)
        finite_stats = finite_stats.sort_values(
            "delta_theta_abs_median_arcsec", ascending=True, kind="mergesort"
        )
        rows: list[list[str]] = []
        axis_values: list[float] = []

        if finite_stats.empty:
            ax.text(
                0.5, 0.5, "Нет принятых соответствий с конечным Δθ",
                ha="center", va="center", fontsize=7.5, color=colors["muted"],
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            y_positions = np.arange(len(finite_stats), dtype=np.float64)
            interval_color = "#7896AE" if self._theme == "light" else "#88A8C0"
            median_color = "#3C927C" if self._theme == "light" else "#70B6A0"
            theory_color = "#C75E50" if self._theme == "light" else "#F08A78"

            for y, row in zip(y_positions, finite_stats.itertuples(index=False)):
                pair_id = str(row.pair_id)
                median = self._as_float(row.delta_theta_abs_median_arcsec)
                q10 = self._as_float(getattr(row, "delta_theta_abs_q10_arcsec", np.nan))
                q90 = self._as_float(getattr(row, "delta_theta_abs_q90_arcsec", np.nan))
                status_ready = str(row.identification_status) == "ready"
                best = self._as_float(
                    getattr(row, "best_theory_delta_theta_arcsec", np.nan)
                )

                for value in (median, q10, q90):
                    if np.isfinite(value):
                        axis_values.append(value)
                if status_ready and np.isfinite(best):
                    axis_values.append(best)

                if np.isfinite(q10) and np.isfinite(q90):
                    # Q10–Q90 is a measured interval, not an uncertainty on
                    # the theoretical candidate. Small caps make its ends clear.
                    ax.hlines(
                        y, q10, q90, color=interval_color, linewidth=2.4,
                        alpha=0.95, zorder=2,
                    )
                    if q90 > q10:
                        ax.vlines(
                            [q10, q90], y - 0.075, y + 0.075,
                            color=interval_color, linewidth=0.9, zorder=2,
                        )
                if np.isfinite(median):
                    ax.scatter(
                        [median], [y], s=22, marker="o",
                        facecolor=median_color, edgecolor=colors["text"],
                        linewidth=0.55, zorder=4,
                    )
                if status_ready and np.isfinite(best):
                    ax.scatter(
                        [best], [y + 0.17], s=27, marker="D",
                        facecolor=theory_color, edgecolor=colors["canvas"],
                        linewidth=0.65, zorder=5,
                    )

                best_hkl = getattr(row, "best_hkl", None)
                best_error = getattr(row, "best_abs_difference_arcsec", np.nan)
                assignment_count = self._count_int(row.matched_assignment_count)
                unique_partner_count = self._count_int(
                    getattr(row, "unique_partner_peak_count", assignment_count),
                    default=assignment_count,
                )
                rows.append([
                    pair_id,
                    f"{assignment_count:,} / {unique_partner_count:,}",
                    self._fmt(median, 2),
                    f"{self._fmt(q10, 1)}–{self._fmt(q90, 1)}",
                    str(best_hkl) if status_ready and pd.notna(best_hkl) else "—",
                    self._fmt(best_error, 2) if status_ready else "—",
                    "готово" if status_ready else "недостаточно данных",
                ])

            labels = [str(row.pair_id) for row in finite_stats.itertuples(index=False)]
            ax.set_yticks(y_positions, labels)
            x_max = max(axis_values, default=1.0)
            if not np.isfinite(x_max) or x_max <= 0.0:
                x_max = 1.0
            scale_mode, linthresh = comparison_axis_parameters(
                np.asarray(axis_values, dtype=np.float64)
            )
            if scale_mode == "symlog" and linthresh is not None:
                ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
                axis_caption = "Абсолютное разделение |Δθ|, ″ · symlog"
            else:
                axis_caption = "Абсолютное разделение |Δθ|, ″"
            ax.set_xlim(0.0, x_max * 1.08)
            ax.set_xlabel(axis_caption, fontsize=7.2, labelpad=2)
            if scale_mode == "linear":
                x_formatter = ScalarFormatter(useMathText=True)
                x_formatter.set_scientific(True)
                x_formatter.set_powerlimits((0, 0))
                x_formatter.set_useOffset(False)
                ax.xaxis.set_major_formatter(x_formatter)
                ax.xaxis.get_offset_text().set_color(colors["text"])
            ax.set_ylabel("Пара ветвей", fontsize=7.2, labelpad=2)
            ax.set_title("Измеренный интервал и теоретический кандидат", fontsize=7.8, pad=3)
            ax.grid(axis="x", color=colors["grid"], linewidth=0.5, alpha=0.32)
            ax.set_axisbelow(True)
            ax.set_ylim(-0.42, len(finite_stats) - 0.42 + 0.17)
            ax.tick_params(axis="both", labelsize=6.8, colors=colors["text"], length=2.0, pad=1.8)
            for spine in ax.spines.values():
                spine.set_color(colors["border"])
                spine.set_linewidth(0.6)
            ax.xaxis.label.set_color(colors["text"])
            ax.yaxis.label.set_color(colors["text"])
            ax.title.set_color(colors["text"])
            handles = [
                Line2D([0], [0], color=interval_color, linewidth=2.4, label="Q10–Q90 измерений"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=median_color,
                       markeredgecolor=colors["text"], markersize=4.2, label="Медиана измерений"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor=theory_color,
                       markeredgecolor=colors["canvas"], markersize=4.2, label="Лучший теоретический HKL"),
            ]
            legend = ax.legend(handles=handles, frameon=False, fontsize=5.9,
                               loc="upper center", bbox_to_anchor=(0.5, -0.24),
                               ncol=3, handlelength=1.45,
                               borderaxespad=0.0, columnspacing=0.9, labelspacing=0.35)
            if legend is not None:
                for label in legend.get_texts():
                    label.set_color(colors["text"])

        # Keep pairs with zero matches in the audit table too.
        finite_ids = set(finite_stats["pair_id"].astype(str).tolist())
        for row in stats.loc[~stats["pair_id"].astype(str).isin(finite_ids)].itertuples(index=False):
            assignment_count = self._count_int(row.matched_assignment_count)
            unique_partner_count = self._count_int(
                getattr(row, "unique_partner_peak_count", assignment_count),
                default=assignment_count,
            )
            rows.append([
                str(row.pair_id),
                f"{assignment_count:,} / {unique_partner_count:,}",
                "—", "—", "—", "—", "нет измерений",
            ])
        self._set_table(self.pair_table, rows)
        figure.subplots_adjust(left=0.13, right=0.992, bottom=0.25, top=0.84)
        self.all_canvas.draw_idle()

