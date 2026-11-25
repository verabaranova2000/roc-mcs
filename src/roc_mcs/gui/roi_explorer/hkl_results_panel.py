"""Compact scientific results panel for Kα doublet / HKL identification."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QLabel, QSizePolicy,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from roc_mcs.gui.roi_explorer.theme import theme_colors


class HklResultsPanel(QWidget):
    """Renders saved HKL results without altering the scientific artifact."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hklResultsPanel")
        self._result = None
        self._theme = "light"

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(5)

        self.summary_label = QLabel("Нет результата теоретической идентификации.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_label.setObjectName("hklSummary")
        root.addWidget(self.summary_label)

        self.pair_combo = QComboBox(self)
        self.pair_combo.setObjectName("hklPairSelector")
        self.pair_combo.setToolTip("Выберите пару ветвей для просмотра экспериментального распределения и теоретических рефлексов.")
        self.pair_combo.currentIndexChanged.connect(self._render_selected_pair)
        root.addWidget(self.pair_combo)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("hklResultTabs")
        root.addWidget(self.tabs)

        self.pair_tab = QWidget(self.tabs)
        pair_layout = QVBoxLayout(self.pair_tab)
        pair_layout.setContentsMargins(2, 4, 2, 2)
        pair_layout.setSpacing(4)
        self.pair_chart_figure = Figure(figsize=(4.0, 2.0), dpi=100, tight_layout=True)
        self.pair_canvas = FigureCanvasQTAgg(self.pair_chart_figure)
        self.pair_canvas.setMinimumHeight(165)
        self.pair_canvas.setMaximumHeight(220)
        self.pair_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pair_layout.addWidget(self.pair_canvas)
        self.pair_stats_label = QLabel("", self.pair_tab)
        self.pair_stats_label.setWordWrap(True)
        pair_layout.addWidget(self.pair_stats_label)
        self.reflection_table = self._make_table(
            ["#", "HKL", "d, Å", "θ₁, °", "Δθтеор, ″", "|Δ−med|, ″", "|F_H(Kα₁)|²", "Darwin σ/π, ″"],
            max_height=190,
        )
        pair_layout.addWidget(self.reflection_table)
        self.tabs.addTab(self.pair_tab, "Пара / рефлексы")

        self.all_pairs_tab = QWidget(self.tabs)
        all_layout = QVBoxLayout(self.all_pairs_tab)
        all_layout.setContentsMargins(2, 4, 2, 2)
        all_layout.setSpacing(4)
        self.all_chart_figure = Figure(figsize=(4.0, 2.1), dpi=100, tight_layout=True)
        self.all_canvas = FigureCanvasQTAgg(self.all_chart_figure)
        self.all_canvas.setMinimumHeight(175)
        self.all_canvas.setMaximumHeight(235)
        self.all_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        all_layout.addWidget(self.all_canvas)
        self.pair_table = self._make_table(
            ["Пара", "N", "Медиана, ″", "Q10–Q90, ″", "HKL", "Ошибка, ″", "Статус"],
            max_height=190,
        )
        all_layout.addWidget(self.pair_table)
        self.tabs.addTab(self.all_pairs_tab, "Сравнение пар")

        self.method_note = QLabel(
            "HKL ранжированы по геометрической близости расчётного Δθ к медиане измерений. "
            "Это не апостериорные вероятности и не оценка интенсивности рефлексов.", self
        )
        self.method_note.setWordWrap(True)
        self.method_note.setObjectName("hklMethodNote")
        root.addWidget(self.method_note)
        self.apply_theme(self._theme)
        self.set_result(None)

    @staticmethod
    def _make_table(headers: list[str], *, max_height: int) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(21)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(110)
        table.setMaximumHeight(max_height)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        return table

    @staticmethod
    def _fmt(value: Any, digits: int = 2) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return "—"
        return f"{number:.{digits}f}" if np.isfinite(number) else "—"

    @staticmethod
    def _set_table(table: QTableWidget, rows: list[list[str]]) -> None:
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(rows))
            for row_index, values in enumerate(rows):
                for column_index, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if column_index == 0:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(row_index, column_index, item)
            table.resizeColumnsToContents()
        finally:
            table.setUpdatesEnabled(True)

    def apply_theme(self, theme: str | None = None) -> None:
        self._theme = str(theme or self._theme).lower()
        colors = theme_colors(self._theme)
        self.setStyleSheet(
            f"QLabel#hklMethodNote {{ color: {colors['muted']}; font-size: 9px; }}"
            f"QLabel#hklSummary {{ color: {colors['text']}; font-weight: 600; }}"
        )
        if self._result is None:
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Выберите результат HKL")
            self._draw_empty(self.all_chart_figure, self.all_canvas, "Нет данных для сравнения пар")
        else:
            # A theme change must restyle the existing result, not erase it.
            self._render_all_pairs()
            if self.pair_combo.currentIndex() >= 0:
                self._render_selected_pair(self.pair_combo.currentIndex())
            else:
                self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет измеренных пар")

    def set_result(self, result: Any | None) -> None:
        self._result = result
        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        if result is None:
            self.summary_label.setText("Нет результата теоретической идентификации.")
            self.pair_stats_label.setText("")
            self._set_table(self.reflection_table, [])
            self._set_table(self.pair_table, [])
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет результата HKL")
            self._draw_empty(self.all_chart_figure, self.all_canvas, "Нет результата HKL")
            self.pair_combo.setEnabled(False)
            self.pair_combo.blockSignals(False)
            return

        stats = result.pair_statistics
        for row in stats.itertuples(index=False):
            label = f"{row.pair_id} · N={int(row.matched_assignment_count)}"
            self.pair_combo.addItem(label, str(row.pair_id))
        self.pair_combo.setEnabled(self.pair_combo.count() > 0)
        metadata = result.metadata
        cell_description = metadata.get("unit_cell_description")
        if not cell_description:
            cell_description = f"a={self._fmt(metadata.get('lattice_a_A'), 5)} Å"
        self.summary_label.setText(
            f"{metadata.get('crystal_name', 'Кристалл')} · {cell_description} · "
            f"Mo Kα₁={self._fmt(metadata.get('lambda_ka1_A'), 7)} Å; "
            f"Kα₂={self._fmt(metadata.get('lambda_ka2_A'), 7)} Å\n"
            f"Рефлексов: {len(result.theoretical_reflections)} · отброшено по Брэггу: {len(result.rejected_reflections)} · "
            f"измерений: {len(result.measurements)} · пар: {len(stats)}"
        )
        self._render_all_pairs()
        self.pair_combo.blockSignals(False)
        if self.pair_combo.count():
            self._render_selected_pair(self.pair_combo.currentIndex())
        else:
            self.pair_stats_label.setText("Нет пар с принятыми соответствиями.")
            self._set_table(self.reflection_table, [])
            self._draw_empty(self.pair_chart_figure, self.pair_canvas, "Нет измеренных пар")

    def _draw_empty(self, figure: Figure, canvas: FigureCanvasQTAgg, message: str) -> None:
        figure.clear()
        ax = figure.add_subplot(111)
        colors = theme_colors(self._theme)
        figure.patch.set_facecolor(colors["canvas"])
        ax.set_facecolor(colors["axes"])
        ax.text(0.5, 0.5, message, ha="center", va="center", color=colors["muted"], transform=ax.transAxes)
        ax.set_axis_off()
        canvas.draw_idle()

    def _render_selected_pair(self, _index: int = -1) -> None:
        if self._result is None or self.pair_combo.currentIndex() < 0:
            return
        pair_id = str(self.pair_combo.currentData())
        stats_rows = self._result.pair_statistics
        selected = stats_rows.loc[stats_rows["pair_id"].astype(str) == pair_id]
        if selected.empty:
            return
        stats = selected.iloc[0]
        measurements = self._result.measurements
        measurements = measurements.loc[measurements["pair_id"].astype(str) == pair_id]
        ranking = self._result.pair_ranking
        ranking = ranking.loc[ranking["pair_id"].astype(str) == pair_id].sort_values("rank")
        self._render_pair_histogram(pair_id, stats, measurements, self._result.theoretical_reflections, ranking)
        if str(stats["identification_status"]) == "ready" and not ranking.empty:
            best = ranking.iloc[0]
            self.pair_stats_label.setText(
                f"Принято соответствий: {int(stats['matched_assignment_count']):,}; "
                f"медиана |Δθ|={self._fmt(stats['delta_theta_abs_median_arcsec'], 2)}″; "
                f"Q10–Q90={self._fmt(stats['delta_theta_abs_q10_arcsec'], 2)}–"
                f"{self._fmt(stats['delta_theta_abs_q90_arcsec'], 2)}″; "
                f"MAD={self._fmt(stats['delta_theta_abs_mad_arcsec'], 2)}″. "
                f"Ближайший геометрический рефлекс: {best['hkl']}, "
                f"Δθтеор={self._fmt(best['delta_theta_arcsec'], 2)}″, "
                f"|ошибка|={self._fmt(best['difference_from_exp_median_arcsec'], 2)}″."
            )
        else:
            self.pair_stats_label.setText(
                f"Для пары {pair_id} найдено {int(stats['matched_assignment_count'])} соответствий; "
                "недостаточно измерений для устойчивого ранжирования (нужно не менее двух)."
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

    def _render_pair_histogram(
        self, pair_id: str, stats: pd.Series, measurements: pd.DataFrame,
        reflections: pd.DataFrame, ranking: pd.DataFrame,
    ) -> None:
        figure = self.pair_chart_figure
        figure.clear()
        ax = figure.add_subplot(111)
        values = pd.to_numeric(measurements["delta_theta_abs_arcsec"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            self._draw_empty(figure, self.pair_canvas, "Нет конечных измеренных Δθ")
            return
        width = float(self._result.metadata.get("histogram_bin_width_arcsec", 7.0))
        low = float(np.floor(np.min(values) / width) * width)
        high = float(np.ceil(np.max(values) / width) * width)
        if high <= low:
            high = low + width
        bins = np.arange(low, high + width * 1.001, width)
        if bins.size < 2:
            bins = np.array([low, low + width])
        ax.hist(values, bins=bins, color="#7298B8", edgecolor="#233746", linewidth=0.55, alpha=0.72, zorder=2)
        median = float(stats["delta_theta_abs_median_arcsec"])
        mean = float(stats["delta_theta_abs_mean_arcsec"])
        ax.axvline(median, color="#C66B24", linestyle="--", linewidth=1.8, label=f"Медиана {median:.1f}″", zorder=5)
        ax.axvline(mean, color="#3C8A68", linestyle="-.", linewidth=1.5, label=f"Среднее {mean:.1f}″", zorder=5)
        span = max(high - low, width)
        view_low, view_high = low - span * 0.08, high + span * 0.08
        visible = reflections.loc[
            (reflections["delta_theta_arcsec"] >= view_low)
            & (reflections["delta_theta_arcsec"] <= view_high)
        ].copy()
        best_hkl = str(ranking.iloc[0]["hkl"]) if not ranking.empty else None
        # Draw every in-range theoretical line; label the nearest few only.
        for row in visible.itertuples(index=False):
            is_best = str(row.hkl) == best_hkl
            ax.axvline(
                float(row.delta_theta_arcsec),
                color="#A33D3D" if is_best else "#777777",
                linestyle="-" if is_best else ":",
                linewidth=2.0 if is_best else 0.75,
                alpha=0.95 if is_best else 0.55,
                zorder=4 if is_best else 1,
            )
        label_hkls = set(ranking.head(5)["hkl"].astype(str).tolist())
        y_top = ax.get_ylim()[1]
        for index, row in enumerate(visible.itertuples(index=False)):
            if str(row.hkl) not in label_hkls:
                continue
            ax.text(
                float(row.delta_theta_arcsec), y_top * (0.96 if index % 2 == 0 else 0.78),
                str(row.hkl), rotation=90, ha="right", va="top",
                fontsize=7, color="#8E2929" if str(row.hkl) == best_hkl else "#555555",
            )
        ax.set_xlim(view_low, view_high)
        ax.set_xlabel("|Δθ|, угловые секунды")
        ax.set_ylabel("Число соответствий")
        ax.set_title(f"Эксперимент и теоретические Kα₁/Kα₂-разделения · {pair_id}", fontsize=9)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        self._style_axes(ax)
        figure.tight_layout(pad=1.1)
        self.pair_canvas.draw_idle()

    def _render_all_pairs(self) -> None:
        """Render a compact KDE-ridge forest and an auditable per-pair table."""
        result = self._result
        if result is None:
            return
        stats = result.pair_statistics.copy(deep=True)
        measurements = result.measurements
        figure = self.all_chart_figure
        figure.clear()
        ax = figure.add_subplot(111)
        rows: list[list[str]] = []

        finite_stats = stats.loc[
            np.isfinite(pd.to_numeric(stats["delta_theta_abs_median_arcsec"], errors="coerce"))
        ].sort_values("delta_theta_abs_median_arcsec", ascending=True, kind="mergesort")
        if finite_stats.empty:
            ax.text(0.5, 0.5, "Нет принятых соответствий с конечным Δθ", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            from scipy.stats import gaussian_kde

            pair_colors = ["#547C9F", "#4C927A", "#9875A5", "#C18A49", "#4C9BA5", "#B46A72"]
            y_positions = np.arange(len(finite_stats))
            for y, row in zip(y_positions, finite_stats.itertuples(index=False)):
                pair_id = str(row.pair_id)
                pair_measurements = measurements.loc[
                    measurements["pair_id"].astype(str) == pair_id, "delta_theta_abs_arcsec"
                ]
                values = pd.to_numeric(pair_measurements, errors="coerce").to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                color = pair_colors[y % len(pair_colors)]
                median = float(row.delta_theta_abs_median_arcsec)
                q10 = float(row.delta_theta_abs_q10_arcsec)
                q90 = float(row.delta_theta_abs_q90_arcsec)
                base = float(y)

                # A per-pair KDE ridge mirrors the notebook forest plot. A KDE
                # is meaningful only when the sample has at least two distinct
                # values; degenerate samples remain visible as a point/interval.
                if values.size >= 2 and np.unique(values).size >= 2:
                    span = max(float(np.ptp(values)), 1.0)
                    x_grid = np.linspace(float(values.min()) - 0.12 * span,
                                         float(values.max()) + 0.12 * span, 180)
                    try:
                        density = gaussian_kde(values, bw_method="scott")(x_grid)
                        peak = float(np.max(density))
                        if np.isfinite(peak) and peak > 0:
                            ridge = density / peak * 0.58
                            ax.fill_between(x_grid, base, base + ridge, color=color, alpha=0.30, zorder=2)
                            ax.plot(x_grid, base + ridge, color=color, linewidth=1.1, zorder=3)
                    except (ValueError, np.linalg.LinAlgError):
                        # Singular covariance is not silently regularized with
                        # an arbitrary bandwidth; the robust interval stays.
                        pass

                ax.hlines(base, q10, q90, color=color, linewidth=3.0,
                          alpha=0.95, zorder=4, label="Q10–Q90" if y == 0 else None)
                ax.plot(median, base, "|", color="#222222", markersize=12,
                        markeredgewidth=1.8, zorder=5,
                        label="Медиана" if y == 0 else None)

                status_ready = str(row.identification_status) == "ready"
                best = pd.to_numeric(pd.Series([getattr(row, "best_theory_delta_theta_arcsec", np.nan)]), errors="coerce").iloc[0]
                if status_ready and np.isfinite(best):
                    ax.plot(best, base + 0.02, "D", color="#A33D3D", markersize=4.2,
                            zorder=6, label="Лучший теоретический HKL" if y == 0 else None)
                best_hkl = getattr(row, "best_hkl", None)
                best_error = getattr(row, "best_abs_difference_arcsec", np.nan)
                rows.append([
                    pair_id, f"{int(row.matched_assignment_count):,}", self._fmt(median, 2),
                    f"{self._fmt(q10, 1)}–{self._fmt(q90, 1)}",
                    str(best_hkl) if status_ready and pd.notna(best_hkl) else "—",
                    self._fmt(best_error, 2) if status_ready else "—",
                    "готово" if status_ready else "недостаточно данных",
                ])

            labels = [
                f"{row.pair_id} · N={int(row.matched_assignment_count)}"
                for row in finite_stats.itertuples(index=False)
            ]
            ax.set_yticks(y_positions, labels)
            ax.set_xlabel("|Δθ|, угловые секунды")
            ax.set_ylabel("Пара ветвей")
            ax.set_title("Распределения по парам: KDE и робастный интервал", fontsize=9)
            ax.grid(axis="x", alpha=0.22)
            ax.legend(frameon=False, fontsize=7, loc="best")
            ax.set_ylim(-0.35, len(finite_stats) - 0.05 + 0.72)

        # Keep pairs with zero matches in the table too: absence is informative.
        for row in stats.loc[~stats["pair_id"].astype(str).isin(set(finite_stats["pair_id"].astype(str)))].itertuples(index=False):
            rows.append([
                str(row.pair_id), f"{int(row.matched_assignment_count):,}", "—", "—", "—", "—",
                "нет измерений",
            ])
        self._set_table(self.pair_table, rows)
        self._style_axes(ax)
        figure.tight_layout(pad=1.1)
        self.all_canvas.draw_idle()

    def _style_axes(self, ax) -> None:
        colors = theme_colors(self._theme)
        ax.figure.patch.set_facecolor(colors["canvas"])
        ax.set_facecolor(colors["axes"])
        ax.tick_params(colors=colors["text"], labelsize=7)
        ax.xaxis.label.set_color(colors["text"])
        ax.yaxis.label.set_color(colors["text"])
        ax.title.set_color(colors["text"])
        for spine in ax.spines.values():
            spine.set_color(colors["border"])
        ax.grid(color=colors["grid"], alpha=0.2)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(colors["axes"])
            legend.get_frame().set_edgecolor(colors["border"])
            for text in legend.get_texts():
                text.set_color(colors["text"])
