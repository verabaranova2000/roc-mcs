"""Compact data-driven 3D illustration of spatial-angular branch linking."""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QSize, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPalette, QPolygonF
from PyQt6.QtWidgets import QWidget, QSizePolicy

from .branch_preview_data import build_branch_preview_graph


class BranchDetectionPreview(QWidget):
    """Draw a scan-derived spatial-angular peak graph as small floating sheets.

    Every dot is an unthinned real peak event from a small contiguous patch of
    the selected parent catalogue.  The detector's X/Y lattice is the floor;
    height is theta.  Accepted links form colored, translucent ribbons and,
    where four real corner events and all four accepted perimeter links exist,
    small translucent sheet cells. Dashed links are exact one-to-one candidate
    matches rejected by the current θ-gate. No points or links are invented.
    Connected components are local to the displayed patch, not global scan IDs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theta_gate = 60.0
        self._catalog = None
        self._patch_info: dict[str, Any] = {"available": False}
        self._graph: dict[str, Any] = {
            "events": {}, "candidate_edges": [], "accepted_edges": [],
            "n_branches": 0, "branch_by_peak_id": {},
        }
        self._message = "Выберите родительский результат с каталогом пиков."
        self.setObjectName("branchDetectionPreview")
        self.setMinimumSize(380, 154)
        self.setMaximumHeight(198)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(
            "Трёхмерная иллюстрация сопоставления пиков в пространственно-угловых ветвях"
        )
        self.setToolTip(
            "Данные не прореживаются: каждая точка — реальный peak-event выбранного "
            "фрагмента. Основание — соседство пикселей X/Y, высота — θ. Цвет точки/ленты "
            "задаёт локальную компоненту связности. Сплошная лента — принятое ребро, "
            "оранжевый пунктир — оптимальное one-to-one назначение, отвергнутое θ-gate. "
            "Полупрозрачная площадка появляется только при наличии четырёх реальных "
            "угловых пиков и всех четырёх принятых рёбер. Пересечение линий в проекции "
            "не означает общий узел: его определяет только одна точка/peak_id."
        )

    def sizeHint(self) -> QSize:
        return QSize(560, 182)

    def set_catalog(self, catalog: Any | None, info: dict[str, Any] | None = None) -> None:
        """Replace the compact real-data patch selected by the stats worker."""
        self._catalog = catalog.copy() if catalog is not None else None
        self._patch_info = dict(info or {})
        if not self._patch_info.get("available", self._catalog is not None):
            self._catalog = None
            self._message = str(
                self._patch_info.get("message")
                or "В выбранном результате нет пригодного фрагмента каталога пиков."
            )
        elif self._catalog is None or getattr(self._catalog, "empty", True):
            self._catalog = None
            self._message = "В локальном фрагменте нет событий пиков."
        else:
            self._message = ""
        self._recalculate()
        self.update()

    def clear_catalog(self, message: str = "Загрузка фрагмента каталога пиков…") -> None:
        self._catalog = None
        self._patch_info = {"available": False}
        self._graph = {
            "events": {}, "candidate_edges": [], "accepted_edges": [],
            "n_branches": 0, "branch_by_peak_id": {},
        }
        self._message = str(message)
        self.update()

    @property
    def theta_gate(self) -> float:
        return float(self._theta_gate)

    def set_theta_gate(self, value: float) -> None:
        gate = float(value)
        if not np.isfinite(gate) or gate <= 0.0 or gate == self._theta_gate:
            return
        self._theta_gate = gate
        self._recalculate()
        self.update()

    def _recalculate(self) -> None:
        if self._catalog is None or getattr(self._catalog, "empty", True):
            self._graph = {
                "events": {}, "candidate_edges": [], "accepted_edges": [],
                "n_branches": 0, "branch_by_peak_id": {},
            }
            return
        try:
            self._graph = build_branch_preview_graph(
                self._catalog, theta_gate=self._theta_gate
            )
            self._message = ""
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._graph = {
                "events": {}, "candidate_edges": [], "accepted_edges": [],
                "n_branches": 0, "branch_by_peak_id": {},
            }
            self._message = f"Не удалось построить локальный граф: {exc}"

    @staticmethod
    def _component_color(component_id: int, *, alpha: int = 255) -> QColor:
        # Golden-angle hue spacing is deterministic and remains legible when
        # the gate splits or merges connected components.
        hue = int((int(component_id) * 137.508 + 24.0) % 360)
        color = QColor.fromHsv(hue, 188, 242)
        color.setAlpha(int(alpha))
        return color

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, end: QPointF, color: QColor) -> None:
        painter.setPen(QPen(color, 1.25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)
        direction_x = end.x() - start.x()
        direction_y = end.y() - start.y()
        length = max(float(np.hypot(direction_x, direction_y)), 1e-9)
        ux, uy = direction_x / length, direction_y / length
        size = 4.0
        for sign in (-1.0, 1.0):
            wing = QPointF(
                end.x() - size * ux + sign * size * 0.55 * uy,
                end.y() - size * uy - sign * size * 0.55 * ux,
            )
            painter.drawLine(end, wing)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        text = palette.color(QPalette.ColorRole.Text)
        muted = palette.color(QPalette.ColorRole.Mid)
        surface = palette.color(QPalette.ColorRole.AlternateBase)

        card = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(muted, 0.8))
        painter.setBrush(QBrush(surface))
        painter.drawRoundedRect(card, 7.0, 7.0)

        if self._catalog is None or not self._graph.get("events"):
            painter.setPen(QPen(text, 1.0))
            painter.drawText(
                card.adjusted(14, 10, -14, -10),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._message or "Нет данных для иллюстрации.",
            )
            painter.end()
            return

        events = self._graph["events"]
        branch_by_id = self._graph["branch_by_peak_id"]
        accepted_edges = self._graph["accepted_edges"]
        candidate_edges = self._graph["candidate_edges"]
        info = self._patch_info
        patch_width = int(info.get("width", 4))
        patch_height = int(info.get("height", 3))
        source_label = str(info.get("position_label", "пики"))
        accepted_count = len(accepted_edges)
        component_count = int(self._graph.get("n_branches", 0))

        # One compact, non-overlapping header: the right-hand gate value gets
        # a reserved area; the descriptive part is safely elided if needed.
        heading_font = QFont(self.font())
        heading_font.setPointSizeF(max(7.5, self.font().pointSizeF() - 0.35))
        heading_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(heading_font)
        painter.setPen(QPen(text, 1.0))
        gate_text = f"θ-gate {self._theta_gate:.4g}″"
        gate_width = painter.fontMetrics().horizontalAdvance(gate_text)
        heading = (
            f"{source_label}  ·  {patch_width}×{patch_height} px  ·  "
            f"{len(events)} пиков  ·  {accepted_count}/{len(candidate_edges)} связей  ·  "
            f"{component_count} компонент"
        )
        heading_rect = card.adjusted(11, 5, -(gate_width + 23), -card.height() + 23)
        heading = painter.fontMetrics().elidedText(
            heading, Qt.TextElideMode.ElideRight, max(20, heading_rect.width())
        )
        painter.drawText(
            heading_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            heading,
        )
        painter.setPen(QPen(muted, 1.0))
        painter.drawText(
            card.adjusted(0, 5, -11, -card.height() + 23),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            gate_text,
        )

        plot = card.adjusted(12, 27, -12, -25)
        x_values = [int(event["x"]) for event in events.values()]
        y_values = [int(event["y"]) for event in events.values()]
        theta_values = [float(event["theta"]) for event in events.values()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        theta_min, theta_max = min(theta_values), max(theta_values)
        x_span = max(1, x_max - x_min)
        y_span = max(1, y_max - y_min)
        theta_span = max(theta_max - theta_min, 1e-12)

        # Isometric projection: X/Y form a detector-pixel floor grid; θ is the
        # vertical coordinate.  The same fixed projection is retained while
        # the user changes θ-gate, so only the real graph topology changes.
        available_width = max(180.0, float(plot.width()) - 36.0)
        step_x = min(33.0, available_width / (x_span + 0.58 * y_span + 0.6))
        step_y = step_x * 0.58
        projected_width = x_span * step_x + y_span * step_y
        origin_x = float(plot.left()) + (float(plot.width()) - projected_width) / 2.0
        baseline_y = float(plot.bottom()) - 12.0
        depth_x = step_x * 0.18
        depth_y = step_y * 0.78
        height_theta = min(70.0, max(34.0, float(plot.height()) * 0.72))

        def project(event: dict[str, Any]) -> QPointF:
            xr = float(event["x"] - x_min)
            yr = float(event["y"] - y_min)
            theta_norm = (float(event["theta"]) - theta_min) / theta_span
            px = origin_x + xr * step_x + yr * step_y
            py = baseline_y - xr * depth_x + yr * depth_y - theta_norm * height_theta
            return QPointF(px, py)

        projected = {int(peak_id): project(event) for peak_id, event in events.items()}
        muted_grid = QColor(muted)
        muted_grid.setAlpha(92)

        # Draw a small, explicit X/Y detector lattice at θ-min.  It anchors the
        # abstract projection to detector neighbours rather than a free cloud.
        for x in range(x_min, x_max + 1):
            start_point = project({"x": x, "y": y_min, "theta": theta_min})
            end_point = project({"x": x, "y": y_max, "theta": theta_min})
            painter.setPen(QPen(muted_grid, 0.75, Qt.PenStyle.DotLine))
            painter.drawLine(start_point, end_point)
        for y in range(y_min, y_max + 1):
            start_point = project({"x": x_min, "y": y, "theta": theta_min})
            end_point = project({"x": x_max, "y": y, "theta": theta_min})
            painter.setPen(QPen(muted_grid, 0.75, Qt.PenStyle.DotLine))
            painter.drawLine(start_point, end_point)

        # Draw compact axis arrows at the front-left corner.
        axis_origin = project({"x": x_min, "y": y_min, "theta": theta_min})
        axis_x = project({"x": x_max, "y": y_min, "theta": theta_min})
        axis_y = project({"x": x_min, "y": y_max, "theta": theta_min})
        axis_theta = project({"x": x_min, "y": y_min, "theta": theta_max})
        axis_color = QColor(muted)
        axis_color.setAlpha(190)
        self._draw_arrow(painter, axis_origin, axis_x, axis_color)
        self._draw_arrow(painter, axis_origin, axis_y, axis_color)
        self._draw_arrow(painter, axis_origin, axis_theta, axis_color)
        painter.setPen(QPen(text, 1.0))
        painter.drawText(QPointF(axis_x.x() + 3, axis_x.y() + 1), "X")
        painter.drawText(QPointF(axis_y.x() + 3, axis_y.y() + 1), "Y")
        painter.drawText(QPointF(axis_theta.x() + 3, axis_theta.y() - 2), "θ, ″")

        accepted_keys = {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b, _delta in accepted_edges
        }
        components = sorted({int(value) for value in branch_by_id.values()})

        # A tinted sheet cell is drawn only if the four actual corner events
        # belong to one component and all four perimeter links are accepted.
        # Multiple events from the same component at a single pixel are kept
        # separate; ambiguous cells are not fabricated.
        pixel_nodes: dict[tuple[int, int, int], list[int]] = {}
        for peak_id, event in events.items():
            component = int(branch_by_id.get(int(peak_id), -1))
            key = (int(event["x"]), int(event["y"]), component)
            pixel_nodes.setdefault(key, []).append(int(peak_id))

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for component in components:
                    corners = [
                        pixel_nodes.get((x, y, component), []),
                        pixel_nodes.get((x + 1, y, component), []),
                        pixel_nodes.get((x + 1, y + 1, component), []),
                        pixel_nodes.get((x, y + 1, component), []),
                    ]
                    if any(len(values) != 1 for values in corners):
                        continue
                    p00, p10, p11, p01 = (values[0] for values in corners)
                    perimeter = {
                        (min(p00, p10), max(p00, p10)),
                        (min(p10, p11), max(p10, p11)),
                        (min(p11, p01), max(p11, p01)),
                        (min(p01, p00), max(p01, p00)),
                    }
                    if not perimeter.issubset(accepted_keys):
                        continue
                    polygon = QPolygonF([
                        projected[p00], projected[p10],
                        projected[p11], projected[p01],
                    ])
                    fill = self._component_color(component, alpha=48)
                    edge = self._component_color(component, alpha=165)
                    painter.setPen(QPen(edge, 1.0))
                    painter.setBrush(QBrush(fill))
                    painter.drawPolygon(polygon)

        # Every accepted edge gets a translucent wide under-stroke (a small
        # ribbon) and a narrow centreline.  The broad colour patches evoke a
        # spatial-angular sheet while preserving the exact graph topology.
        for peak_a, peak_b, _delta in accepted_edges:
            component = int(branch_by_id.get(int(peak_a), 0))
            start_point = projected[int(peak_a)]
            end_point = projected[int(peak_b)]
            ribbon = self._component_color(component, alpha=58)
            painter.setPen(QPen(ribbon, 7.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start_point, end_point)
            core = self._component_color(component, alpha=238)
            painter.setPen(QPen(core, 1.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start_point, end_point)

        # Dashed candidates are the real one-to-one minimum-|Δθ| assignments
        # that the current gate rejected; they remain visible to explain why a
        # small gate can split a spatial-angular structure.
        for peak_a, peak_b, delta, accepted in candidate_edges:
            if accepted:
                continue
            start_point = projected[int(peak_a)]
            end_point = projected[int(peak_b)]
            rejected = QColor("#D18B55")
            rejected.setAlpha(225)
            painter.setPen(QPen(
                rejected, 1.55, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap
            ))
            painter.drawLine(start_point, end_point)

        # Peak events are explicit graph vertices. The dark outline survives
        # the colored ribbons, and a small light highlight makes each dot read
        # as a point in 3D rather than a flat node label.
        for peak_id, point in projected.items():
            component = int(branch_by_id.get(int(peak_id), 0))
            color = self._component_color(component)
            painter.setPen(QPen(QColor(17, 22, 30, 235), 1.2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(point, 4.8, 4.8)
            highlight = QColor(255, 255, 255, 180)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(highlight))
            painter.drawEllipse(QPointF(point.x() - 1.2, point.y() - 1.3), 1.1, 1.1)

        # Small, consistent legend: node, accepted ribbon, rejected assignment.
        legend_y = float(card.bottom()) - 8.0
        x_cursor = float(card.left()) + 12.0
        painter.setPen(QPen(QColor(17, 22, 30, 230), 1.0))
        painter.setBrush(QBrush(self._component_color(0)))
        painter.drawEllipse(QPointF(x_cursor + 3.0, legend_y - 2.5), 3.2, 3.2)
        painter.setPen(QPen(muted, 1.0))
        painter.drawText(QPointF(x_cursor + 11.0, legend_y + 1.0), "пик")
        x_cursor += 42.0
        ribbon_color = QColor(63, 155, 193, 185)
        painter.setPen(QPen(ribbon_color, 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(x_cursor, legend_y - 2.5), QPointF(x_cursor + 15.0, legend_y - 2.5))
        painter.setPen(QPen(muted, 1.0))
        painter.drawText(QPointF(x_cursor + 20.0, legend_y + 1.0), "связь принята")
        x_cursor += 102.0
        rejected = QColor("#D18B55")
        painter.setPen(QPen(rejected, 1.6, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(x_cursor, legend_y - 2.5), QPointF(x_cursor + 15.0, legend_y - 2.5))
        painter.setPen(QPen(muted, 1.0))
        painter.drawText(QPointF(x_cursor + 20.0, legend_y + 1.0), "назначение отклонено")
        painter.end()
