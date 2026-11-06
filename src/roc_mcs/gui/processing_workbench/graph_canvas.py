"""Compact interactive DAG canvas for Processing Workbench.

The persistent model remains WorkflowHistory/DAG. This module only renders it and
forwards select / add / preview gestures to the owning QWidget.
"""
from __future__ import annotations

from collections import defaultdict
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPalette
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGraphicsItem, QGraphicsObject, QGraphicsPathItem,
    QGraphicsScene, QGraphicsView, QWidget,
)

from roc_mcs.analysis.workflow_history import NodeStatus, ReviewStatus, WorkflowHistory


class _WorkflowNodeItem(QGraphicsObject):
    selected = pyqtSignal(str)
    add_requested = pyqtSignal(str)
    preview_requested = pyqtSignal(str)

    WIDTH = 226.0
    HEIGHT = 88.0

    def __init__(
        self, node, *, can_add: bool, can_preview: bool, is_selected: bool = False
    ):
        super().__init__()
        self.node = node
        self.can_add = bool(can_add)
        self.can_preview = bool(can_preview)
        self._hovered = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setZValue(2)
        if is_selected:
            self.setSelected(True)
        self.setToolTip(
            f"{node.label}\n{node.operation_id}\n"
            f"Execution: {node.status}\nReview: {node.review_status}\n"
            f"Summary: {node.summary or '—'}\n"
            f"Parameters: {node.parameters}\nID: {node.id}\n"
            "Глаз — открыть результат; + — создать следующую ветвь."
        )

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _preview_rect(self) -> QRectF:
        return QRectF(self.WIDTH - 58.0, self.HEIGHT - 28.0, 20.0, 20.0)

    def _plus_rect(self) -> QRectF:
        return QRectF(self.WIDTH - 30.0, self.HEIGHT - 28.0, 20.0, 20.0)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        palette = QApplication.palette()
        window = palette.color(QPalette.ColorRole.Window)
        text = palette.color(QPalette.ColorRole.WindowText)
        muted = palette.color(QPalette.ColorRole.Mid)
        highlight = palette.color(QPalette.ColorRole.Highlight)

        fill = QColor(window)
        fill = fill.lighter(112) if fill.lightness() < 128 else fill.darker(102)
        border = QColor(highlight) if self.isSelected() else QColor(muted)
        border.setAlpha(235 if self.isSelected() else 150)
        if self._hovered and not self.isSelected():
            border = highlight
            border.setAlpha(190)

        state_colors = {
            NodeStatus.PENDING: QColor("#8B96A5"),
            NodeStatus.RUNNING: QColor("#C18A36"),
            NodeStatus.SUCCEEDED: QColor("#3B9A83"),
            NodeStatus.FAILED: QColor("#C15C5C"),
            NodeStatus.CANCELLED: QColor("#89909A"),
        }
        state_color = state_colors.get(self.node.status, muted)
        if self.node.review_status == ReviewStatus.CANDIDATE and self.node.status == NodeStatus.SUCCEEDED:
            review_text = "НА ПРОВЕРКЕ"
            state_color = QColor("#C18A36")
        elif self.node.review_status == ReviewStatus.REJECTED:
            review_text = "ОТКЛОНЁН"
            state_color = QColor("#A45D65")
        elif self.node.review_status == ReviewStatus.ACCEPTED:
            review_text = "ПРИНЯТ"
        else:
            review_text = ""

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, 1.7 if self.isSelected() else 1.0))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(self.boundingRect(), 9.0, 9.0)

        stripe = QRectF(0.0, 8.0, 3.0, self.HEIGHT - 16.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(state_color))
        painter.drawRoundedRect(stripe, 1.5, 1.5)

        title_font = QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSizeF(max(8.0, title_font.pointSizeF()))
        painter.setFont(title_font)
        painter.setPen(text)
        title_rect = QRectF(13.0, 10.0, self.WIDTH - 26.0, 25.0)
        title = QFontMetricsF(title_font).elidedText(
            self.node.label, Qt.TextElideMode.ElideRight, int(title_rect.width())
        )
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title)

        small_font = QFont(painter.font())
        small_font.setBold(False)
        small_font.setPointSizeF(max(7.0, small_font.pointSizeF() - 1.0))
        painter.setFont(small_font)
        painter.setPen(muted)
        operation = self.node.operation_id
        painter.drawText(
            QRectF(13.0, 36.0, self.WIDTH - 26.0, 16.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            operation,
        )

        painter.setPen(state_color)
        state_label = {
            NodeStatus.PENDING: "ОЖИДАЕТ ЗАПУСКА",
            NodeStatus.RUNNING: "ВЫПОЛНЯЕТСЯ",
            NodeStatus.SUCCEEDED: "ВЫПОЛНЕНО",
            NodeStatus.FAILED: "ОШИБКА",
            NodeStatus.CANCELLED: "ОТМЕНЕНО",
        }.get(self.node.status, str(self.node.status).upper())
        if review_text:
            state_label += f"  ·  {review_text}"
        painter.drawText(
            QRectF(13.0, 57.0, self.WIDTH - 94.0, 18.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            state_label,
        )

        if self.can_preview:
            preview = self._preview_rect()
            preview_fill = QColor(highlight)
            preview_fill.setAlpha(25 if self._hovered else 12)
            painter.setPen(QPen(muted, 0.9))
            painter.setBrush(QBrush(preview_fill))
            painter.drawRoundedRect(preview, 5.0, 5.0)
            # Minimal eye glyph: explicit preview affordance without a raster icon.
            cx, cy = preview.center().x(), preview.center().y()
            eye_path = QPainterPath()
            eye_path.moveTo(cx - 6.0, cy)
            eye_path.quadTo(cx, cy - 6.0, cx + 6.0, cy)
            eye_path.quadTo(cx, cy + 6.0, cx - 6.0, cy)
            painter.setPen(QPen(muted, 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(eye_path)
            painter.drawEllipse(QPointF(cx, cy), 1.65, 1.65)

        if self.can_add:
            plus = self._plus_rect()
            plus_fill = QColor(highlight)
            plus_fill.setAlpha(30 if self._hovered else 18)
            painter.setPen(QPen(highlight, 1.0))
            painter.setBrush(QBrush(plus_fill))
            painter.drawRoundedRect(plus, 5.0, 5.0)
            painter.setPen(QPen(highlight, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            cx, cy = plus.center().x(), plus.center().y()
            painter.drawLine(QPointF(cx - 4.0, cy), QPointF(cx + 4.0, cy))
            painter.drawLine(QPointF(cx, cy - 4.0), QPointF(cx, cy + 4.0))

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.node.id)
            if self.can_preview and self._preview_rect().contains(event.pos()):
                self.preview_requested.emit(self.node.id)
                event.accept()
                return
            if self.can_add and self._plus_rect().contains(event.pos()):
                self.add_requested.emit(self.node.id)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.can_preview:
            self.preview_requested.emit(self.node.id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ProcessingGraphView(QGraphicsView):
    """Read-only workflow graph with selection, preview and inline branch creation."""

    node_selected = pyqtSignal(str)
    add_stage_requested = pyqtSignal(str)
    preview_requested = pyqtSignal(str)

    H_GAP = 78.0
    V_GAP = 30.0
    LEVEL_STEP = _WorkflowNodeItem.WIDTH + H_GAP

    def __init__(self, parent: QWidget | None = None):
        self._graph_scene = QGraphicsScene(parent)
        super().__init__(self._graph_scene, parent)
        self._items: dict[str, _WorkflowNodeItem] = {}
        self._history: WorkflowHistory | None = None
        self._selected_id: str | None = None
        self._show_rejected = False
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(300, 300)
        self.setBackgroundBrush(QApplication.palette().color(QPalette.ColorRole.Base))
        self.setFrameShape(QFrame.Shape.NoFrame)

    @property
    def selected_node_id(self) -> str | None:
        return self._selected_id

    def set_history(
        self,
        history: WorkflowHistory,
        *,
        selected_id: str | None = None,
        show_rejected: bool = False,
    ) -> None:
        self._history = history
        self._show_rejected = bool(show_rejected)
        visible_nodes = [
            node for node in history.nodes
            if self._show_rejected or node.review_status != ReviewStatus.REJECTED
        ]
        visible_ids = {node.id for node in visible_nodes}
        self._graph_scene.clear()
        self._items.clear()

        if not visible_nodes:
            self._graph_scene.setSceneRect(QRectF(0, 0, 480, 240))
            self._selected_id = None
            return

        depths: dict[str, int] = {}
        for node in visible_nodes:
            parents = [parent for parent in node.parent_ids if parent in visible_ids]
            depths[node.id] = max((depths.get(parent, 0) + 1 for parent in parents), default=0)

        columns: dict[int, list] = defaultdict(list)
        for node in visible_nodes:
            columns[depths[node.id]].append(node)

        positions: dict[str, QPointF] = {}
        max_y = 0.0
        max_x = 0.0
        selected = selected_id or self._selected_id
        palette = QApplication.palette()
        edge_color = palette.color(QPalette.ColorRole.Mid)
        for depth in sorted(columns):
            for row, node in enumerate(columns[depth]):
                x = 24.0 + depth * self.LEVEL_STEP
                y = 24.0 + row * (_WorkflowNodeItem.HEIGHT + self.V_GAP)
                positions[node.id] = QPointF(x, y)
                max_x = max(max_x, x + _WorkflowNodeItem.WIDTH)
                max_y = max(max_y, y + _WorkflowNodeItem.HEIGHT)

        # Connections are drawn before node cards so they remain visually quiet.
        for node in visible_nodes:
            target_pos = positions[node.id]
            target_center = target_pos.y() + _WorkflowNodeItem.HEIGHT / 2.0
            for parent_id in node.parent_ids:
                if parent_id not in positions:
                    continue
                source_pos = positions[parent_id]
                source_center = source_pos.y() + _WorkflowNodeItem.HEIGHT / 2.0
                start = QPointF(source_pos.x() + _WorkflowNodeItem.WIDTH, source_center)
                end = QPointF(target_pos.x(), target_center)
                bend = max(24.0, (end.x() - start.x()) * 0.45)
                path = QPainterPath(start)
                path.cubicTo(
                    QPointF(start.x() + bend, start.y()),
                    QPointF(end.x() - bend, end.y()),
                    end,
                )
                edge = QGraphicsPathItem(path)
                pen_color = QColor(edge_color)
                pen_color.setAlpha(150)
                edge.setPen(QPen(pen_color, 1.25))
                edge.setZValue(0)
                self._graph_scene.addItem(edge)

                # Small directional arrow: workflow dependencies flow left → right.
                arrow_path = QPainterPath(QPointF(end.x(), end.y()))
                arrow_path.lineTo(QPointF(end.x() - 7.0, end.y() - 4.0))
                arrow_path.lineTo(QPointF(end.x() - 7.0, end.y() + 4.0))
                arrow_path.closeSubpath()
                arrow = QGraphicsPathItem(arrow_path)
                arrow.setPen(QPen(pen_color, 0.6))
                arrow.setBrush(QBrush(pen_color))
                arrow.setZValue(1)
                self._graph_scene.addItem(arrow)

        for node in visible_nodes:
            item = _WorkflowNodeItem(
                node,
                can_preview=node.status == NodeStatus.SUCCEEDED,
                can_add=(
                    node.status == NodeStatus.SUCCEEDED
                    and node.review_status == ReviewStatus.ACCEPTED
                    and node.operation_id != "fit.scan"
                ),
                is_selected=node.id == selected,
            )
            item.setPos(positions[node.id])
            item.selected.connect(self._on_item_selected)
            item.add_requested.connect(self.add_stage_requested)
            item.preview_requested.connect(self.preview_requested)
            self._graph_scene.addItem(item)
            self._items[node.id] = item

        self._graph_scene.setSceneRect(QRectF(0, 0, max_x + 48.0, max_y + 48.0))
        if selected in self._items:
            self._selected_id = selected
            self.select_node(selected)
        elif self._items:
            self._selected_id = next(iter(self._items))
            self.select_node(self._selected_id)
        else:
            self._selected_id = None

    def _on_item_selected(self, node_id: str) -> None:
        self._selected_id = node_id
        for key, item in self._items.items():
            item.setSelected(key == node_id)
        self.node_selected.emit(node_id)

    def select_node(self, node_id: str | None, *, center: bool = False) -> None:
        if node_id is None or node_id not in self._items:
            return
        self._selected_id = node_id
        for key, item in self._items.items():
            item.setSelected(key == node_id)
        if center:
            self.centerOn(self._items[node_id])

    def fit_graph(self) -> None:
        rect = self._graph_scene.sceneRect()
        if rect.isValid():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)
