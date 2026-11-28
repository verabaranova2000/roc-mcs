"""Compact interactive DAG canvas for Processing Workbench.

The persistent model remains WorkflowHistory/DAG. This module only renders it and
forwards select / add / preview gestures to the owning QWidget.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, QPointF, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPalette
from PyQt6.QtWidgets import (
    QApplication, QAbstractScrollArea, QFrame, QGraphicsItem, QGraphicsObject,
    QGraphicsPathItem, QGraphicsScene, QGraphicsView, QSizePolicy, QWidget,
)

from roc_mcs.analysis.workflow_history import NodeStatus, ReviewStatus, WorkflowHistory
from .graph_layout import (
    scaled_graph_viewport_height, vertical_dag_positions, vertical_dag_edge_ports,
)


class _WorkflowNodeItem(QGraphicsObject):
    selected = pyqtSignal(str)
    add_requested = pyqtSignal(str)
    preview_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)

    WIDTH = 242.0
    HEIGHT = 78.0

    def __init__(
        self, node, *, can_add: bool, can_preview: bool, can_delete: bool,
        is_selected: bool = False, input_descriptions: tuple[str, ...] = (),
    ):
        super().__init__()
        self.node = node
        self.can_add = bool(can_add)
        self.can_preview = bool(can_preview)
        self.can_delete = bool(can_delete)
        self._hovered = False
        self._delete_hovered = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setZValue(2)
        if is_selected:
            self.setSelected(True)
        tooltip = (
            f"{node.label}\n{node.operation_id}\n"
            f"Execution: {node.status}\nReview: {node.review_status}\n"
            f"Summary: {node.summary or '—'}\n"
            f"Parameters: {node.parameters}\nID: {node.id}\n"
            "Глаз — открыть результат; + — добавить следующий этап; × — удалить этот узел и все зависимые от него этапы."
        )
        if input_descriptions:
            tooltip += "\n\nСохранённые входы операции:\n" + "\n".join(
                f"• {description}" for description in input_descriptions
            )
        self.setToolTip(tooltip)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _preview_rect(self) -> QRectF:
        return QRectF(self.WIDTH - 58.0, self.HEIGHT - 28.0, 20.0, 20.0)

    def _plus_rect(self) -> QRectF:
        return QRectF(self.WIDTH - 30.0, self.HEIGHT - 28.0, 20.0, 20.0)

    def _delete_rect(self) -> QRectF:
        return QRectF(self.WIDTH - 23.0, 5.0, 16.0, 16.0)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        views = self.scene().views() if self.scene() is not None else []
        palette = views[0].palette() if views else QApplication.palette()
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
        title_rect = QRectF(13.0, 10.0, self.WIDTH - (49.0 if self.can_delete else 26.0), 25.0)
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
            preview_fill.setAlpha(58 if self._hovered else 38)
            painter.setPen(QPen(highlight, 1.0))
            painter.setBrush(QBrush(preview_fill))
            painter.drawRoundedRect(preview, 5.0, 5.0)
            # Minimal eye glyph: explicit preview affordance without a raster icon.
            cx, cy = preview.center().x(), preview.center().y()
            eye_path = QPainterPath()
            eye_path.moveTo(cx - 6.0, cy)
            eye_path.quadTo(cx, cy - 6.0, cx + 6.0, cy)
            eye_path.quadTo(cx, cy + 6.0, cx - 6.0, cy)
            eye_color = QColor(text)
            eye_color.setAlpha(235)
            painter.setPen(QPen(eye_color, 1.25))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(eye_path)
            painter.drawEllipse(QPointF(cx, cy), 1.65, 1.65)

        if self.can_add:
            plus = self._plus_rect()
            plus_fill = QColor(highlight)
            plus_fill.setAlpha(58 if self._hovered else 38)
            painter.setPen(QPen(highlight, 1.0))
            painter.setBrush(QBrush(plus_fill))
            painter.drawRoundedRect(plus, 5.0, 5.0)
            glyph_color = QColor(text)
            glyph_color.setAlpha(245)
            painter.setPen(QPen(glyph_color, 1.85, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            cx, cy = plus.center().x(), plus.center().y()
            painter.drawLine(QPointF(cx - 4.0, cy), QPointF(cx + 4.0, cy))
            painter.drawLine(QPointF(cx, cy - 4.0), QPointF(cx, cy + 4.0))

        if self.can_delete:
            remove_rect = self._delete_rect()
            remove_color = QColor("#D84F5B")
            fill = QColor(remove_color)
            fill.setAlpha(64 if self._delete_hovered else 24)
            painter.setPen(QPen(remove_color, 0.8 if self._delete_hovered else 0.55))
            painter.setBrush(QBrush(fill))
            painter.drawRoundedRect(remove_rect, 4.0, 4.0)
            inset = 5.0 if self._delete_hovered else 5.25
            cx, cy = remove_rect.center().x(), remove_rect.center().y()
            painter.setPen(QPen(remove_color, 1.65, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(cx - inset / 2, cy - inset / 2), QPointF(cx + inset / 2, cy + inset / 2))
            painter.drawLine(QPointF(cx + inset / 2, cy - inset / 2), QPointF(cx - inset / 2, cy + inset / 2))

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event) -> None:
        delete_hovered = self.can_delete and self._delete_rect().contains(event.pos())
        if delete_hovered != self._delete_hovered:
            self._delete_hovered = delete_hovered
            self.update(self._delete_rect())
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self._delete_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.node.id)
            if self.can_delete and self._delete_rect().contains(event.pos()):
                self.delete_requested.emit(self.node.id)
                event.accept()
                return
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
    delete_requested = pyqtSignal(str)
    scene_geometry_changed = pyqtSignal()
    view_transform_changed = pyqtSignal()

    H_GAP = 34.0
    V_GAP = 38.0

    def __init__(self, parent: QWidget | None = None):
        self._graph_scene = QGraphicsScene(parent)
        super().__init__(self._graph_scene, parent)
        self._items: dict[str, _WorkflowNodeItem] = {}
        self._history: WorkflowHistory | None = None
        self._selected_id: str | None = None
        self._show_rejected = False
        self.setObjectName("processingGraphView")
        self._vertical_scroll_enabled = True
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
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        # QGraphicsView::sizeHint() is derived from transform.mapRect(sceneRect()).
        # That means Ctrl+wheel zoom changes the widget's preferred size; in the
        # ROI sidebar this propagates through QSplitter -> Workbench -> QScrollArea
        # and creates large blank regions around the workflow.  Keep the viewport
        # layout size independent of the scene transform.
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(300, 300)
        self.setBackgroundBrush(self.palette().color(QPalette.ColorRole.Base))
        self.setFrameShape(QFrame.Shape.NoFrame)

    def sizeHint(self) -> QSize:
        """Stable preferred viewport size; never derive it from zoom/scene size."""
        return QSize(420, 300)

    @property
    def selected_node_id(self) -> str | None:
        return self._selected_id

    def set_history(
        self,
        history: WorkflowHistory,
        *,
        selected_id: str | None = None,
        show_rejected: bool = False,
        addable_node_ids: set[str] | None = None,
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
            QTimer.singleShot(0, self._finish_scene_update)
            return

        positions_xy = vertical_dag_positions(
            visible_nodes, width=_WorkflowNodeItem.WIDTH,
            height=_WorkflowNodeItem.HEIGHT, horizontal_gap=self.H_GAP,
            vertical_gap=self.V_GAP, origin_x=22.0, origin_y=20.0,
        )
        positions = {key: QPointF(x, y) for key, (x, y) in positions_xy.items()}
        max_x = max((point.x() + _WorkflowNodeItem.WIDTH for point in positions.values()), default=0.0)
        max_y = max((point.y() + _WorkflowNodeItem.HEIGHT for point in positions.values()), default=0.0)
        selected = selected_id or self._selected_id
        palette = self.palette()
        edge_color = palette.color(QPalette.ColorRole.Mid)
        edge_ports = vertical_dag_edge_ports(
            visible_nodes,
            {key: (point.x(), point.y()) for key, point in positions.items()},
            width=_WorkflowNodeItem.WIDTH,
            height=_WorkflowNodeItem.HEIGHT,
            port_margin=22.0,
        )

        # The canvas renders the transitive reduction: redundant ancestor
        # shortcuts are omitted, while the exact direct inputs remain visible
        # in each node's tooltip.
        nodes_by_id = {node.id: node for node in visible_nodes}
        history_nodes_by_id = {node.id: node for node in history.nodes}
        input_descriptions_by_node: dict[str, tuple[str, ...]] = {}
        for node in visible_nodes:
            descriptions = []
            for port, reference in node.input_refs.items():
                source_node = history_nodes_by_id.get(reference.get("node_id", ""))
                source_label = (
                    source_node.label if source_node is not None
                    else reference.get("node_id", "неизвестный узел")
                )
                descriptions.append(
                    f"{port} ← {source_label} [{reference.get('output', '?')}]"
                )
            input_descriptions_by_node[node.id] = tuple(descriptions)
        for (parent_id, child_id), (start_xy, end_xy) in edge_ports.items():
            start = QPointF(*start_xy)
            end = QPointF(*end_xy)
            bend = max(18.0, (end.y() - start.y()) * 0.45)
            path = QPainterPath(start)
            path.cubicTo(QPointF(start.x(), start.y() + bend),
                         QPointF(end.x(), end.y() - bend), end)
            pen_color = QColor(edge_color)
            pen_color.setAlpha(175)
            edge = QGraphicsPathItem(path)
            child = nodes_by_id[child_id]
            input_ports = [
                port for port, ref in child.input_refs.items()
                if ref.get("node_id") == parent_id
            ]
            if input_ports:
                edge.setPen(QPen(pen_color, 1.35, Qt.PenStyle.SolidLine))
                tooltip = "Вычислительный вход: " + ", ".join(input_ports)
            else:
                edge.setPen(QPen(pen_color, 1.15, Qt.PenStyle.DashLine))
                tooltip = (
                    "Связь происхождения ветви: прямой вход от этого узла "
                    "операцией не читается."
                )
            if input_descriptions_by_node[child_id]:
                tooltip += "\nВсе прямые входы операции перечислены в подсказке карточки."
            edge.setToolTip(tooltip)
            edge.setZValue(0)
            self._graph_scene.addItem(edge)

            arrow_path = QPainterPath(end)
            arrow_path.lineTo(QPointF(end.x() - 4.0, end.y() - 7.0))
            arrow_path.lineTo(QPointF(end.x() + 4.0, end.y() - 7.0))
            arrow_path.closeSubpath()
            arrow = QGraphicsPathItem(arrow_path)
            arrow.setPen(QPen(pen_color, 0.6))
            arrow.setBrush(QBrush(pen_color))
            arrow.setZValue(1)
            self._graph_scene.addItem(arrow)

        for node in visible_nodes:
            item = _WorkflowNodeItem(
                node,
                input_descriptions=input_descriptions_by_node[node.id],
                can_preview=node.status == NodeStatus.SUCCEEDED,
                can_delete=(node.operation_id != "source.scan" and node.status != NodeStatus.RUNNING),
                can_add=(
                    (
                        node.status == NodeStatus.SUCCEEDED
                        and node.review_status == ReviewStatus.ACCEPTED
                        and node.operation_id not in {"peaks.branch_correlation"}
                    )
                    if addable_node_ids is None
                    else node.id in addable_node_ids
                ),
                is_selected=node.id == selected,
            )
            item.setPos(positions[node.id])
            item.selected.connect(self._on_item_selected)
            item.add_requested.connect(self.add_stage_requested)
            item.preview_requested.connect(self.preview_requested)
            item.delete_requested.connect(self.delete_requested)
            self._graph_scene.addItem(item)
            self._items[node.id] = item

        self._graph_scene.setSceneRect(QRectF(0, 0, max_x + 26.0, max_y + 28.0))
        if selected in self._items:
            self._selected_id = selected
            self.select_node(selected)
        elif self._items:
            self._selected_id = next(iter(self._items))
            self.select_node(self._selected_id)
        else:
            self._selected_id = None
        # Refit only after the scene geometry is committed, then notify the host.
        # An embedded host in automatic full-height mode can now recalculate its
        # requested height from the new scene and the resulting view transform.
        QTimer.singleShot(0, self._finish_scene_update)

    def _finish_scene_update(self) -> None:
        self.fit_graph()
        self.scene_geometry_changed.emit()

    def required_viewport_height(self) -> int:
        """Height needed to display the complete DAG at the *current* zoom.

        This is derived from the scene's logical extent and current transform,
        never from the widget's current height.  Consequently shrinking the
        sidebar cannot feed back into an oversized or undersized expansion
        request, and zooming out proportionally reduces the requested height.
        """
        scene_rect = self._graph_scene.sceneRect()
        if not scene_rect.isValid() or scene_rect.height() <= 0.0:
            return 0
        mapped_rect = self.transform().mapRect(scene_rect)
        horizontal_extra = 0
        # Predict the horizontal scrollbar from the transformed scene width
        # rather than its current range: immediately after a zoom event Qt may
        # not have recalculated scrollbar ranges yet.
        if mapped_rect.width() > max(1, int(self.viewport().width())):
            horizontal_extra = max(0, int(self.horizontalScrollBar().sizeHint().height()))
        frame_extra = max(0, 2 * int(self.frameWidth())) + 12
        return scaled_graph_viewport_height(
            mapped_rect.height(),
            1.0,
            padding=frame_extra,
            extra=horizontal_extra,
        )

    @property
    def vertical_scroll_enabled(self) -> bool:
        return self._vertical_scroll_enabled

    def set_vertical_scroll_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._vertical_scroll_enabled:
            return
        self._vertical_scroll_enabled = enabled
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded if enabled
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded if enabled
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.fit_graph()

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
        """Fit graph width, optionally fitting its complete height as well.

        Scroll mode keeps node text readable and leaves long histories vertically
        scrollable. Fit-all mode scales the whole DAG into the viewport and hides
        scroll bars; it never enlarges the graph beyond its native 100% scale.
        """
        rect = self._graph_scene.sceneRect()
        viewport = self.viewport()
        if not rect.isValid() or viewport.width() <= 0 or viewport.height() <= 0:
            return
        self.resetTransform()
        available_w = max(1, viewport.width() - 12)
        available_h = max(1, viewport.height() - 12)
        scale_w = available_w / max(1.0, rect.width())
        if self._vertical_scroll_enabled:
            scale = min(1.0, scale_w)
            self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        else:
            scale_h = available_h / max(1.0, rect.height())
            scale = min(1.0, scale_w, scale_h)
            self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.scale(scale, scale)
        self.centerOn(rect.center())
        if self._vertical_scroll_enabled:
            # The first node remains at the top when the viewport is shorter
            # than the graph; users can scroll through the complete history.
            self.verticalScrollBar().setValue(self.verticalScrollBar().minimum())
        self.view_transform_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._vertical_scroll_enabled:
            QTimer.singleShot(0, self.fit_graph)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            if delta:
                factor = 1.15 if delta > 0 else (1.0 / 1.15)
                self.scale(factor, factor)
                self.view_transform_changed.emit()
            event.accept()
            return

        # In the compact sidebar the graph is nested inside the outer
        # leftSidebarScroll. QWidget's default wheelEvent ignores an event at
        # the scrollbar limit, which then bubbles to the outer area and moves
        # the whole sidebar away from the graph. Own the complete wheel gesture
        # whenever the inner DAG really has a vertical range, including at both
        # limits. If it has no range (full-height mode), deliberately ignore the
        # event so the outer sidebar can scroll instead.
        if self._vertical_scroll_enabled:
            bar = self.verticalScrollBar()
            has_internal_range = bar.maximum() > bar.minimum()
            if has_internal_range:
                pixel_delta = int(event.pixelDelta().y())
                angle_delta = int(event.angleDelta().y())
                if pixel_delta:
                    scroll_delta = pixel_delta
                elif angle_delta:
                    # One wheel notch follows the familiar three-line rhythm;
                    # use at least 20 logical pixels even if the style reports
                    # a one-pixel singleStep for a graphics view.
                    line_step = max(20, int(bar.singleStep()))
                    scroll_delta = int(round((angle_delta / 120.0) * line_step * 3))
                else:
                    scroll_delta = 0

                if scroll_delta:
                    bar.setValue(int(bar.value()) - scroll_delta)
                # Also consume zero-delta begin/end events from high-resolution
                # touchpads, otherwise their gesture may leak to the parent.
                if scroll_delta or event.phase() != Qt.ScrollPhase.NoScrollPhase:
                    event.accept()
                    return

        event.ignore()
