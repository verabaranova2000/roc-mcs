import pyqtgraph as pg

from PyQt6 import QtCore, QtGui, QtWidgets


# -------------------------------------------------------------------
# Подсветка
# -------------------------------------------------------------------


class RocSliceHighlight(QtWidgets.QGraphicsObject):
    """
    Подсветка ROC-среза, рисуемая не внутри ViewBox, а отдельным item'ом сцены.

    Что это дает:
    - не клипается ViewBox;
    - может выходить за границы ROC-map;
    - толщина градиента задана в пикселях и визуально не меняется при zoom/pan.
    """

    def __init__(
        self,
        viewbox: pg.ViewBox,
        color=(100, 220, 255),
        fade_px: int = 24,
        edge_alpha: int = 110,
        parent=None,
    ):
        super().__init__(parent)
        self._vb = viewbox
        self._base_color = QtGui.QColor(*color)
        self._fade_px = max(1, int(fade_px))
        self._edge_alpha = int(edge_alpha)

        # Срез в координатах данных
        self._data_rect = QtCore.QRectF()

        # Текущая геометрия в координатах сцены
        self._scene_rect = QtCore.QRectF()
        self._scene_fade = 0.0

        self.setZValue(1_000_000)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        scene = self._vb.scene()
        if scene is None:
            raise RuntimeError("ViewBox scene is not ready yet")

        # ВАЖНО: добавляем прямо в сцену, не через plotItem.addItem()
        scene.addItem(self)

        # Следим за изменением масштаба/размера view
        self._vb.sigRangeChanged.connect(self._on_view_geometry_changed)
        self._vb.sigResized.connect(self._on_view_geometry_changed)

        self.hide()

    def boundingRect(self) -> QtCore.QRectF:
        if self._scene_rect.isNull() or self._scene_fade <= 0:
            return QtCore.QRectF()

        return QtCore.QRectF(
            self._scene_rect.left(),
            self._scene_rect.top() - self._scene_fade,
            self._scene_rect.width(),
            self._scene_rect.height() + 2.0 * self._scene_fade,
        )

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        if self._scene_rect.isNull() or self._scene_fade <= 0 or not self.isVisible():
            return

        painter.save()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        base = QtGui.QColor(self._base_color)
        opaque = QtGui.QColor(base)
        opaque.setAlpha(self._edge_alpha)

        transparent = QtGui.QColor(base)
        transparent.setAlpha(0)

        x = self._scene_rect.left()
        w = self._scene_rect.width()

        # Верхний градиент: от прозрачного сверху к цветному у границы ROC-среза
        top_rect = QtCore.QRectF(
            x,
            self._scene_rect.top() - self._scene_fade,
            w,
            self._scene_fade,
        )
        top_grad = QtGui.QLinearGradient(top_rect.topLeft(), top_rect.bottomLeft())
        top_grad.setColorAt(0.0, transparent)
        top_grad.setColorAt(1.0, opaque)
        painter.fillRect(top_rect, QtGui.QBrush(top_grad))

        # Нижний градиент: от цветного у границы ROC-среза к прозрачному вниз
        bottom_rect = QtCore.QRectF(
            x,
            self._scene_rect.bottom(),
            w,
            self._scene_fade,
        )
        bottom_grad = QtGui.QLinearGradient(bottom_rect.topLeft(), bottom_rect.bottomLeft())
        bottom_grad.setColorAt(0.0, opaque)
        bottom_grad.setColorAt(1.0, transparent)
        painter.fillRect(bottom_rect, QtGui.QBrush(bottom_grad))

        painter.restore()

    def show_slice(self, left: float, right: float, y_min: float, y_max: float) -> None:
        """
        left/right/y_min/y_max — в координатах данных.
        """
        if right <= left:
            self.hide()
            return

        self._data_rect = QtCore.QRectF(
            QtCore.QPointF(left, y_min),
            QtCore.QPointF(right, y_max),
        ).normalized()

        self._recompute_geometry()
        self.show()
        self.update()

    def refresh(self) -> None:
        """Вызывать после изменения диапазона/масштаба, если подсветка уже видна."""
        if self.isVisible():
            self._recompute_geometry()
            self.update()

    def hide(self) -> None:
        super().hide()
        self.update()

    def _on_view_geometry_changed(self, *args) -> None:
        if self.isVisible():
            self._recompute_geometry()
            self.update()

    def _view(self):
        scene = self._vb.scene()
        if scene is None:
            return None
        views = scene.views()
        return views[0] if views else None

    def _pixel_to_scene_dy(self, pixels: int) -> float:
        view = self._view()
        if view is None:
            return 0.0

        p0 = view.mapToScene(QtCore.QPoint(0, 0))
        p1 = view.mapToScene(QtCore.QPoint(0, max(1, int(pixels))))
        return abs(p1.y() - p0.y())

    def _recompute_geometry(self) -> None:
        view = self._view()
        if view is None or self._data_rect.isNull():
            return

        # Переводим срез из координат данных в координаты сцены
        p1 = self._vb.mapViewToScene(QtCore.QPointF(self._data_rect.left(), self._data_rect.top()))
        p2 = self._vb.mapViewToScene(QtCore.QPointF(self._data_rect.right(), self._data_rect.bottom()))
        scene_rect = QtCore.QRectF(p1, p2).normalized()

        # 24 px -> scene units по вертикали
        scene_fade = self._pixel_to_scene_dy(self._fade_px)
        if scene_fade <= 0:
            scene_fade = 1e-6

        self.prepareGeometryChange()
        self._scene_rect = scene_rect
        self._scene_fade = scene_fade    
