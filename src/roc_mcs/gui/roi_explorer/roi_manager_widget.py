from pathlib import Path
from PyQt6.QtCore import QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPalette
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout, QHeaderView, QLabel, QMenu, QScrollArea, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget, QSizePolicy, QSizeGrip)

from roc_mcs.gui.roi_explorer.hex_color_picker import CompactColorPicker, _DragTitle
from roc_mcs.gui.roi_explorer.plot_toolbar_style import ICON_DIR
from roc_mcs.gui.components.icon_provider import svg_icon
from roc_mcs.gui.roi_explorer.theme import apply_widget_palette, apply_popup_theme, resolve_theme

def _symbol_icon(kind: str, color: QColor, size: int = 16) -> QIcon:
    pix = QPixmap(size, size); pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(color, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    r = size - 3
    if kind == "check":
        p.drawLine(3, size // 2, size // 2 - 1, size - 4); p.drawLine(size // 2 - 1, size - 4, size - 3, 3)
    elif kind == "cross":
        p.drawLine(4, 4, size - 4, size - 4); p.drawLine(size - 4, 4, 4, size - 4)
    elif kind == "trash":
        p.drawLine(3, 4, size - 3, 4); p.drawLine(6, 2, size - 6, 2)
        p.drawLine(4, 5, 5, size - 3); p.drawLine(size - 4, 5, size - 5, size - 3)
        p.drawLine(5, size - 3, size - 5, size - 3); p.drawLine(size // 2, 6, size // 2, size - 5)
    elif kind in ("left", "right"):
        cx, cy = size // 2, size // 2
        if kind == "right":
            p.drawLine(cx - 2, cy - 4, cx + 2, cy); p.drawLine(cx + 2, cy, cx - 2, cy + 4)
        else:
            p.drawLine(cx + 2, cy - 4, cx - 2, cy); p.drawLine(cx - 2, cy, cx + 2, cy + 4)
    p.end(); return QIcon(pix)


# _ROI_ICON_DIR = Path(r"C:\Users\Baranova Vera\Downloads\рентген практика\h5 файлы\Новая папка\icons")

# _ROI_ICON_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"

def _pane_toggle_icon(expanded: bool, color: QColor | None = None, size: int = 16) -> QIcon:
    """Загружает векторную иконку нужного состояния; fallback остаётся чётким."""
    path = ICON_DIR / ("pane_expanded.svg" if expanded else "pane_collapsed.svg")
    if path.is_file(): return svg_icon(path, size=size, color=color or QApplication.palette().color(QPalette.ColorRole.WindowText))
    pix = QPixmap(size, size); pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(color or QColor(130, 130, 130), 1.2)); p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(2, 2, size - 4, size - 4, 2, 2)
    x = round(size * (15 / 24 if expanded else 9 / 24)); p.drawLine(x, 3, x, size - 3); p.end()
    return QIcon(pix)


class RoiTableWidget(QTableWidget):
    """Плотный список ROI для режима подробного редактирования."""
    roi_edited = pyqtSignal(int, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6); self.setHorizontalHeaderLabels(["Label", "X", "Y", "W", "H", "Ang"])
        self.verticalHeader().setVisible(False); self.verticalHeader().setDefaultSectionSize(18)
        self.setAlternatingRowColors(True); self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.horizontalHeader(); header.setMinimumSectionSize(25); header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        for col, width in enumerate((76, 31, 31, 34, 34, 38)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed); self.setColumnWidth(col, width)
        self.setFixedWidth(260); self.setMinimumHeight(95)
        self.setStyleSheet("""
            QTableWidget { font-size: 9px; border: 1px solid palette(mid); gridline-color: palette(midlight); }
            QHeaderView::section { padding: 1px 2px; min-height: 17px; font-weight: 600; font-size: 9px; border: 0; border-right: 1px solid palette(midlight); border-bottom: 1px solid palette(mid); }
            QTableWidget::item { padding: 0 1px; }
            QTableWidget::item:selected { background: palette(highlight); color: palette(highlighted-text); }
        """)
        self.itemChanged.connect(self._on_cell_changed)

    def _on_cell_changed(self, item: QTableWidgetItem) -> None:
        id_item = self.item(item.row(), 0)
        if id_item is None: return
        roi_id = id_item.data(Qt.ItemDataRole.UserRole)
        field = {0: "label", 1: "cx", 2: "cy", 3: "width", 4: "height", 5: "angle"}.get(item.column())
        if field is not None and roi_id is not None: self.roi_edited.emit(int(roi_id), field, item.text().strip())


class _RoiSwatchButton(QToolButton):
    context_requested = pyqtSignal()
    long_press = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent); self.setFixedSize(14, 14); self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._right_down = False; self._long_fired = False
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(750); self._timer.timeout.connect(self._on_long_press)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._right_down = True; self._long_fired = False; self._timer.start(); event.accept(); return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._right_down:
            self._timer.stop(); self._right_down = False
            if not self._long_fired: self.context_requested.emit()
            event.accept(); return
        super().mouseReleaseEvent(event)

    def _on_long_press(self):
        if self._right_down: self._long_fired = True; self.long_press.emit()


class _RoiEditorPopup(QFrame):
    """Редактор ROI в виде плотной таблицы; значения применяются при завершении редактирования."""
    save_requested = pyqtSignal(int, dict)
    delete_requested = pyqtSignal(int)
    FIELDS = (("Label", "label"), ("X", "cx"), ("Y", "cy"), ("W", "width"), ("H", "height"), ("Angle", "angle"))

    def __init__(self, saved_roi, color: str, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.roi_id = int(saved_roi.id); self._roi_type = str(saved_roi.roi.type); self._color = color; self._busy = False
        self.setObjectName("roiEditorPopup"); apply_widget_palette(self, resolve_theme(parent))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True); self.setMinimumWidth(190); self.setMaximumWidth(360); self._drag_offset = None
        self.setStyleSheet("""
            QFrame#roiEditorPopup { background: palette(window); color: palette(window-text); border: 1px solid palette(mid); border-radius: 4px; }
            QLabel { border: none; font-size: 9px; }
            QToolButton { border: 1px solid transparent; border-radius: 3px; padding: 0; }
            QToolButton:hover { background: palette(midlight); border-color: palette(mid); }
            QTableWidget { font-size: 9px; border: 1px solid palette(mid); gridline-color: palette(midlight); }
            QHeaderView::section { padding: 1px 2px; min-height: 17px; font-weight: 600; font-size: 9px; border: 0; border-right: 1px solid palette(midlight); border-bottom: 1px solid palette(mid); }
            QTableWidget::item { padding: 0 1px; }
            QTableWidget::item:selected { background: palette(highlight); color: palette(highlighted-text); }
        """)
        root = QVBoxLayout(self); root.setContentsMargins(5, 4, 4, 3); root.setSpacing(3)
        head = QHBoxLayout(); head.setContentsMargins(0, 0, 0, 0); head.setSpacing(4)
        title = _DragTitle(f"ROI {self.roi_id}", self); title.setStyleSheet("font-size: 9px; font-weight: 600; border: none;")
        self.color_button = QToolButton(); self.color_button.setFixedSize(15, 15); self.color_button.setToolTip("Choose ROI color"); self._set_color_button(color); self.color_button.clicked.connect(self._open_color_picker)
        self.close_button = QToolButton(self); self.close_button.setObjectName("roiEditorCloseButton"); self.close_button.setText("×"); self.close_button.setFixedSize(18, 18); self.close_button.setToolTip("Close ROI editor"); self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.close_button.clicked.connect(self.close)
        head.addWidget(title); head.addStretch(1); head.addWidget(self.color_button); head.addWidget(self.close_button); root.addLayout(head)
        self.table = QTableWidget(len(self.FIELDS), 2, self); self.table.setHorizontalHeaderLabels(["Field", "Value"])
        self.table.verticalHeader().setVisible(False); self.table.verticalHeader().setDefaultSectionSize(18); self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems); self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.horizontalHeader().setMinimumSectionSize(30); self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed); self.table.setColumnWidth(0, 56); self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumWidth(178); self.table.setMinimumHeight(130); self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._last_valid = {"label": str(saved_roi.roi.label), "cx": float(saved_roi.roi.cx), "cy": float(saved_roi.roi.cy), "width": float(saved_roi.roi.width), "height": float(saved_roi.roi.height), "angle": float(saved_roi.roi.angle), "color": QColor(color).name()}
        for row, (caption, field) in enumerate(self.FIELDS):
            key_item = QTableWidgetItem(caption); key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable); key_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            value = self._last_valid[field]; text = str(value) if field == "label" else f"{value:.6g}"
            value_item = QTableWidgetItem(text); value_item.setData(Qt.ItemDataRole.UserRole, field)
            if (self._roi_type == "point" and field in ("width", "height")) or (self._roi_type == "line" and field == "height"):
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, key_item); self.table.setItem(row, 1, value_item)
        root.addWidget(self.table); self.table.itemChanged.connect(self._on_item_changed)
        footer = QHBoxLayout(); footer.setContentsMargins(0, 0, 0, 0); footer.setSpacing(2)
        self.delete_button = QToolButton(); self.delete_button.setIcon(_symbol_icon("trash", self.palette().color(QPalette.ColorRole.WindowText), 14)); self.delete_button.setIconSize(QSize(12, 12)); self.delete_button.setFixedSize(19, 18); self.delete_button.setToolTip("Delete ROI"); self.delete_button.clicked.connect(self._delete)
        footer.addWidget(self.delete_button); footer.addStretch(1); self.size_grip = QSizeGrip(self); self.size_grip.setFixedSize(10, 10); footer.addWidget(self.size_grip); root.addLayout(footer)
        self.resize(194, 176)
        apply_popup_theme(self, resolve_theme(parent))

    def apply_theme(self, theme: str | None = None) -> None:
        apply_popup_theme(self, theme)
        if hasattr(self, "delete_button"):
            self.delete_button.setIcon(_symbol_icon("trash", self.palette().color(QPalette.ColorRole.WindowText), 14))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 23:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft(); event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset); event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self._drag_offset = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape: self.close(); event.accept(); return
        super().keyPressEvent(event)

    def _set_color_button(self, color):
        self.color_button.setStyleSheet(f"QToolButton {{ background: {color}; border: 1px solid palette(mid); border-radius: 2px; }} QToolButton:hover {{ border: 2px solid palette(highlight); }}")

    def _open_color_picker(self):
        picker = CompactColorPicker(self._color, self); self._color_picker = picker; picker.destroyed.connect(lambda *_ , p=picker: self._forget_color_picker(p)); picker.color_selected.connect(self._on_color_selected)
        pos = self.color_button.mapToGlobal(QPoint(0, self.color_button.height() + 3)); _place_popup(picker, pos); picker.show()

    def _forget_color_picker(self, picker):
        if self._color_picker is picker: self._color_picker = None

    def _on_color_selected(self, color):
        if not QColor(color).isValid(): return
        self._color = QColor(color).name(); self._last_valid["color"] = self._color; self._set_color_button(self._color); self.save_requested.emit(self.roi_id, dict(self._last_valid))

    def _on_item_changed(self, item):
        if self._busy or item.column() != 1: return
        field = item.data(Qt.ItemDataRole.UserRole)
        if field not in self._last_valid: return
        text = item.text().strip(); candidate = dict(self._last_valid)
        try:
            candidate[field] = text if field == "label" else float(text.replace(",", "."))
            if field == "width" and candidate[field] < 0 or field == "height" and candidate[field] < 0: raise ValueError
            if self._roi_type in ("rect", "rectangle", "rotated_rectangle") and (candidate["width"] <= 0 or candidate["height"] <= 0): raise ValueError
            if self._roi_type == "line" and candidate["width"] <= 0: raise ValueError
        except (TypeError, ValueError):
            self._busy = True; item.setText(str(self._last_valid[field]) if field == "label" else f"{self._last_valid[field]:.6g}"); self._busy = False; return
        self._last_valid = candidate; self.save_requested.emit(self.roi_id, dict(candidate))

    def _delete(self):
        self.delete_requested.emit(self.roi_id); self.close()


def _place_popup(popup, pos: QPoint):
    popup.adjustSize(); screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry(); max_x = area.right() - popup.width() + 1; max_y = area.bottom() - popup.height() + 1
        x = min(max(pos.x(), area.left()), max_x); y = min(max(pos.y(), area.top()), max_y); pos = QPoint(x, y)
    popup.move(pos)


class RoiManagerWidget(QWidget):
    """Компактная рейка ROI с точечным редактором и таблицей по запросу."""
    edit_requested = pyqtSignal(int, dict)
    color_changed = pyqtSignal(int, str)
    delete_requested = pyqtSignal(int)
    export_requested = pyqtSignal(int)
    copy_requested = pyqtSignal(int, str)
    expanded_changed = pyqtSignal(bool)
    _FALLBACK_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")

    def __init__(self, parent=None):
        super().__init__(parent); self.table = RoiTableWidget(self); self._rois = []; self._row_by_id = {}; self._buttons = {}; self._selected_id = None; self._expanded = False; self._color_resolver = None; self._active_editor = None; self.setObjectName("compactRoiManager"); self.setFixedWidth(22); self.setMinimumHeight(60); self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(2)
        self.rail = QWidget(self); self.rail.setObjectName("roiPaletteRail"); self.rail.setFixedWidth(20); self.rail.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        rail_layout = QVBoxLayout(self.rail); rail_layout.setContentsMargins(0, 0, 0, 0); rail_layout.setSpacing(1)
        self.expand_button = QToolButton(self.rail); self.expand_button.setObjectName("roiExpandButton"); self.expand_button.setIcon(_pane_toggle_icon(False, self.palette().color(QPalette.ColorRole.WindowText))); self.expand_button.setIconSize(QSize(14, 14)); self.expand_button.setToolTip("Expand ROI list"); self.expand_button.setAutoRaise(True); self.expand_button.setFixedSize(20, 20); self.expand_button.setCursor(Qt.CursorShape.PointingHandCursor); self.expand_button.setStyleSheet("QToolButton { background: transparent; border: none; border-radius: 5px; padding: 1px; } QToolButton:hover { background: palette(midlight); }")
        rail_layout.addWidget(self.expand_button, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.palette_scroll = QScrollArea(self.rail); self.palette_scroll.setObjectName("roiPaletteScroll"); self.palette_scroll.setWidgetResizable(True); self.palette_scroll.setFixedWidth(20); self.palette_scroll.setFrameShape(QFrame.Shape.NoFrame); self.palette_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.palette_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.palette_scroll.setStyleSheet("QScrollArea { border: none; background: palette(window); }")
        self.palette_scroll.viewport().setObjectName("roiPaletteViewport")
        self.palette_body = QWidget(); self.palette_body.setObjectName("roiPaletteBody"); self.palette_layout = QVBoxLayout(self.palette_body); self.palette_layout.setContentsMargins(2, 2, 2, 2); self.palette_layout.setSpacing(2); self.palette_layout.setAlignment(Qt.AlignmentFlag.AlignTop); self.palette_scroll.setWidget(self.palette_body); rail_layout.addWidget(self.palette_scroll, 1)
        root.addWidget(self.rail); root.addWidget(self.table, 0); self.table.hide(); self.table.itemSelectionChanged.connect(self._sync_selection_from_table); self.expand_button.clicked.connect(self._toggle_expanded)

    def apply_theme(self, _theme: str | None = None) -> None:
        active_theme = apply_widget_palette(self, _theme)
        color = self.palette().color(QPalette.ColorRole.WindowText)
        self.expand_button.setIcon(_pane_toggle_icon(self._expanded, color))
        if self._active_editor is not None: self._active_editor.apply_theme(active_theme)
        if getattr(self, "_color_picker", None) is not None: self._color_picker.apply_theme(active_theme)

    def _toggle_expanded(self):
        self._expanded = not self._expanded; self.table.setVisible(self._expanded)
        if self._expanded:
            self.setFixedWidth(282); self.expand_button.setIcon(_pane_toggle_icon(True, self.palette().color(QPalette.ColorRole.WindowText))); self.expand_button.setToolTip("Collapse ROI list")
        else:
            self.setFixedWidth(22); self.expand_button.setIcon(_pane_toggle_icon(False, self.palette().color(QPalette.ColorRole.WindowText))); self.expand_button.setToolTip(f"Expand ROI list ({len(self._rois)})")
        self.expanded_changed.emit(self._expanded)

    def _clear_palette(self):
        while self.palette_layout.count():
            item = self.palette_layout.takeAt(0); widget = item.widget()
            if widget is not None: widget.deleteLater()
        self._buttons.clear()

    def _color_for(self, saved_roi):
        color = str(getattr(saved_roi.roi, "color", "") or "").strip()
        if QColor(color).isValid(): return color
        if callable(self._color_resolver): return str(self._color_resolver(int(saved_roi.id)))
        return self._FALLBACK_COLORS[(int(saved_roi.id) - 1) % len(self._FALLBACK_COLORS)]

    def sync_compact_view(self, rois=None, selected_id=None, color_resolver=None):
        if rois is not None: self._rois = list(rois)
        self._selected_id = selected_id
        if color_resolver is not None: self._color_resolver = color_resolver
        self._row_by_id = {}; row_data = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None: continue
            roi_id = item.data(Qt.ItemDataRole.UserRole)
            if roi_id is not None: self._row_by_id[int(roi_id)] = row; row_data[int(roi_id)] = item.text()
        self._clear_palette(); roi_by_id = {int(saved.id): saved for saved in self._rois}
        for roi_id, label in row_data.items():
            saved = roi_by_id.get(roi_id)
            if saved is None: continue
            color = self._color_for(saved); button = _RoiSwatchButton(self.palette_body); button.setObjectName("roiSwatch")
            button.setToolTip(f"{label or f'ROI {roi_id}'}\nLeft click: select\nRight click: actions\nHold right button: edit")
            button.setStyleSheet(self._swatch_style(color, roi_id == self._selected_id))
            button.clicked.connect(lambda _=False, rid=roi_id: self._select_roi(rid))
            button.context_requested.connect(lambda rid=roi_id, b=button: self._show_context_menu(rid, b))
            button.long_press.connect(lambda rid=roi_id, b=button: self._open_editor(rid, b))
            self.palette_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter); self._buttons[roi_id] = button
        self.palette_scroll.setVisible(True); self.expand_button.setToolTip(("Collapse ROI list" if self._expanded else f"Expand ROI list ({len(row_data)})"))

    @staticmethod
    def _swatch_style(color: str, selected: bool) -> str:
        border = "2px solid palette(highlight)" if selected else "1px solid palette(mid)"
        return f"QToolButton#roiSwatch {{ background-color: {color}; border: {border}; border-radius: 3px; padding: 0; }} QToolButton#roiSwatch:hover {{ border: 2px solid palette(highlight); }}"

    def _select_roi(self, roi_id: int):
        row = self._row_by_id.get(int(roi_id))
        if row is None: return
        self.table.selectRow(row); self._selected_id = int(roi_id)
        for current_id, button in self._buttons.items():
            saved = next((r for r in self._rois if int(r.id) == current_id), None)
            if saved is not None: button.setStyleSheet(self._swatch_style(self._color_for(saved), current_id == roi_id))

    def _sync_selection_from_table(self):
        items = self.table.selectedItems()
        self._selected_id = None
        if items:
            item = self.table.item(items[0].row(), 0); value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            self._selected_id = int(value) if value is not None else None
        for roi_id, button in self._buttons.items():
            saved = next((r for r in self._rois if int(r.id) == roi_id), None)
            if saved is not None: button.setStyleSheet(self._swatch_style(self._color_for(saved), roi_id == self._selected_id))

    def _show_context_menu(self, roi_id: int, button):
        self._select_roi(roi_id); menu = QMenu(self); menu.setStyleSheet("QMenu { font-size: 10px; padding: 3px; } QMenu::item { padding: 4px 18px 4px 18px; } QMenu::separator { height: 1px; background: palette(mid); margin: 3px 5px; }"); apply_popup_theme(menu, resolve_theme(self))
        edit_action = menu.addAction("Edit ROI…"); color_action = menu.addAction("Change color…"); menu.addSeparator(); export_action = menu.addAction("Save rocking curve…")
        copy_menu = menu.addMenu("Copy"); metadata_action = copy_menu.addAction("Metadata"); data_action = copy_menu.addAction("Curve data (theta, I)")
        menu.addSeparator(); delete_action = menu.addAction("Delete ROI")
        action = menu.exec(button.mapToGlobal(QPoint(button.width() + 3, 0)))
        if action == edit_action: self._open_editor(roi_id, button)
        elif action == color_action: self._choose_color(roi_id, button)
        elif action == export_action: self.export_requested.emit(int(roi_id))
        elif action == metadata_action: self.copy_requested.emit(int(roi_id), "metadata")
        elif action == data_action: self.copy_requested.emit(int(roi_id), "data")
        elif action == delete_action: self.delete_requested.emit(int(roi_id))

    def _forget_color_picker(self, picker):
        if self._color_picker is picker: self._color_picker = None

    def _choose_color(self, roi_id: int, anchor=None):
        saved = next((r for r in self._rois if int(r.id) == int(roi_id)), None)
        if saved is None: return
        picker = CompactColorPicker(self._color_for(saved), self); self._color_picker = picker; picker.destroyed.connect(lambda *_ , p=picker: self._forget_color_picker(p)); picker.color_selected.connect(lambda value, rid=int(roi_id): self.color_changed.emit(rid, value))
        if anchor is None: anchor = self._buttons.get(int(roi_id), self.expand_button)
        pos = anchor.mapToGlobal(QPoint(anchor.width() + 3, 0)); _place_popup(picker, pos); picker.show()

    def _forget_editor(self, popup):
        if self._active_editor is popup: self._active_editor = None

    def _open_editor(self, roi_id: int, button=None):
        saved = next((r for r in self._rois if int(r.id) == int(roi_id)), None)
        if saved is None: return
        if self._active_editor is not None: self._active_editor.close()
        popup = _RoiEditorPopup(saved, self._color_for(saved), self); self._active_editor = popup
        popup.save_requested.connect(self.edit_requested.emit); popup.delete_requested.connect(self.delete_requested.emit)
        if button is None: button = self._buttons.get(int(roi_id), self.expand_button)
        popup.adjustSize(); anchor_pos = button.mapToGlobal(QPoint(0, 0)); screen = button.screen() or QApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else popup.geometry()
        x = anchor_pos.x() - popup.width() - 6
        if x < area.left() + 4: x = min(anchor_pos.x() + button.width() + 6, area.right() - popup.width() + 1)
        y = anchor_pos.y() + (button.height() - popup.height()) // 2
        _place_popup(popup, QPoint(x, y)); popup.destroyed.connect(lambda _=None, p=popup: self._forget_editor(p)); popup.show(); self._active_editor = popup
        first_item = popup.table.item(0, 1)
        if first_item is not None and first_item.flags() & Qt.ItemFlag.ItemIsEditable:
            popup.table.setCurrentItem(first_item); popup.table.setFocus(Qt.FocusReason.OtherFocusReason); popup.table.editItem(first_item)
