"""Compact, multi-select manager for connected diffraction branches."""
from __future__ import annotations

from typing import Any, Iterable

from PyQt6.QtCore import QAbstractListModel, QItemSelectionModel, QModelIndex, QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QPainter, QPen, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QSizePolicy,
    QStyledItemDelegate,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from roc_mcs.gui.roi_explorer.hex_color_picker import CompactColorPicker
from roc_mcs.gui.roi_explorer.theme import (
    apply_popup_theme,
    apply_widget_palette,
    resolve_theme,
    theme_colors,
)


class _BranchListModel(QAbstractListModel):
    BranchIdRole = int(Qt.ItemDataRole.UserRole) + 1
    ColorRole = int(Qt.ItemDataRole.UserRole) + 2
    SummaryRole = int(Qt.ItemDataRole.UserRole) + 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"Ветвь {row['branch_id']}"
        if role == Qt.ItemDataRole.ToolTipRole:
            included = "включена в анализ" if row["analysis_included"] else "исключена порогом размера"
            theta = "θ: нет конечных значений"
            if row["median_theta"] is not None:
                theta = (
                    f"θ: median {row['median_theta']:.6g}; "
                    f"range {row['theta_min']:.6g}…{row['theta_max']:.6g}"
                )
            return (
                f"Ветвь {row['branch_id']}\n"
                f"Событий: {row['event_count']:,}\n"
                f"Занятых пикселей: {row['occupied_pixels']:,}\n"
                f"{theta}\nСтатус: {included}\n\n"
                "Ctrl+click — добавить/убрать ветвь из выделения; "
                "Shift+click — диапазон; правый клик — действия."
            )
        if role == self.BranchIdRole:
            return int(row["branch_id"])
        if role == self.ColorRole:
            return str(row["color"])
        if role == self.SummaryRole:
            return dict(row)
        return None

    def set_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = [dict(row) for row in rows]
        self.endResetModel()

    def update_color(self, branch_id: int, color: str) -> None:
        for index, row in enumerate(self._rows):
            if int(row["branch_id"]) == int(branch_id):
                row["color"] = str(color)
                model_index = self.index(index, 0)
                self.dataChanged.emit(
                    model_index, model_index,
                    [self.ColorRole, Qt.ItemDataRole.ToolTipRole],
                )
                return

    def set_compact(self, compact: bool) -> None:
        # Item geometry is owned by the delegate; a layout change asks Qt to
        # recalculate the visible rows without rebuilding model identity.
        self.layoutChanged.emit()


class _BranchItemDelegate(QStyledItemDelegate):
    def __init__(self, owner: "BranchManagerWidget"):
        super().__init__(owner)
        self.owner = owner

    def sizeHint(self, option, index):
        return QSize(22 if not self.owner.is_expanded else 240, 24)

    def paint(self, painter: QPainter, option, index):
        row = index.data(_BranchListModel.SummaryRole) or {}
        if not row:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = option.rect
        palette = option.palette
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected:
            painter.fillRect(rect, palette.brush(QPalette.ColorRole.Highlight))
        elif hovered and self.owner.is_expanded:
            painter.fillRect(rect, palette.brush(QPalette.ColorRole.AlternateBase))
        elif self.owner.is_expanded and index.row() % 2:
            painter.fillRect(rect, palette.brush(QPalette.ColorRole.Base))

        color = QColor(str(row.get("color", "#4C9FD6")))
        if not color.isValid():
            color = QColor("#4C9FD6")
        center = rect.left() + (11 if not self.owner.is_expanded else 14)
        radius = 5.2
        included = bool(row.get("analysis_included", True))
        pen_color = palette.color(QPalette.ColorRole.Highlight) if selected else color.darker(125)
        pen = QPen(pen_color, 2.0 if selected else 1.15)
        if not included:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(color if included else color.lighter(145)))
        painter.drawEllipse(
            QPoint(round(center), round(rect.center().y())),
            round(radius), round(radius),
        )

        if self.owner.is_expanded:
            text_color = (
                palette.color(QPalette.ColorRole.HighlightedText)
                if selected else palette.color(QPalette.ColorRole.Text)
            )
            painter.setPen(text_color)
            font = QFont(option.font)
            font.setPointSize(max(8, font.pointSize() - 1))
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            branch_text = f"Ветвь {int(row['branch_id'])}"
            painter.drawText(
                rect.adjusted(29, 0, -88, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                branch_text,
            )
            detail_color = QColor(text_color)
            detail_color.setAlpha(190)
            painter.setPen(detail_color)
            detail_font = QFont(option.font)
            detail_font.setPointSize(max(7, detail_font.pointSize() - 2))
            painter.setFont(detail_font)
            theta = row.get("median_theta")
            detail = (
                f"{int(row['event_count']):,} событий"
                + (f" · θ̄={float(theta):.4g}" if theta is not None else "")
            )
            painter.drawText(
                rect.adjusted(112, 0, -8, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                detail,
            )
        painter.restore()


class BranchManagerWidget(QWidget):
    """A narrow coloured-circle rail that expands into a branch object manager.

    Selection is transient. User colour overrides are stored by the owner in a
    per-analysis display sidecar, never in the scientific branch artifact.
    """

    selection_changed = pyqtSignal(object)       # tuple[int, ...]
    color_changed = pyqtSignal(int, str)
    footprints_toggled = pyqtSignal(bool)
    focus_3d_toggled = pyqtSignal(bool)
    parameter_map_requested = pyqtSignal(int, str)
    expanded_changed = pyqtSignal(bool)

    COMPACT_WIDTH = 22
    EXPANDED_WIDTH = 270

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("branchManager")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(self.COMPACT_WIDTH)
        self._expanded = False
        self._selected_ids: tuple[int, ...] = ()
        self._parameter_fields: list[dict[str, str]] = []
        self._footprints_enabled = False
        self._focus_3d_enabled = False
        self._active_color_picker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        header = QWidget(self)
        header.setObjectName("branchManagerHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(1, 0, 1, 0)
        header_layout.setSpacing(4)
        self.expand_button = QToolButton(header)
        self.expand_button.setObjectName("branchManagerExpand")
        self.expand_button.setFixedSize(20, 20)
        self.expand_button.setAutoRaise(True)
        self.expand_button.setText("›")
        self.expand_button.setToolTip("Развернуть менеджер ветвей")
        self.expand_button.clicked.connect(self._toggle_expanded)
        header_layout.addWidget(self.expand_button)
        self.title_label = QLabel("ВЕТВИ", header)
        self.title_label.setObjectName("branchManagerTitle")
        self.title_label.setStyleSheet("font-weight: 700; font-size: 9px; letter-spacing: 0.6px;")
        header_layout.addWidget(self.title_label)
        self.count_label = QLabel("0", header)
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.count_label.setObjectName("branchManagerCount")
        header_layout.addWidget(self.count_label, 1)
        root.addWidget(header)

        self.view = QListView(self)
        self.view.setObjectName("branchManagerList")
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setUniformItemSizes(True)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setAlternatingRowColors(False)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model = _BranchListModel(self)
        self.view.setModel(self.model)
        self.view.setItemDelegate(_BranchItemDelegate(self))
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        root.addWidget(self.view, 1)

        self.actions_widget = QWidget(self)
        self.actions_widget.setObjectName("branchManagerActions")
        actions = QVBoxLayout(self.actions_widget)
        actions.setContentsMargins(3, 4, 3, 3)
        actions.setSpacing(4)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(4)
        self.footprints_button = QToolButton(self.actions_widget)
        self.footprints_button.setObjectName("branchFootprintsButton")
        self.footprints_button.setText("2D footprints")
        self.footprints_button.setCheckable(True)
        self.footprints_button.setToolTip(
            "Наложить XY-проекции выбранных ветвей: бинарную занятость "
            "пикселей детектора, а не карту интенсивности. Сглаживание "
            "применяется только для отображения; цвета соответствуют кружкам менеджера."
        )
        self.footprints_button.toggled.connect(self._set_footprints)
        buttons.addWidget(self.footprints_button, 1)
        self.focus_3d_button = QToolButton(self.actions_widget)
        self.focus_3d_button.setObjectName("branchFocus3DButton")
        self.focus_3d_button.setText("3D")
        self.focus_3d_button.setCheckable(True)
        self.focus_3d_button.setToolTip(
            "Сфокусировать 3D на центрах событий и рёбрах выбранных ветвей. "
            "Маркеры и рёбра остальных ветвей скрываются; интенсивностный "
            "объём остаётся контекстным фоном и не сегментируется по branch_id."
        )
        self.focus_3d_button.toggled.connect(self._set_focus_3d)
        buttons.addWidget(self.focus_3d_button)
        actions.addLayout(buttons)

        self.parameter_button = QToolButton(self.actions_widget)
        self.parameter_button.setObjectName("branchParameterButton")
        self.parameter_button.setText("Карта параметра  ▾")
        self.parameter_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.parameter_menu = QMenu(self.parameter_button)
        self.parameter_button.setMenu(self.parameter_menu)
        actions.addWidget(self.parameter_button)

        self.clear_button = QToolButton(self.actions_widget)
        self.clear_button.setObjectName("branchClearSelectionButton")
        self.clear_button.setText("Снять выделение")
        self.clear_button.setToolTip("Снять выделение всех ветвей")
        self.clear_button.clicked.connect(self.clear_selection)
        actions.addWidget(self.clear_button)
        root.addWidget(self.actions_widget)
        self.actions_widget.hide()
        self._sync_parameter_menu()
        self._sync_enabled_actions()
        self.apply_theme()

    @property
    def is_expanded(self) -> bool:
        return bool(self._expanded)

    @property
    def selected_branch_ids(self) -> tuple[int, ...]:
        return self._selected_ids

    @property
    def footprints_enabled(self) -> bool:
        return bool(self._footprints_enabled)

    @property
    def focus_3d_enabled(self) -> bool:
        return bool(self._focus_3d_enabled)

    def apply_theme(self, theme: str | None = None) -> None:
        active = apply_widget_palette(self, theme)
        colors = theme_colors(active)
        self.setStyleSheet(
            "QWidget#branchManager {"
            f"background:{colors['window']}; border-right:1px solid {colors['border']};"
            "}"
            "QWidget#branchManagerHeader {"
            f"background:{colors['alternate']}; border-bottom:1px solid {colors['border']};"
            "}"
            f"QLabel#branchManagerTitle, QLabel#branchManagerCount {{ color:{colors['text']}; }}"
            "QListView#branchManagerList {"
            f"background:{colors['window']}; border:0; color:{colors['text']};"
            "outline:0; padding:0;"
            "}"
            "QToolButton {"
            f"color:{colors['text']}; background:{colors['button']};"
            f"border:1px solid {colors['border']}; border-radius:3px;"
            "padding:3px 5px; font-size:9px;"
            "}"
            "QToolButton:hover {"
            f"background:{colors['hover']}; border-color:{colors['highlight']};"
            "}"
            "QToolButton:checked {"
            f"background:{colors['highlight']}; border-color:{colors['highlight']};"
            "}"
        )
        self.expand_button.setStyleSheet(
            "QToolButton { background:transparent; border:0; padding:0; font-size:16px; }"
            f"QToolButton:hover {{ background:{colors['hover']}; border-radius:3px; }}"
        )
        if self._active_color_picker is not None:
            self._active_color_picker.apply_theme(active)
        self.view.viewport().update()

    def set_branches(
        self,
        summaries: Iterable[dict[str, Any]],
        colors: dict[int, str] | None = None,
        parameter_fields: Iterable[dict[str, str]] = (),
    ) -> None:
        palette = colors or {}
        rows = []
        for summary in summaries:
            row = dict(summary)
            branch_id = int(row["branch_id"])
            row["color"] = str(palette.get(branch_id, "#4C9FD6"))
            rows.append(row)
        selection_model = self.view.selectionModel()
        view_blocker = self.view.blockSignals(True)
        selection_blocker = selection_model.blockSignals(True)
        try:
            selection_model.clearSelection()
            self.view.setCurrentIndex(QModelIndex())
            self.model.set_rows(rows)
        finally:
            selection_model.blockSignals(selection_blocker)
            self.view.blockSignals(view_blocker)
        self._selected_ids = ()
        self._parameter_fields = [dict(item) for item in parameter_fields]
        # Replacing the catalog is a new branch context. Reset only the local
        # action state without emitting user interaction signals downstream.
        self._footprints_enabled = False
        self._focus_3d_enabled = False
        for button in (self.footprints_button, self.focus_3d_button):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        self.count_label.setText(str(len(rows)))
        self.expand_button.setToolTip(f"Развернуть менеджер ветвей ({len(rows)})")
        self._sync_parameter_menu()
        self._sync_enabled_actions()
        self.view.viewport().update()

    def is_branch_color_available(self, branch_id: int, color: str) -> bool:
        """Return whether a valid color is unused by every other branch."""
        parsed = QColor(str(color))
        if not parsed.isValid():
            return False
        candidate = parsed.name(QColor.NameFormat.HexRgb).upper()
        return all(
            int(row["branch_id"]) == int(branch_id)
            or QColor(str(row.get("color", ""))).name(QColor.NameFormat.HexRgb).upper() != candidate
            for row in self.model._rows
        )

    def set_branch_color(self, branch_id: int, color: str) -> bool:
        if not self.is_branch_color_available(branch_id, color):
            return False
        parsed = QColor(str(color))
        self.model.update_color(
            int(branch_id), parsed.name(QColor.NameFormat.HexRgb).upper()
        )
        return True

    def branch_colors(self) -> dict[int, str]:
        return {
            int(row["branch_id"]): str(row["color"])
            for row in self.model._rows
        }

    def clear_selection(self) -> None:
        self.view.selectionModel().clearSelection()
        self.view.setCurrentIndex(QModelIndex())

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        width = self.EXPANDED_WIDTH if self._expanded else self.COMPACT_WIDTH
        self.setFixedWidth(width)
        self.title_label.setVisible(self._expanded)
        self.count_label.setVisible(self._expanded)
        self.actions_widget.setVisible(self._expanded)
        self.view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if self._expanded else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.expand_button.setText("‹" if self._expanded else "›")
        self.expand_button.setToolTip(
            "Свернуть менеджер ветвей" if self._expanded else
            f"Развернуть менеджер ветвей ({self.model.rowCount()})"
        )
        self.model.set_compact(not self._expanded)
        self.expanded_changed.emit(self._expanded)

    def _selected_branch_from_indexes(self) -> tuple[int, ...]:
        values = []
        for index in self.view.selectionModel().selectedIndexes():
            value = index.data(_BranchListModel.BranchIdRole)
            if value is not None:
                values.append(int(value))
        return tuple(sorted(set(values)))

    def _on_selection_changed(self, *_args) -> None:
        self._selected_ids = self._selected_branch_from_indexes()
        self._sync_enabled_actions()
        self.selection_changed.emit(self._selected_ids)
        # A checked display action follows the current multi-selection; it must
        # never remain bound to the branch IDs selected before the last click.
        if self._footprints_enabled:
            if self._selected_ids:
                self.footprints_toggled.emit(True)
            else:
                self._set_footprints(False)
        if self._focus_3d_enabled:
            if self._selected_ids:
                self.focus_3d_toggled.emit(True)
            else:
                self._set_focus_3d(False)
        self.view.viewport().update()

    def _sync_enabled_actions(self) -> None:
        count = len(self._selected_ids)
        self.footprints_button.setEnabled(count >= 1)
        self.focus_3d_button.setEnabled(count >= 1)
        self.parameter_button.setEnabled(count == 1 and bool(self._parameter_fields))
        self.clear_button.setEnabled(count > 0)
        if count == 0:
            self.parameter_button.setToolTip("Выберите одну ветвь, чтобы построить карту параметра")
        elif count > 1:
            self.parameter_button.setToolTip("Карта параметра доступна только для одной ветви")
        else:
            self.parameter_button.setToolTip("Построить карту выбранного параметра одной ветви")

    def _sync_parameter_menu(self) -> None:
        self.parameter_menu.clear()
        for field in self._parameter_fields:
            key = str(field.get("key", ""))
            label = str(field.get("label", key))
            unit = str(field.get("unit", ""))
            text = f"{label} [{unit}]" if unit else label
            action = self.parameter_menu.addAction(text)
            action.setData(key)
            action.triggered.connect(
                lambda _checked=False, parameter=key: self._request_parameter_map(parameter)
            )
        self.parameter_button.setEnabled(
            len(self._selected_ids) == 1 and bool(self._parameter_fields)
        )

    def _request_parameter_map(self, parameter: str) -> None:
        if len(self._selected_ids) != 1:
            return
        self.parameter_map_requested.emit(int(self._selected_ids[0]), str(parameter))

    def _set_footprints(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._footprints_enabled = enabled
        if self.footprints_button.isChecked() != enabled:
            self.footprints_button.blockSignals(True)
            self.footprints_button.setChecked(enabled)
            self.footprints_button.blockSignals(False)
        if enabled and not self._selected_ids:
            self._footprints_enabled = False
            self.footprints_button.blockSignals(True)
            self.footprints_button.setChecked(False)
            self.footprints_button.blockSignals(False)
            return
        self.footprints_toggled.emit(enabled)

    def _set_focus_3d(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._focus_3d_enabled = enabled
        if self.focus_3d_button.isChecked() != enabled:
            self.focus_3d_button.blockSignals(True)
            self.focus_3d_button.setChecked(enabled)
            self.focus_3d_button.blockSignals(False)
        if enabled and not self._selected_ids:
            self._focus_3d_enabled = False
            self.focus_3d_button.blockSignals(True)
            self.focus_3d_button.setChecked(False)
            self.focus_3d_button.blockSignals(False)
            return
        self.focus_3d_toggled.emit(enabled)

    def _show_context_menu(self, position: QPoint) -> None:
        index = self.view.indexAt(position)
        if not index.isValid():
            return
        branch_id = int(index.data(_BranchListModel.BranchIdRole))
        selected = self._selected_branch_from_indexes()
        if branch_id not in selected:
            self.view.selectionModel().clearSelection()
            self.view.selectionModel().select(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
            selected = (branch_id,)
        summary = index.data(_BranchListModel.SummaryRole) or {}
        menu = QMenu(self)
        apply_popup_theme(menu, resolve_theme(self))
        footprints_action = menu.addAction("Показать XY-footprints на 2D карте")
        footprints_action.setCheckable(True)
        footprints_action.setChecked(self._footprints_enabled)
        focus_action = menu.addAction("Фокус на маркерах и рёбрах выбранных ветвей в 3D")
        focus_action.setCheckable(True)
        focus_action.setChecked(self._focus_3d_enabled)
        parameter_menu = menu.addMenu("Карта параметра выбранной ветви")
        parameter_menu.setEnabled(len(selected) == 1 and bool(self._parameter_fields))
        parameter_actions = {}
        for field in self._parameter_fields:
            key = str(field.get("key", ""))
            label = str(field.get("label", key))
            unit = str(field.get("unit", ""))
            action = parameter_menu.addAction(f"{label} [{unit}]" if unit else label)
            parameter_actions[action] = key
        menu.addSeparator()
        color_action = menu.addAction(
            f"Изменить цвет ветви {branch_id}…" if len(selected) == 1
            else f"Изменить цвет ветви {branch_id}…"
        )
        clear_action = menu.addAction("Снять выделение")
        action = menu.exec(self.view.viewport().mapToGlobal(position))
        if action == footprints_action:
            self._set_footprints(footprints_action.isChecked())
        elif action == focus_action:
            self._set_focus_3d(focus_action.isChecked())
        elif action in parameter_actions:
            self._request_parameter_map(parameter_actions[action])
        elif action == color_action:
            self._choose_color(branch_id, index)
        elif action == clear_action:
            self.clear_selection()

    def _choose_color(self, branch_id: int, index: QModelIndex | None = None) -> None:
        color = index.data(_BranchListModel.ColorRole) if index is not None else None
        if not color:
            summary = next(
                (row for row in self.model._rows if int(row["branch_id"]) == int(branch_id)),
                None,
            )
            color = summary.get("color", "#4C9FD6") if summary else "#4C9FD6"
        unavailable_colors = [
            str(row.get("color", ""))
            for row in self.model._rows
            if int(row["branch_id"]) != int(branch_id)
        ]
        picker = CompactColorPicker(
            str(color), self, unavailable_colors=unavailable_colors,
        )
        self._active_color_picker = picker
        picker.destroyed.connect(lambda *_args, p=picker: self._forget_color_picker(p))
        picker.color_selected.connect(
            lambda value, rid=int(branch_id): self.color_changed.emit(rid, str(value))
        )
        rect = self.view.visualRect(index) if index is not None and index.isValid() else self.expand_button.rect()
        anchor = self.view.viewport() if index is not None and index.isValid() else self.expand_button
        position = anchor.mapToGlobal(QPoint(rect.right() + 4, rect.top()))
        picker.adjustSize()
        screen = anchor.screen() or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            x = min(max(area.left() + 4, position.x()), area.right() - picker.width() + 1)
            y = min(max(area.top() + 4, position.y()), area.bottom() - picker.height() + 1)
            picker.move(x, y)
        else:
            picker.move(position)
        picker.show()

    def _forget_color_picker(self, picker) -> None:
        if self._active_color_picker is picker:
            self._active_color_picker = None
