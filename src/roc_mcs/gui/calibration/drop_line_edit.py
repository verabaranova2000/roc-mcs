from PyQt6.QtWidgets import QLineEdit, QToolTip, QToolButton
from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QFontMetrics

class DropLineEdit(QLineEdit):
    """
    Поле для ввода пути к MCS-файлу.

    Поддерживает:
    - перетаскивание .mcs-файла из проводника;
    - встроенную кнопку выбора файла;
    - сигнал browse_requested при нажатии кнопки;
    - подсказку с полным путем, если он не помещается в поле.
    """
    browse_requested = pyqtSignal()    # Сигнал для передачи клика по кнопке "..." в главное окно

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Путь к .mcs файлу (перетащите файл сюда)...")
        self.setMinimumHeight(26)   

        self.btn_width = 20                               # Ширина встроенной кнопки (20px идеально совпадает с шириной стрелки QComboBox в Windows)
        self.setTextMargins(0, 0, self.btn_width + 4, 0)  # Резервируем пространство справа, чтобы длинный путь файла не прятался под кнопкой

        # --- Кнопка внутри QLineEdit ---
        self.btn_browse = QToolButton(self)
        self.btn_browse.setText("...")
        self.btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse.setToolTip("Выбрать файл через проводник")
        
        # Стилизуем под родную кнопку комбобокса: серый блок с аккуратной границей слева
        self.btn_browse.setStyleSheet("""
            QToolButton {
                background-color: #e5e5e5;
                border: none;
                border-left: 1px solid #c0c0c0;
                color: #333;
                font-weight: bold;
                padding-bottom: 2px;
            }
            QToolButton:hover {
                background-color: #d4d4d4;
            }
            QToolButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        self.btn_browse.clicked.connect(self.browse_requested.emit)

        self.textChanged.connect(self._update_tooltip)
        self._update_tooltip(self.text())

    # ===================================
    # Кнопка "..." - открыть проводник
    # ===================================    
    def resizeEvent(self, event):
        """Размещает (и удерживает динамически) встроенную кнопку у правого края поля."""
        super().resizeEvent(event)
        btn_height = self.height() - 2                               # Высота кнопки равна высоте строки минус 2 пикселя (на верхнюю и нижнюю границу самой строки)
        self.btn_browse.resize(self.btn_width, btn_height)
        self.btn_browse.move(self.width() - self.btn_width - 1, 1)   # Сдвигаем в самый правый край (отступ 1 пиксель сверху и справа для рамки)

    # ===================================
    # Всплывающая подсказка
    # ===================================
    def _is_text_truncated(self) -> bool:
        """Проверяет, помещается ли полный путь в поле."""
        text = self.displayText()
        if not text:
            return False
        fm = QFontMetrics(self.font())
        text_width = fm.horizontalAdvance(text)
        avalable_width = self.contentsRect().width() - self.btn_width - 8
        return text_width > avalable_width 

    def _update_tooltip(self, text: str):
        """Обновляет текст всплывающей подсказки."""
        self.setToolTip(text)

    def event(self, event):
        """Показывает полный путь при наведении на обрезанный текст."""
        if event.type() == QEvent.Type.ToolTip:
            if self.text() and self._is_text_truncated():
                QToolTip.showText(event.globalPos(), self.text(), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    # ===================================
    # Перетаскивание файлов 
    # ===================================
    def dragEnterEvent(self, event):
        """Разрешает перетаскивание файлов в поле и меняет его вид."""
        if event.mimeData().hasUrls():
            self.setStyleSheet("border: 1px solid #2e7d32; background-color: #e8f5e9;")
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        """Возвращает обычный вид поля после выхода файла."""
        self.setStyleSheet("")  
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        """Принимает перетащенный .mcs-файл и записывает его путь в поле."""
        self.setStyleSheet("")  
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                file_path_lower = file_path.lower()
                if file_path_lower.endswith('.mcs'): 
                    self.setText(file_path)
                    event.acceptProposedAction()
                    return
        else:
            super().dropEvent(event)