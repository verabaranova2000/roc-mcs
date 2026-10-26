from PyQt6.QtWidgets import QComboBox, QCompleter
from PyQt6.QtCore import Qt

class EntryComboBox(QComboBox):
    """
    Умный выпадающий список для entry-сканов.
    - Автоматически прокручивается в конец при открытии (к самым новым сканам).
    - Имеет встроенный умный поиск по вхождению подстроки без мерцаний.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setPlaceholderText("ID entry (напр. 331)...")
        
        # Получаем доступ к встроенному completer'у, который создается 
        # автоматически при self.setEditable(True)
        self.smart_completer = self.completer()
        if self.smart_completer:
            # Обязательно PopupCompletion! InlineCompletion часто вызывает баги и мерцания
            self.smart_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            # Искать совпадения по началу строки
            self.smart_completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
            # Отключаем чувствительность к регистру на всякий случай
            self.smart_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            
            # Тонкая настройка: заставляем popup комплитера вести себя смирно 
            # и не конфликтовать с основным списком комбобокса
            popup = self.smart_completer.popup()
            popup.setUniformItemSizes(True) # Ускоряет отрисовку огромных списков
            popup.setLayoutMode(popup.LayoutMode.Batched)

    def showPopup(self):
        """
        Перехватываем момент клика по стрелочке ▼
        """
        # 1. Вызываем базовый метод, чтобы Qt штатно построил и отобразил список
        super().showPopup()
        
        # 2. Если в списке есть элементы, мгновенно скроллим в самый низ
        if self.count() > 0:
            # view() возвращает внутренний QAbstractItemView (список)
            self.view().scrollToBottom()