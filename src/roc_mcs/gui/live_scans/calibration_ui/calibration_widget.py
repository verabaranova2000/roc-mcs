# calibration_widget.py

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, 
    QProgressBar, QSplitter, 
    QGroupBox, QComboBox,
    QFormLayout,
    QToolButton,
    QToolBar,
    QFileDialog
)
from PyQt6.QtGui import QAction
from roc_mcs.gui.components.icon_provider import resource_svg_icon
from PyQt6.QtCore import Qt, QSize

from roc_mcs.io.sardana_h5 import H51DEntryProvider
from .calibration_worker import CalibrationWorker
from .drop_line_edit import DropLineEdit
from .entry_combo_box import EntryComboBox
from .plot_box import PlotBox
from .h5_settings_dialog import H5SettingsDialog
from roc_mcs.io.mcs import load_mcs
from roc_mcs.processing.alignment import find_phase
from roc_mcs.processing.alignment import extract_branch


class CalibrationWidget(QWidget):
    """
    Виджет калибровки. 
    Его можно встроить в любое приложение PyQt6 через layout.addWidget()
    """
    # --- Геометрия основного интерфейса ---
    LEFT_PANEL_MIN_WIDTH = 260       # Минимальная ширина левой панели
    LEFT_PANEL_MAX_WIDTH = 280       # Максимальная ширина левой панели

    RIGHT_PANEL_GAP = 6              # Вертикальный зазор между рядами графиков
    PREVIEW_ROW_GAP = 6              # Горизонтальный зазор между верхними графиками

    STATUS_BAR_HEIGHT = 30           # Высота нижней статусной панели

    # --- Стили статус-бара ---
    STYLE_NORMAL = "color: #aab8ca; border: none; font-size: 11px;"
    STYLE_READY = "color: #8fc5ff; border: none; font-size: 11px; font-weight: bold;"

    def __init__(self, entry_provider=None, parent=None):
        super().__init__(parent)
        self.entry_provider = entry_provider                              # Ссылка на словарь/ТРС_data.h5
        if self.entry_provider and getattr(self.entry_provider, "file_path", None):
            self.entry_h5_path = Path(self.entry_provider.file_path)      # извлекаем путь из провайдера, если он там есть
        else:
            self.entry_h5_path = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)       # Убираем лишние отступы по краям
        main_layout.setSpacing(0)  # Убираем зазор между рабочим полем и подвалом
        self.splitter = QSplitter(Qt.Orientation.Horizontal)  # Разделитель (Splitter) позволяет мышкой двигать границу между панелью ввода и графиком
        
        main_layout.addWidget(self.splitter)

        # --- ЛЕВАЯ ПАНЕЛЬ: Управление и Консоль ---
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(self.LEFT_PANEL_MIN_WIDTH)
        self.left_panel.setMaximumWidth(self.LEFT_PANEL_MAX_WIDTH)      # Ограничение ширины
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)


        # Блок ввода данных
        self.input_group = QGroupBox()
        input_outer = QVBoxLayout(self.input_group)
        input_outer.setContentsMargins(6, 6, 6, 6)
        input_outer.setSpacing(4)
        
        # Верхняя узкая панель действий
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QSize(14, 14))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setStyleSheet("""
            QToolBar {
                border: none;
                padding: 0px;
                margin: 0px;
                spacing: 2px;
                background: transparent;
            }
        """)
        
        self.act_run = QAction(resource_svg_icon("run_icon.svg", widget=self, size=16),
                               "Рассчитать калибровку",
                               self)
        self.act_run.triggered.connect(self.start_calibration)
        
        self.act_stop = QAction(resource_svg_icon("stop_icon.svg", widget=self, size=16),
                                "Остановить расчёт (не реализовано)",
                                self)
        self.act_stop.setEnabled(False)
        self.act_stop.setToolTip("Остановка длительного расчёта пока не реализована")
        
        self.act_info = QAction(resource_svg_icon("info_icon.svg", widget=self, size=16),
                                # self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
                                "Справка",
                                self)
        self.act_info.triggered.connect(lambda: self.log("[СПРАВКА] Выберите эталонный моторный скан в HDF5 и соответствующий MCS-файл одного физического объекта. Расчёт ищет θ = A·s + B."))

        self.act_settings = QAction(resource_svg_icon("setting_icon.svg", widget=self, size=16), 
                                    "Настроить HDF5", 
                                    self)
        self.act_settings.triggered.connect(self.open_h5_settings)

        toolbar.addAction(self.act_run)
        toolbar.addAction(self.act_stop)
        toolbar.addAction(self.act_info)
        toolbar.addAction(self.act_settings)
        
        input_outer.addWidget(toolbar)

        # # --- Состояние ---
        # self.entry_h5_path = None
        
        # Правая часть — поля
        self.fields_widget = QWidget()
        self.input_form = QFormLayout(self.fields_widget)
        self.input_form.setContentsMargins(0, 0, 0, 0)
        self.input_form.setHorizontalSpacing(6)
        self.input_form.setVerticalSpacing(6)
        self.input_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Кнопки предпросмотра наследуют тему основного приложения.

        # --- 1. Поле entry скана ---
        self.combo_entry_id = EntryComboBox()
        if self.entry_provider:
            ids = self.entry_provider.list_ids()
            for entry_id in ids:
                self.combo_entry_id.addItem(str(entry_id), entry_id)
        self.combo_entry_id.setCurrentIndex(-1)

        
        self.btn_preview_entry = QToolButton()
        icon = resource_svg_icon("gaussian_icon.svg", widget=self, size=16)        # 📈
        self.btn_preview_entry.setIcon(icon)
        self.btn_preview_entry.setIconSize(QSize(18, 18))
        self.btn_preview_entry.setToolTip("Предпросмотр моторного эталона")
        self.btn_preview_entry.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_entry.setObjectName("iconPreviewButton")
        self.btn_preview_entry.clicked.connect(lambda: self.preview_data(source="entry"))  # поменяли подключение кнопок
        

        entry_layout = QHBoxLayout()
        entry_layout.setContentsMargins(0, 0, 0, 0)
        entry_layout.setSpacing(5)
        entry_layout.addWidget(self.combo_entry_id)
        entry_layout.addWidget(self.btn_preview_entry)
        
        self.input_form.addRow("Эталонный скан:", entry_layout)


        # --- 2. Поле MCS файла ---
        self.input_mcs_path = DropLineEdit()
        # Просто подключаем сигнал нашей новой строгой кнопки к функции открытия проводника
        self.input_mcs_path.browse_requested.connect(self.browse_mcs_file)
        
        self.btn_preview_mcs = QToolButton()
        icon = resource_svg_icon("gaussian_icon.svg", widget=self, size=16)        # 📈
        self.btn_preview_mcs.setIcon(icon)
        self.btn_preview_mcs.setIconSize(QSize(18, 18))
        self.btn_preview_mcs.setToolTip("Предпросмотр MCS-профиля")
        self.btn_preview_mcs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_mcs.setObjectName("iconPreviewButton")
        self.btn_preview_mcs.clicked.connect(lambda: self.preview_data(source="mcs"))
        
        mcs_layout = QHBoxLayout()
        mcs_layout.setContentsMargins(0, 0, 0, 0)
        mcs_layout.setSpacing(5)
        mcs_layout.addWidget(self.input_mcs_path)
        
        mcs_layout.addWidget(self.btn_preview_mcs)
        self.input_form.addRow("Файл MCS:", mcs_layout)

        # --- 3. Выбор ветви ---
        self.combo_branch = QComboBox()
        self.combo_branch.addItem("Нижняя ветвь (down)", "down")
        self.combo_branch.addItem("Верхняя ветвь (up)", "up")
        self.combo_branch.setMinimumWidth(150)                                                  # Делаем комбобокс компактным
        self.combo_branch.currentTextChanged.connect(lambda: self.preview_data(silent=True))  # При изменении ветви тоже логично обновлять предпросмотр (опционально)
        self.input_form.addRow("Ветвь скана:", self.combo_branch)
        
        input_outer.addWidget(self.fields_widget)
        left_layout.addWidget(self.input_group)


        # Консоль
        log_group = QGroupBox("Ход расчёта")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(2, 5, 2, 2) # Уменьшаем отступы внутри рамки лога
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color:#111720; color:#c4d0df; border:1px solid #303b4a; font-family:Consolas,monospace; font-size:9pt;")
        log_layout.addWidget(self.log_console)
        
        left_layout.addWidget(log_group)
        self.splitter.addWidget(self.left_panel)

        
        # --- ПРАВАЯ ПАНЕЛЬ: Графики ---       
        self.right_panel = QWidget()
        self.layout_inputs = QVBoxLayout(self.right_panel)
        
        self.layout_inputs.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.layout_inputs.setContentsMargins(0, 0, 0, 0)
        self.layout_inputs.setSpacing(self.RIGHT_PANEL_GAP)
        
        # --- Верхний ряд: исходные данные (preview) ---
        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(self.PREVIEW_ROW_GAP)
        
        self.preview_entry = PlotBox("Эталонный моторный скан")
        self.preview_mcs = PlotBox("Скан многоканальника")

        preview_row.addWidget(self.preview_entry, 1)
        preview_row.addWidget(self.preview_mcs, 1)
        
        self.layout_inputs.addLayout(preview_row)
        
        # --- Нижний график: результат калибровки ---
        self.preview_result = PlotBox("Результат калибровки")  # "Результат калибровки"
        self.layout_inputs.addWidget(self.preview_result)

        # --- Добавляем правую панель в splitter ---
        self.splitter.addWidget(self.right_panel)
    
        # =================================
        
        # Настраиваем поведение сплиттера: правая часть растягивается активнее
        self.splitter.setSizes([280, 700])
        self.splitter.setStretchFactor(0, 0) # Левая панель не тянется
        self.splitter.setStretchFactor(1, 1) # Правая тянется по максимуму


        # --- STATUS BAR ---
        self.status_container = QWidget()
        self.status_container.setFixedHeight(self.STATUS_BAR_HEIGHT)  # Фиксируем высоту подвала
        self.status_container.setStyleSheet("background-color: #141b26; border-top: 1px solid #303b4a;")
        
        status_layout = QHBoxLayout(self.status_container)
        status_layout.setContentsMargins(10, 0, 10, 0)
        status_layout.setSpacing(10)
        
        # Текстовый статус текущей операции (занимает всё свободное место слева)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #cbd5e1; border: none; font-size: 11px;")
        status_layout.addWidget(self.lbl_status, stretch=1)
        
        # Текст с цифрами прогресса (строго фиксированная ширина, чтобы не скакало!)
        self.lbl_progress_text = QLabel("")
        self.lbl_progress_text.setFixedWidth(130)
        self.lbl_progress_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_progress_text.setStyleSheet("color: #aab8ca; border: none; font-family: Consolas, Monospace; font-size: 11px;")
        status_layout.addWidget(self.lbl_progress_text)
        
        # Компактный академичный прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(150, 10)  # Короткий и тонкий, прижат вправо
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3b4a5f;
                border-radius: 2px;
                background-color: #202a38;
            }
            QProgressBar::chunk {
                background-color: #448bd4; /* Строгий синий цвет, типичный для IDE и научных программ */
                border-radius: 1px;
            }
        """)
        
        status_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_container)

        # --- Динамическая валидация готовности к калибровке ---
        self.input_mcs_path.textChanged.connect(self.update_ready_state)
        self.combo_entry_id.currentIndexChanged.connect(self.update_ready_state)
        self.combo_entry_id.currentTextChanged.connect(self.update_ready_state)

        # Первичный расчет состояния при загрузке окна
        self.update_ready_state()
        
    def log(self, text):
        """Вывод сообщений в консоль GUI"""
        # self.log_console.append(text)
        # Используем <pre> для гарантированного сохранения всех пробелов и отступов
        formatted_msg = f'<pre style="margin: 0; line-height: 1.2;">{text}</pre>'
        self.log_console.append(formatted_msg)


    @property
    def current_entry_id(self):
        """
        Геттер для Entry ID. 
        Гарантирует, что введенный текст является валидным числовым идентификатором.
        """
        text = self.combo_entry_id.currentText().strip()
        if not text:
            return None
        
        # Если в комбобоксе выбран реальный элемент из выпадающего списка
        index = self.combo_entry_id.findText(text)
        if index != -1 and text == self.combo_entry_id.itemText(index):
            return self.combo_entry_id.itemData(index)
        
        # Если пользователь вводит текст руками: проверяем, что это строго целое число
        if text.isdigit():
            return int(text)
            
        # Если ввели буквы ("ь", "abc" и т.д.) — считаем, что валидного ID нет
        return None


    # ===============================
    # Статус готовности к калибровке
    # ===============================    
    def update_ready_state(self):
        """
        Проверяет полноту введенных данных, обновляет текст строки статуса
        и блокирует/разблокирует кнопку выполнения калибровки.
        """
        # 1. Проверяем наличие пути к MCS-файлу
        mcs_path = self.input_mcs_path.text().strip()
        has_mcs = bool(mcs_path)

        # 2. Проверяем наличие источника данных (загружен ли HDF5 или словарь)
        has_provider = self.entry_provider is not None

        # 3. Проверяем выбор Entry ID (выбран ли пункт в combo или введен текст)
        has_entry = self.current_entry_id is not None

        # Динамическая реакция интерфейса
        if has_mcs and has_provider and has_entry:            # < --- Все данные собраны
            self.lbl_status.setText("Готов к калибровке")
            self.lbl_status.setStyleSheet(self.STYLE_READY)
            self.act_run.setEnabled(True)
        elif not has_mcs and not has_provider:                # < --- Нехватка: Нет ни MCS, ни HDF5
            self.lbl_status.setText("Ожидание данных: укажите MCS-файл и путь к HDF5")
            self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
            self.act_run.setEnabled(False)
        elif not has_provider:                                # < --- Нехватка: MCS загружен, но провайдер HDF5 пуст
            self.lbl_status.setText("Ожидание данных: укажите путь к HDF5")
            self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
            self.act_run.setEnabled(False)
        elif not has_mcs and not has_entry:                   # < --- Нехватка: База HDF5 есть, но ничего не выбрано и нет MCS
            self.lbl_status.setText("Ожидание данных: укажите MCS-файл и выберите Entry ID")
            self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
            self.act_run.setEnabled(False)
        elif not has_mcs:                                     # < --- Нехватка: нет только MCS; база есть, Entry выбран
            self.lbl_status.setText("Ожидание данных: укажите MCS-файл")
            self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
            self.act_run.setEnabled(False)
        else:                                                 # < --- Нехватка: нет только ENTRY ID; все файлы на месте, осталось выбрать скан
            self.lbl_status.setText("Ожидание данных: выберите Entry ID")
            self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
            self.act_run.setEnabled(False)


    # ===========================
    # Метод настройки: путь к data.h5
    # ===========================
    def open_h5_settings(self):
        current_path = self.entry_h5_path or ""

        dialog = H5SettingsDialog(current_path=current_path, parent=self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        new_path = Path(dialog.get_path())
        self.entry_h5_path = new_path
        self.entry_provider = H51DEntryProvider(new_path)

        self.reload_entry_ids()
        self.update_ready_state()    # обновить статус интерфейса
        self.log(f"[ИНФО] Путь HDF5 обновлён: {new_path}")

    def set_h5_path(self, path):
        """Загружает HDF5 программно и обновляет список эталонных сканов."""
        new_path = Path(path).expanduser()
        if not new_path.is_file(): raise FileNotFoundError(f"Файл HDF5 не найден: {new_path}")
        provider = H51DEntryProvider(new_path)
        self.entry_h5_path = new_path
        self.entry_provider = provider
        self.reload_entry_ids()
        self.update_ready_state()

    def reload_entry_ids(self):
        """
        Метод обновления combo box
        Чтобы после смены файла список entry обновлялся:
        """
        self.combo_entry_id.blockSignals(True)
        self.combo_entry_id.clear()

        if self.entry_provider:
            for entry_id in self.entry_provider.list_ids():
                self.combo_entry_id.addItem(str(entry_id), entry_id)

        self.combo_entry_id.setCurrentIndex(-1)
        self.combo_entry_id.blockSignals(False)


    # ===========================
    # Метод вызова проводника
    # ===========================
    def browse_mcs_file(self):
        """Открывает диалог выбора файла и записывает путь в строку"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите MCS файл", 
            "", 
            "Файлы MCS (*.mcs);;Все файлы (*.*)"
        )
        if file_path:
            self.input_mcs_path.setText(file_path)

        
    # ===========================
    # Прогресс-бар
    # ===========================
    def update_progress(self, status_msg, current, total):
        self.lbl_status.setText(status_msg)      # Обновляем текст слева
        self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
        if total > 0 and current > 0:            # Обновляем цифры справа (например, "201 / 201")
            self.lbl_progress_text.setText(f"{current} / {total}")
            pct = int((current / total) * 100)   # Рассчитываем проценты для самого бара
            self.progress_bar.setValue(pct)
        else:                                    # Если передали (..., 0, 100) или что-то без точных шагов
            self.lbl_progress_text.setText("")
            self.progress_bar.setValue(current) 

        
    # ===========================
    # Графики (исходные данные)
    # ===========================
    def preview_data(self, source="both", silent=False):
        mcs_path = self.input_mcs_path.text().strip()
        # entry_id = self.combo_entry_id.currentData()
        entry_id = self.current_entry_id
        branch = self.combo_branch.currentData()
    
        if source in ("mcs", "both") and mcs_path:
            try:
                mcs = load_mcs(mcs_path)
                counts, n_channels = mcs["counts"], mcs["n_channels"]
                phi, _ = find_phase(counts, n_channels)
                s_mca, y_mca = extract_branch(counts, phi, n_channels, branch=branch)
                self.preview_mcs.update_plot(s_mca, y_mca, r"Параметр развёртки, $s=\sin\varphi$", "Интенсивность")
            except Exception as e:
                self.log(f"[ОШИБКА ПРЕДПРОСМОТРА MCS]: {e}")
    
        elif source == "mcs" and not mcs_path and not silent:
            self.log("[ИНФО] Укажите файл MCS для предпросмотра.")
    
        if source in ("entry", "both") and entry_id:
            try:
                curve_entry = self.entry_provider.load(entry_id)
                theta_entry, y_motor = curve_entry["theta"], curve_entry["intensity"]
                self.preview_entry.update_plot(theta_entry, y_motor, r"Угол, $\theta$, угл. с", "Интенсивность")
            except Exception as e:
                self.log(f"[ОШИБКА ПРЕДПРОСМОТРА ENTRY]: {e}")
    
        elif source == "entry" and not entry_id and not silent:
            self.log("[ИНФО] Укажите entry для предпросмотра.")
            

    def start_calibration(self):
        mcs_path = self.input_mcs_path.text().strip()
        # entry_id = self.combo_entry_id.currentData()
        entry_id = self.current_entry_id
        branch = self.combo_branch.currentData()

        if not mcs_path or entry_id is None:
            self.log("[ОШИБКА] Укажите путь к MCS файлу и ID entry!")
            return

        self.preview_result.clear_plot()   # Очистить предыдущий результат
        self.act_run.setEnabled(False)
        # --- ИЗМЕНЕНИЕ 1: Сброс нового статус-бара перед запуском ---
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Чтение файлов...")
        self.lbl_status.setStyleSheet(self.STYLE_NORMAL)
        self.lbl_progress_text.setText("")
        
        self.log_console.clear()

        # --- 1. ПРЕДВАРИТЕЛЬНЫЙ ПАРСИНГ И ОТРИСОВКА В GUI ---
        try:
            self.log("Чтение входных данных...")
            mcs = load_mcs(mcs_path)
            counts, n_channels = mcs["counts"], mcs["n_channels"]
            phi, _ = find_phase(counts, n_channels)
            self._cache_phi_mcs = float(phi)  # Фаза только калибровочного MCS; live-серия определяет свою φ заново.
            s_mca, y_mca = extract_branch(counts, phi, n_channels, branch=branch)

            curve_entry = self.entry_provider.load(entry_id)
            theta_entry, y_motor = curve_entry["theta"], curve_entry["intensity"]
            
            # Рисуем и переключаемся на первую вкладку
            self.preview_mcs.update_plot(s_mca, y_mca, r"Параметр развёртки, $s=\sin\varphi$", "Интенсивность")
            self.preview_entry.update_plot(theta_entry, y_motor, r"Угол, $\theta$, угл. с", "Интенсивность")
            
            #  --- Кэшируем исходные массивы в памяти виджета ---
            self._cache_s_mca = s_mca
            self._cache_y_mca = y_mca
            self._cache_theta_entry = theta_entry
            self._cache_y_motor = y_motor
        
        except Exception as e:
            self.log(f"[КРИТИЧЕСКАЯ ОШИБКА ЧТЕНИЯ]: {e}")
            self.lbl_status.setText("Ошибка чтения файлов") # Информируем в статус-баре
            self.act_run.setEnabled(True)
            return

        
        self.log(">>> Инициализация расчета...")
        self.lbl_status.setText("Подготовка расчёта...")

        # Сохраняем контекст именно этого расчёта: поля нельзя перепутать после завершения потока.
        self._calibration_context = {"source_mcs": mcs_path, "reference_hdf5": str(self.entry_h5_path or ""),
                                     "entry_id": entry_id, "branch": branch, "reference_phi": float(self._cache_phi_mcs)}
        # Контекст серии дополняется подклассом до запуска фонового потока.
        self._calibration_context.update(getattr(self, "_extra_calibration_context", {}))
        # Формируем kwargs для функции калибровки
        kwargs = {
            'file_path_mcs': mcs_path,
            'entry_reference_id': entry_id,
            'branch_mcs': self.combo_branch.currentData(),
            'entry_provider': self.entry_provider,   # Источник entry (ЕРС_data.h5)
        }

        # Запуск фонового потока
        self.worker = CalibrationWorker(kwargs)
        self.worker.log_signal.connect(self.log)

        self.worker.progress_signal.connect(self.update_progress)     # Направляем сигнал в новый метод update_progress
        self.worker.finished_signal.connect(self.on_calibration_finished)
        self.worker.error_signal.connect(self.on_calibration_error)
        self.worker.start()

    def on_calibration_finished(self, result, fig):
        self.act_run.setEnabled(True)
        # self.log("\n[УСПЕХ] Расчет завершен успешно!")
        # self.log(f"Параметры калибровки:\nA* = {result.A_best:.6f}\nB* = {result.B_best:.6f}")

        self.lbl_status.setText("Готово")     # Красиво завершаем статус-бар
        self.lbl_status.setStyleSheet(self.STYLE_READY)
        self.lbl_progress_text.setText("")
        self.progress_bar.setValue(100)
        
        # --- Восстановление физической угловой оси (theta = A * s + B) --- 
        theta_reconstructed = result.A_best * self._cache_s_mca + result.B_best
    
        # --- Построение результата калибровки ---
        self.preview_result.update_result_plot(
            self._cache_theta_entry,
            self._cache_y_motor,
            theta_reconstructed,
            self._cache_y_mca,
        )
    

    def on_calibration_error(self, err_msg):
        self.act_run.setEnabled(True)
        self.log(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: {err_msg}")

        self.lbl_status.setText("Ошибка калибровки!")  # Отражаем ошибку в статус-баре ---
        self.lbl_progress_text.setText("")
        self.progress_bar.setValue(0)