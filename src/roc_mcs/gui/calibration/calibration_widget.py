# calibration_widget.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTextEdit, 
    QProgressBar, QSplitter, 
    QGroupBox, QComboBox,
    QFormLayout,
    QToolButton,
    QStyle,
    QToolBar,
    QFileDialog
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QSize


from roc_mcs.gui.calibration.calibration_worker import CalibrationWorker
from roc_mcs.gui.calibration.drop_line_edit import DropLineEdit
from roc_mcs.gui.calibration.entry_combo_box import EntryComboBox
from roc_mcs.gui.calibration.plot_box import PlotBox
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

    def __init__(self, entry_provider=None, parent=None):
        super().__init__(parent)
        self.entry_provider = entry_provider # Ссылка на словарь/ТРС_data.h5
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
        
        self.act_run = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                               "Run calibration",
                               self)
        self.act_run.triggered.connect(self.start_calibration)
        
        self.act_stop = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
                                "Stop",
                                self)
        self.act_stop.triggered.connect(lambda: self.log("[INFO] Stop пока не реализован"))
        
        self.act_info = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
                                "Info",
                                self)
        self.act_info.triggered.connect(lambda: self.log("[INFO] Info пока не реализован"))
        
        toolbar.addAction(self.act_run)
        toolbar.addAction(self.act_stop)
        toolbar.addAction(self.act_info)
        
        input_outer.addWidget(toolbar)
        
        # Правая часть — поля
        self.fields_widget = QWidget()
        self.input_form = QFormLayout(self.fields_widget)
        self.input_form.setContentsMargins(0, 0, 0, 0)
        self.input_form.setHorizontalSpacing(6)
        self.input_form.setVerticalSpacing(6)
        self.input_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        
        # --- 1. Поле MCS файла ---
        self.input_mcs_path = DropLineEdit()
        # Просто подключаем сигнал нашей новой строгой кнопки к функции открытия проводника
        self.input_mcs_path.browse_requested.connect(self.browse_mcs_file)
        
        self.btn_preview_mcs = QToolButton()
        icon = QIcon("gaussian_icon.svg")  # 📈
        self.btn_preview_mcs.setIcon(icon)
        self.btn_preview_mcs.setIconSize(QSize(18, 18))
        self.btn_preview_mcs.setToolTip("Предварительный просмотр спектра")
        self.btn_preview_mcs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_mcs.setStyleSheet("QToolButton { font-size: 14px; border: 1px solid #ccc; border-radius: 4px; padding: 2px 4px; background-color: #f8f9fa; } QToolButton:hover { background-color: #e2e6ea; }")
        self.btn_preview_mcs.clicked.connect(lambda: self.preview_data(source="mcs"))
        
        mcs_layout = QHBoxLayout()
        mcs_layout.setContentsMargins(0, 0, 0, 0)
        mcs_layout.setSpacing(5)
        mcs_layout.addWidget(self.input_mcs_path)
        
        mcs_layout.addWidget(self.btn_preview_mcs)
        self.input_form.addRow("Файл MCS:", mcs_layout)

        # --- 2. Поле entry скана ---
        self.combo_entry_id = EntryComboBox()
        if self.entry_provider:
            ids = self.entry_provider.list_ids()
            for entry_id in ids:
                self.combo_entry_id.addItem(str(entry_id), entry_id)
        self.combo_entry_id.setCurrentIndex(-1)

        
        self.btn_preview_entry = QToolButton()
        icon = QIcon("gaussian_icon.svg")  # 📈
        self.btn_preview_entry.setIcon(icon)
        self.btn_preview_entry.setIconSize(QSize(18, 18))
        self.btn_preview_entry.setToolTip("Предварительный просмотр эталона")
        self.btn_preview_entry.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview_entry.setStyleSheet(self.btn_preview_mcs.styleSheet())
        self.btn_preview_entry.clicked.connect(lambda: self.preview_data(source="entry"))  # поменяли подключение кнопок
        

        entry_layout = QHBoxLayout()
        entry_layout.setContentsMargins(0, 0, 0, 0)
        entry_layout.setSpacing(5)
        entry_layout.addWidget(self.combo_entry_id)
        entry_layout.addWidget(self.btn_preview_entry)
        
        self.input_form.addRow("Entry ID:", entry_layout)

        # --- 3. Выбор ветви ---
        self.combo_branch = QComboBox()
        self.combo_branch.addItems(["down", "up"])
        self.combo_branch.setFixedWidth(100)                                                  # Делаем комбобокс компактным
        self.combo_branch.currentTextChanged.connect(lambda: self.preview_data(silent=True))  # При изменении ветви тоже логично обновлять предпросмотр (опционально)
        self.input_form.addRow("Ветвь:", self.combo_branch)
        
        input_outer.addWidget(self.fields_widget)
        left_layout.addWidget(self.input_group)


        # Консоль
        log_group = QGroupBox("Лог расчета")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(2, 5, 2, 2) # Уменьшаем отступы внутри рамки лога
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas, Monospace; font-size: 9pt;")
        log_layout.addWidget(self.log_console)
        
        left_layout.addWidget(log_group)
        self.splitter.addWidget(self.left_panel)

        
        # --- ПРАВАЯ ПАНЕЛЬ: Графики ---       
        self.right_panel = QWidget()
        self.layout_inputs = QVBoxLayout(self.right_panel)
        
        self.layout_inputs.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.layout_inputs.setContentsMargins(0, 0, 0, 0)
        self.layout_inputs.setSpacing(self.RIGHT_PANEL_GAP)
        
        # --- Верхний ряд: исходные данные ---
        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(self.PREVIEW_ROW_GAP)
        
        self.preview_mcs = PlotBox("MCS preview")
        self.preview_entry = PlotBox("Entry preview")
        
        preview_row.addWidget(self.preview_mcs, 1)
        preview_row.addWidget(self.preview_entry, 1)
        
        self.layout_inputs.addLayout(preview_row)
        
        # --- Нижний график: результат калибровки ---
        self.preview_result = PlotBox("Результат калибровки")
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
        self.status_container.setStyleSheet("background-color: #f8f9fa; border-top: 1px solid #dcdcdc;")
        
        status_layout = QHBoxLayout(self.status_container)
        status_layout.setContentsMargins(10, 0, 10, 0)
        status_layout.setSpacing(10)
        
        # Текстовый статус текущей операции (занимает всё свободное место слева)
        self.lbl_status = QLabel("Готово")
        self.lbl_status.setStyleSheet("color: #333333; border: none; font-size: 11px;")
        status_layout.addWidget(self.lbl_status, stretch=1)
        
        # Текст с цифрами прогресса (строго фиксированная ширина, чтобы не скакало!)
        self.lbl_progress_text = QLabel("")
        self.lbl_progress_text.setFixedWidth(130)
        self.lbl_progress_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_progress_text.setStyleSheet("color: #555555; border: none; font-family: Consolas, Monospace; font-size: 11px;")
        status_layout.addWidget(self.lbl_progress_text)
        
        # Компактный академичный прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(150, 10)  # Короткий и тонкий, прижат вправо
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 2px;
                background-color: #e9ecef;
            }
            QProgressBar::chunk {
                background-color: #007acc; /* Строгий синий цвет, типичный для IDE и научных программ */
                border-radius: 1px;
            }
        """)
        
        status_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_container)


        
    def log(self, text):
        """Вывод сообщений в консоль GUI"""
        self.log_console.append(text)

    # ===========================
    # Метод вызова проводника
    # ===========================
    def browse_mcs_file(self):
        """Открывает диалог выбора файла и записывает путь в строку"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите MCS файл", 
            "", 
            "MCS files (*.mcs);;All files (*.*)"
        )
        if file_path:
            self.input_mcs_path.setText(file_path)

        
    # ===========================
    # Прогресс-бар
    # ===========================
    def update_progress(self, status_msg, current, total):
        self.lbl_status.setText(status_msg)      # Обновляем текст слева
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
        entry_id = self.combo_entry_id.currentData()
        branch = self.combo_branch.currentText()
    
        if source in ("mcs", "both") and mcs_path:
            try:
                mcs = load_mcs(mcs_path)
                counts, n_channels = mcs["counts"], mcs["n_channels"]
                phi, _ = find_phase(counts, n_channels)
                s_mca, y_mca = extract_branch(counts, phi, n_channels, branch=branch)
                self.preview_mcs.update_plot(s_mca, y_mca, r"Параметр развертки, $s=\sin\varphi$", "")
            except Exception as e:
                self.log(f"[ОШИБКА ПРЕДПРОСМОТРА MCS]: {e}")
    
        elif source == "mcs" and not mcs_path and not silent:
            self.log("[ИНФО] Укажите файл MCS для предпросмотра.")
    
        if source in ("entry", "both") and entry_id:
            try:
                curve_entry = self.entry_provider.load(entry_id)
                theta_entry, y_motor = curve_entry["theta"], curve_entry["intensity"]
                self.preview_entry.update_plot(theta_entry, y_motor, r"Угол, $\theta$, угл. с", "")
            except Exception as e:
                self.log(f"[ОШИБКА ПРЕДПРОСМОТРА ENTRY]: {e}")
    
        elif source == "entry" and not entry_id and not silent:
            self.log("[ИНФО] Укажите entry для предпросмотра.")
            

    def start_calibration(self):
        mcs_path = self.input_mcs_path.text().strip()
        entry_id = self.combo_entry_id.currentData()
        branch = self.combo_branch.currentText()

        if not mcs_path or entry_id is None:
            self.log("[ОШИБКА] Укажите путь к MCS файлу и ID entry!")
            return

        self.preview_result.clear_plot()   # Очистить предыдущий результат
        self.act_run.setEnabled(False)
        # --- ИЗМЕНЕНИЕ 1: Сброс нового статус-бара перед запуском ---
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Чтение файлов...")
        self.lbl_progress_text.setText("")
        
        self.log_console.clear()

        # --- 1. ПРЕДВАРИТЕЛЬНЫЙ ПАРСИНГ И ОТРИСОВКА В GUI ---
        try:
            self.log(">>> Чтение файлов...")
            mcs = load_mcs(mcs_path)
            counts, n_channels = mcs["counts"], mcs["n_channels"]
            phi, _ = find_phase(counts, n_channels)
            s_mca, y_mca = extract_branch(counts, phi, n_channels, branch=branch)

            curve_entry = self.entry_provider.load(entry_id)
            theta_entry, y_motor = curve_entry["theta"], curve_entry["intensity"]
            
            # Рисуем и переключаемся на первую вкладку
            self.preview_mcs.update_plot(s_mca, y_mca, r"Параметр развертки, $s=\sin\varphi$", "")
            self.preview_entry.update_plot(theta_entry, y_motor, r"Угол, $\theta$, угл. с", "")
            
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
        self.lbl_status.setText("Инициализация расчета...")

        # Формируем kwargs для функции калибровки
        kwargs = {
            'file_path_mcs': mcs_path,
            'entry_reference_id': entry_id,
            'branch_mcs': self.combo_branch.currentText(),
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
        self.log("\n[УСПЕХ] Расчет завершен успешно!")
        self.log(f"Параметры калибровки:\nA* = {result.A_best:.6f}\nB* = {result.B_best:.6f}")

        self.lbl_status.setText("Готово")     # Красиво завершаем статус-бар
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