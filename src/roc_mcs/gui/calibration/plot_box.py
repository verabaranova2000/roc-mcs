from matplotlib.ticker import FormatStrFormatter
from PyQt6.QtWidgets import (
    QFrame,
    QVBoxLayout, QLabel,
    QSizePolicy,
)

from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


from roc_mcs.processing.utils import normalize01


def _style_scientific_axes(ax, title, xlabel, ylabel):
    """  
    меньше размер фигуры;
    меньше шрифты и отступы;
    компактная нумерация оси Y, лучше в научной нотации/с offset.
    """
    ax.set_xlabel(xlabel, fontsize=6, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=6, labelpad=1)

    ax.tick_params(axis='both', which='major', labelsize=6, length=2, width=0.7, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(6)                      # Размер ×10ⁿ над осью Y

    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


class PlotBox(QFrame):
    """
    Контейнер для отображения одного графика.

    Показывает исходные данные или результат калибровки.
    График строится через Matplotlib и отображается внутри PyQt6.
    """
    # --- Геометрия блока графика ---
    BOX_HEIGHT = 260              # Общая высота одного блока графика
    BOX_MIN_WIDTH = 240           # Минимальная ширина одного блока графика
    PLOT_AREA_HEIGHT = 205        # Высота внутренней области Matplotlib
    TITLE_HEIGHT = 16             # Высота заголовка блока
    OUTER_MARGINS = (4, 3, 4, 4)  # Внешние отступы внутри блока: L, T, R, B
    OUTER_SPACING = 1             # Расстояние между заголовком и областью графика

    def __init__(self, title, parent=None):
        super().__init__(parent)

        self.setObjectName("PlotBox")
        self.setFrameShape(QFrame.Shape.Panel)
        self.setFrameShadow(QFrame.Shadow.Sunken)

        # жестко фиксируем общий размер блока
        self.setFixedHeight(self.BOX_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Фиксируем минимальные размеры компонента
        self.setMinimumHeight(self.BOX_HEIGHT)
        self.setMinimumWidth(self.BOX_MIN_WIDTH)

        self.setStyleSheet("""
            QFrame#PlotBox {
                background: #ffffff;
                border: 1px solid #b9b9b9;
                border-radius: 4px;
            }
            QLabel {
                font-weight: 600;
                font-size: 10pt;
                color: #222222;
            }
            QFrame#PlotArea {
                background: #ffffff;
                border: 1px solid #d0d0d0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*self.OUTER_MARGINS)
        layout.setSpacing(self.OUTER_SPACING)

        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-weight: 600; font-size: 10pt; color: #222222; padding: 0px; margin: 0px;")
        self.label.setFixedHeight(16)
        layout.addWidget(self.label)

        # Внутренняя "вдавленная" область под график
        self.plot_area = QFrame()
        self.plot_area.setObjectName("PlotArea")
        self.plot_area.setFixedHeight(self.PLOT_AREA_HEIGHT)
        self.plot_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        plot_layout = QVBoxLayout(self.plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self.figure = plt.Figure(figsize=(4.0, 2.0), dpi=120)
        self.figure.set_facecolor("white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.hide()

        plot_layout.addWidget(self.canvas)
        layout.addWidget(self.plot_area)

        self.ax = self.figure.add_subplot(111)

    def clear_plot(self):
        """Очищает график и скрывает область Matplotlib."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.canvas.hide()

    def update_plot(self, x, y, xlabel, ylabel):
        """Строит обычный график по переданным данным."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        self.ax.plot(x, y, linewidth=1.0, color="#222222")
        _style_scientific_axes(self.ax, self.label.text(), xlabel, ylabel)
        self.figure.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.20)

        self.canvas.show()
        self.canvas.draw_idle()


    def update_result_plot(self, theta_entry, profile_entry, theta_reconstructed, profile_reconstructed):
        """
        Строит итоговое сопоставление эталонного entry
        и восстановленного профиля после калибровки.
        """
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        # Отрисовка entry скана (сплошная синяя линия)
        self.ax.plot(theta_entry, normalize01(profile_entry), color="#222222", lw=1.0, zorder=2)  #  color="#1f77b4"
        
        # Отрисовка восстановленного (красные точки)
        self.ax.scatter(theta_reconstructed, normalize01(profile_reconstructed), color="#dc143c", s=1, alpha=0.8, zorder=3)

        # Настраиваем оси в вашем научном стиле (можно использовать _style_scientific_axes)
        _style_scientific_axes(self.ax, self.label.text(), xlabel=r"Угол, $\theta$ (угл. сек.)", ylabel="Интенсивность (норм.)")
        self.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))     # Для нормированной интенсивности — обычная десятичная шкала
        
        # Поджимаем края
        self.figure.subplots_adjust(left=0.16, right=0.98, top=0.92, bottom=0.22)

        self.canvas.show()
        self.canvas.draw_idle()