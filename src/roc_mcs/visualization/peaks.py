import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import AutoMinorLocator
import matplotlib.font_manager as fm


from roc_mcs.utils import to_yxt



# ============================================================
# Peak coint.
# ============================================================

def plot_peak_count_map(data_3d, th_array, peak_catalog, theme='dark', pixel_size_um=55.0, save_path=None):
    """
    Строит 2D-карту числа пиков топограммы-имаджинга на детекторе 
    с модуляцией альфа-канала по интегральной интенсивности.
    
    Параметры:
    - data_3d (np.ndarray): 3D массив экспериментальных данных.
    - th_array (np.ndarray): 1D массив оси сканирования (theta).
    - peak_catalog (pd.DataFrame): Каталог пиков, содержащий колонки ['y', 'x'].
    - theme (str): Цветовая тема ('dark' или 'light').
    - save_path (str, optional): Путь для сохранения изображения.
    """
    
    # --- 0. РЕЕСТР ТЕМ ОФОРМЛЕНИЯ (THEME REGISTRY) ---
    themes = {
        'dark': {
            'bg': 'black',             # Фон фигуры и осей
            'text': 'white',           # Цвет шрифтов и тиков
            'spine_main': 'white',     # Рамка основного графика
            'spine_cax': '#444444',    # Рамка легенды
            'grid': '#666666',         # Цвет сетки
            'zero_domain': '#000000',  # Цвет пикселей, где 0 пиков
            'cmap_colors': ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        },
        'light': {
            'bg': 'white',
            'text': 'black',
            'spine_main': 'black',
            'spine_cax': '#AAAAAA',
            'grid': '#CCCCCC',
            'zero_domain': '#FFFFFF',
            'cmap_colors': ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        }
    }
    
    if theme not in themes:
        raise ValueError("Параметр theme должен быть 'dark' или 'light'")
    
    style = themes[theme]
    
    # --- 1. РАСЧЕТ ИНТЕГРАЛЬНОЙ ИНТЕНСИВНОСТИ ---
    intensity_map = np.trapezoid(data_3d, x=th_array, axis=0)
    Ny, Nx = intensity_map.shape

    # --- 2. ИЗВЛЕЧЕНИЕ КАРТЫ ДОМЕНОВ ИЗ КАТАЛОГА ПИКОВ ---
    domain_map = np.zeros((Ny, Nx), dtype=int)
    
    if not peak_catalog.empty:
        peak_counts = peak_catalog.groupby(['y', 'x']).size().reset_index(name='n_peaks')
        domain_map[peak_counts['y'].values, peak_counts['x'].values] = peak_counts['n_peaks'].values

    # --- 3. АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ И АЛЬФА-КАНАЛ ---
    max_val = np.max(intensity_map)
    exp_order = int(np.floor(np.log10(max_val))-1) if max_val > 0 else 4
    scale_factor = 10.0 ** exp_order

    intensity_scaled = intensity_map / scale_factor
    int_min = np.percentile(intensity_scaled, 5)
    int_max = np.percentile(intensity_scaled, 99)
    
    if int_max == int_min:
        alpha_map = np.ones_like(intensity_scaled)
    else:
        alpha_map = (intensity_scaled - int_min) / (int_max - int_min)
    alpha_map = np.clip(alpha_map, 0.0, 1.0) ** 0.6


    # --- 4. ФОРМИРОВАНИЕ RGBA-МАССИВА ТОПОГРАММЫ ---
    full_palette = [style['zero_domain']] + style['cmap_colors']     # собираем палитру: [Цвет нулевого фона] + [Цвета доменов]
    cmap_discrete = mcolors.ListedColormap(full_palette)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap_discrete.N)

    img_rgba = cmap_discrete(norm(domain_map))
    img_rgba[..., 3] = alpha_map

    # --- 5. СОЗДАНИЕ 2D-ЛЕГЕНДЫ И РЕНДЕР ---
    n_int = 100  
    legend_rgba = np.zeros((5, n_int, 4))
    alpha_levels = np.linspace(0.0, 1.0, n_int) ** 0.6 
    
    for i in range(5):
        legend_rgba[i, :, :3] = mcolors.to_rgb(full_palette[i + 1])
        legend_rgba[i, :, 3] = alpha_levels

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # ПРИМЕНЕНИЕ ТЕМЫ: Фон
    fig.patch.set_facecolor(style['bg'])
    ax.set_facecolor(style['bg'])

    ax.imshow(img_rgba, origin='lower', interpolation='nearest')

    cax = inset_axes(ax, width="35%", height="15%", loc='upper right', borderpad=2.5)
    cax.set_facecolor(style['bg'])
    
    cax.imshow(legend_rgba, origin='lower', aspect='auto', 
               extent=[int_min, int_max, 0.5, 5.5], interpolation='nearest')

    # ПРИМЕНЕНИЕ ТЕМЫ: Тексты и оси легенды
    cax.set_yticks([1, 2, 3, 4, 5])
    cax.set_yticklabels(['1', '2', '3', '4', '5+'], color=style['text'], fontsize=9)
    cax.set_ylabel('Число компонент', color=style['text'], fontsize=10, labelpad=5)
    cax.set_xlabel(rf'Интенсивность, $I$ ($10^{{{exp_order}}}$ имп.)', color=style['text'], fontsize=9)
    cax.tick_params(axis='both', colors=style['text'], labelsize=8, right=False, top=False)

    for spine in cax.spines.values():
        spine.set_color(style['spine_cax'])
        spine.set_linewidth(0.8)

    # ПРИМЕНЕНИЕ ТЕМЫ: Тексты и оси главного графика
    ax.set_title("Пространственное распределение числа дифракционных компонент\nи интегральной интенсивности", 
                 color=style['text'], fontsize=13, pad=15)
    ax.set_xlabel("Detector X (pixels)", color=style['text'], fontsize=11)
    ax.set_ylabel("Detector Y (pixels)", color=style['text'], fontsize=11)
    ax.tick_params(colors=style['text'])

    for spine in ax.spines.values():
        spine.set_color(style['spine_main'])

    # ПРИМЕНЕНИЕ ТЕМЫ: Сетка
    ax.grid(True, color=style['grid'], linestyle=':', linewidth=1.0)

    # --- 6. МАСШТАБНАЯ ЛИНЕЙКА (SCALE BAR) ---
    if pixel_size_um is not None:
        target_size_um = 2000.0                           # Поскольку размер матрицы 14 мм (256 * 55 мкм = 14080 мкм), 
        label_text = '2 mm'                               # самая наглядная масштабная линейка будет длиной 2 мм (2000 мкм)
        size_in_pixels = target_size_um / pixel_size_um   # Пересчитываем физическую длину в пиксели детектора
        
        fontprops = fm.FontProperties(size=12, weight='bold')   # family='sans-serif')
        
        scalebar = AnchoredSizeBar(
            ax.transData,             # Привязка к координатам графика (к пикселям)
            size=size_in_pixels,      # Длина линейки в пикселях
            label=label_text,         # Текст над линейкой
            loc='lower left',         # Позиция (нижний левый угол)
            pad=0.8,                  # Отступ от края осей
            color=style['text'],      # Цвет интегрирован в нашу систему тем! (черный/белый)
            frameon=False,            # Убираем уродливую рамку вокруг линейки
            size_vertical=1.0,        # Толщина самой линии (в координатах данных)
            fontproperties=fontprops, # Настройки шрифта
            sep=4                     # Отступ между линией и текстом
        )
        ax.add_artist(scalebar)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    return fig, ax








# ============================================================
# Pixels.
# ============================================================


def get_window_curve(
    data_yxt,
    x,
    y,
    window_size=3,
    exclude_target=False,
):
    """
    Получение средней rocking curve в окне вокруг (x, y).

    data_yxt
        Данные в формате (y, x, theta).

    exclude_target=False
        Среднее по всему окну, включая target pixel.

    exclude_target=True
        Leave-one-out: target pixel исключается из среднего.
    """
    data_yxt = np.asarray(data_yxt, dtype=float)

    Ny, Nx, Ntheta = data_yxt.shape
    half_w = window_size // 2

    y_s = max(0, y - half_w)
    y_e = min(Ny, y + half_w + 1)
    x_s = max(0, x - half_w)
    x_e = min(Nx, x + half_w + 1)

    block = data_yxt[y_s:y_e, x_s:x_e, :]
    curves = block.reshape(-1, Ntheta)

    if exclude_target:
        local_y = y - y_s
        local_x = x - x_s
        local_index = local_y * (x_e - x_s) + local_x

        keep = np.ones(len(curves), dtype=bool)
        keep[local_index] = False
        curves = curves[keep]

    if len(curves) == 0:
        raise ValueError("No neighboring pixels available")

    return np.mean(curves, axis=0)



def plot_pixel_neighborhood(
    data_3d, 
    theta, 
    pixel_peaks, 
    center_x, 
    center_y, 
    branch_a_id=None,
    branch_b_id=None,
    window_size=3, 
    grid_size=3,  # <-- НОВЫЙ ПАРАМЕТР (количество панелей: 1, 2, 3, 4 и т.д.)
    normalize: bool = True
):
    """
    Строит сетку rocking curves вокруг заданного detector pixel,
    отражающую реальное физическое расположение пикселей вокруг (center_x, center_y).
    При включенной нормировке интенсивности всех панелей приводятся к единому масштабу 
    относительно глобального максимума внутри отображаемой сетки.

    data_3d
        Исходный массив данных. Приводится к формату (y, x, theta)
        через to_yxt().

    theta
        Theta axis.

    pixel_peaks
        Lookup:
            (x, y) -> list of peak dictionaries

    center_x, center_y
        Центральный detector pixel.

    branch_a_id, branch_b_id
        Идентификаторы ветвей для выделения.

    window_size
        Размер пространственного окна для averaged curve.

    grid_size
        Размер сетки панелей.

    normalize
        Флаг нормировки. Если True, данные делятся на общий максимальный сигнал 
        в пределах отрисовываемой сетки (grid_size × grid_size). Это сохраняет 
        физическое соотношение амплитуд между соседними пикселями на графике.
    """
    data_yxt = to_yxt(data_3d, theta)
    Ny, Nx, Ntheta = data_yxt.shape
    
    # Размер фигуры (примерно 2.5 дюйма на каждую панель, но не меньше базового)
    fig_width = max(4.0, grid_size * 2.6)
    fig_height = max(3.5, grid_size * 2.3)
    
    fig, axes = plt.subplots(
        grid_size, grid_size, 
        figsize=(fig_width, fig_height),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
        squeeze=False,
        layout='constrained'
    )
    
    legend_elements = {}
    # Смещения для центрирования сетки.
    offset_start = -(grid_size // 2)
    offsets = [offset_start + i for i in range(grid_size)]

    # Единый нормировочный фактор для всех пикселей
    if normalize:
        # границы сетки (учитываем, чтобы не выйти за края детектора)
        x_min, x_max = max(0, center_x + offsets[0]), min(Nx, center_x + offsets[-1] + 1)
        y_min, y_max = max(0, center_y + offsets[0]), min(Ny, center_y + offsets[-1] + 1)      
        # глобальный максимум только в пределах отрисовываемой сетки
        grid_max = np.max(data_yxt[y_min:y_max, x_min:x_max, :])
        norm_factor = grid_max if grid_max > 0 else 1.0
    else:
        norm_factor = 1.0

    
    for row_idx, dy in enumerate(offsets):
        for col_idx, dx in enumerate(offsets):
            ax = axes[row_idx, col_idx]
            x = center_x + dx                         # Физические координаты текущего пикселя на детекторе
            y = center_y + dy
            if y < 0 or y >= Ny or x < 0 or x >= Nx:  # Проверка на выход за границы детектора
                ax.set_visible(False)                
                continue

            # --- Извлечение и НОРМИРОВКА данных и отрисовка roc-кривых ---
            I_exp = data_yxt[y, x, :].astype(float) / norm_factor
            mean_curve = get_window_curve(data_yxt, x=x, y=y, window_size=window_size).astype(float) / norm_factor
            l_exp, = ax.plot(theta, I_exp, color='k', linewidth=1.2)
            l_mean, = ax.plot(theta, mean_curve, color='r', alpha=0.7, linewidth=1.5)
            
            if "Experimental" not in legend_elements:
                legend_elements["Experimental"] = l_exp
            if f"{window_size}×{window_size} Averaged" not in legend_elements:
                legend_elements[f"{window_size}×{window_size} Averaged"] = l_mean

            # --- Отрисовка найденных пиков ---
            for peak in pixel_peaks.get((int(x), int(y)), ()):
                branch_id = peak.get("branch_id")
                if branch_a_id is not None and branch_id == branch_a_id:
                    color, ls, lbl = "#1f77b4", "--", f"Branch {branch_a_id}"
                elif branch_b_id is not None and branch_id == branch_b_id:
                    color, ls, lbl = "#ff7f0e", "--", f"Branch {branch_b_id}"
                else:
                    color, ls, lbl = "gray", ":", "Seeds"
            
                vl = ax.axvline(peak["theta"], color=color, linestyle=ls, linewidth=1.2, alpha=0.8)
                if lbl not in legend_elements:
                    legend_elements[lbl] = vl
            
            # --- Информационная подпись внутри панели ---
            label_text = f"Pixel: {x}, {y}"                      # Для наглядности центральный пиксель можно слегка выделить в подписи
            if dx == 0 and dy == 0:
                label_text += "\n(Center)" 
            ax.text(0.05, 0.92, label_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))
            ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
    fig.supxlabel(r"$\theta$ (arcsec)", fontsize=12)
    ylabel = r"$I/I_{\max}$ (rel. u.)" if normalize else "Intensity (counts)"
    fig.supylabel(ylabel, fontsize=12)
    fig.legend(
        legend_elements.values(), 
        legend_elements.keys(), 
        loc='outside upper center',  
        ncol=4, 
        frameon=False, 
        fontsize=10
    )
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, hspace=0.0, wspace=0.0)
    return fig, axes