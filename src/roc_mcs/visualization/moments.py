import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import pandas as pd
from scipy.signal import savgol_filter


from roc_mcs.processing.moments import cumulative_trapezoid_manual


def plot_cii_quantile_widths(
    theta: np.ndarray,
    intensity: np.ndarray,
    *,
    peak_center: float | None = None,
    centroid: float | None = None,
    legend_details: bool = True,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Определение квантильных ширин CII",
):
    """
    Визуализирует rocking-кривую и функцию накопленной интегральной
    интенсивности (CII) с квантильными ширинами W50, W80 и W90,
    положением максимума и центроидом.

    Параметры
    ----------
    theta:
        Угловая координата, shape (N,).
    intensity:
        Интенсивность rocking-кривой, shape (N,).
        Для расчёта CII используется неотрицательная часть профиля.
    peak_center:
        Положение максимума профиля.
        Если None, определяется как theta[argmax(intensity)].
    centroid:
        Центроид профиля. Если None, вычисляется как первый момент
        неотрицательной интенсивности.
    legend_details:
        Если True, в легенде дополнительно указывается доля интегральной
        интенсивности, содержащаяся в соответствующем квантильном интервале.
    figsize:
        Размер фигуры matplotlib.
    title:
        Заголовок верхней панели.

    Возвращает
    ----------
    fig:
        Объект matplotlib Figure.
    (ax1, ax2):
        Оси верхней и нижней панелей.

    Notes
    -----
    Квантильные ширины CII определяются как

        W50 = theta_0.75 - theta_0.25
        W80 = theta_0.90 - theta_0.10
        W90 = theta_0.95 - theta_0.05

    и соответствуют центральным угловым интервалам, содержащим
    50, 80 и 90 % интегральной интенсивности соответственно.

    CII вычисляется численным интегрированием методом трапеций.
    """
    # --- 1. Проверка и подготовка входных данных ---
    theta = np.asarray(theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if theta.ndim != 1 or intensity.ndim != 1:
        raise ValueError("theta и intensity должны быть одномерными массивами.")
    if theta.size != intensity.size:
        raise ValueError("theta и intensity должны иметь одинаковую длину.")

    mask = np.isfinite(theta) & np.isfinite(intensity)
    theta = theta[mask]
    intensity = intensity[mask]

    if theta.size < 2:
        raise ValueError("Для расчёта CII необходимо хотя бы две точки.")
    if np.any(np.diff(theta) <= 0):
        raise ValueError("theta должен быть строго возрастающим.")

    intensity = np.clip(intensity, 0.0, None)

    if peak_center is None:
        peak_center = theta[np.argmax(intensity)]

    # --- CII ---
    cii = cumulative_trapezoid_manual(intensity, theta)
    total_area = cii[-1]

    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Интегральная интенсивность должна быть положительной.")

    cii /= total_area

    # --- Центроид ---
    if centroid is None:
        centroid = np.trapezoid(theta * intensity, theta) / total_area

    cii_at_centroid = np.interp(centroid, theta, cii)

    # --- Квантили CII ---
    p_25, p_75 = 0.25, 0.75
    p_10, p_90 = 0.10, 0.90
    p_05, p_95 = 0.05, 0.95

    # Для корректной обратной интерполяции при плато CII
    cii_unique, idx_unique = np.unique(cii, return_index=True)
    theta_unique = theta[idx_unique]

    t_25 = np.interp(p_25, cii_unique, theta_unique)
    t_75 = np.interp(p_75, cii_unique, theta_unique)
    t_10 = np.interp(p_10, cii_unique, theta_unique)
    t_90 = np.interp(p_90, cii_unique, theta_unique)
    t_05 = np.interp(p_05, cii_unique, theta_unique)
    t_95 = np.interp(p_95, cii_unique, theta_unique)

    # --- Квантильные ширины CII ---
    w50 = t_75 - t_25
    w80 = t_90 - t_10
    w90 = t_95 - t_05

    if legend_details:
        label_w50 = rf"$W_{{50}}$ = {w50:.3g} (интервал, содержащий $50\%$ интегральной интенсивности)"
        label_w80 = rf"$W_{{80}}$ = {w80:.3g} (интервал, содержащий $80\%$ интегральной интенсивности)"
        label_w90 = rf"$W_{{90}}$ = {w90:.3g} (интервал, содержащий $90\%$ интегральной интенсивности)"
    else:
        label_w50 = rf"$W_{{50}}$ = {w50:.3g}"
        label_w80 = rf"$W_{{80}}$ = {w80:.3g}"
        label_w90 = rf"$W_{{90}}$ = {w90:.3g}"

    # --- График ---
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1.5]},
    )

    # --- ВЕРХНЯЯ ПАНЕЛЬ ---
    ax1.plot(theta, intensity, 'k-', lw=2, label=r'Rocking curve $I(\theta)$')
    ax1.axvline(
        peak_center,
        color='red',
        linestyle='-',
        alpha=0.3,
        label=rf'Peak maximum ($\theta={peak_center:.1f}$)',
    )
    ax1.axvline(
        centroid,
        color='green',
        linestyle='-',
        alpha=0.3,
        label=rf'Centroid ($\bar{{\theta}}={centroid:.1f}$)',
    )

    mask_50 = (theta >= t_25) & (theta <= t_75)
    ax1.fill_between(
        theta, intensity,
        where=mask_50,
        color='royalblue',
        alpha=0.5,
        label=label_w50,
    )

    mask_80_left = (theta >= t_10) & (theta < t_25)
    mask_80_right = (theta > t_75) & (theta <= t_90)
    ax1.fill_between(theta, intensity, where=mask_80_left, color='cornflowerblue', alpha=0.3)
    ax1.fill_between(
        theta, intensity,
        where=mask_80_right,
        color='cornflowerblue',
        alpha=0.3,
        label=label_w80,
    )

    mask_90_left = (theta >= t_05) & (theta < t_10)
    mask_90_right = (theta > t_90) & (theta <= t_95)
    ax1.fill_between(theta, intensity, where=mask_90_left, color='lightsteelblue', alpha=0.3)
    ax1.fill_between(
        theta, intensity,
        where=mask_90_right,
        color='lightsteelblue',
        alpha=0.3,
        label=label_w90,
    )

    for t, ls in zip(
        [t_05, t_10, t_25, t_75, t_90, t_95],
        [':', '--', '-', '-', '--', ':'],
    ):
        ax1.axvline(t, color='gray', linestyle=ls, alpha=0.7)

    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_title(title, pad=15)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, max(intensity) * 1.1)

    # --- НИЖНЯЯ ПАНЕЛЬ: CII ---
    ax2.plot(theta, cii, 'k-', lw=2, label=r'CII $F(\theta)$')
    ax2.axvline(peak_center, color='red', linestyle='-', alpha=0.3)
    ax2.axvline(centroid, color='green', linestyle='-', alpha=0.3)

    levels = [
        (p_95, t_95, '95%'),
        (p_90, t_90, '90%'),
        (p_75, t_75, '75%'),
        (p_25, t_25, '25%'),
        (p_10, t_10, '10%'),
        (p_05, t_05, '5%'),
    ]

    for p, t, text in levels:
        ax2.plot([theta[0], t], [p, p], color='crimson', linestyle='--', alpha=0.6)
        ax2.plot([t, t], [0, p], color='gray', linestyle='--', alpha=0.6)
        ax2.text(theta[0] + 0.2, p + 0.02, text, color='crimson', fontsize=10, va='bottom')

    ax2.set_ylabel(r'CII $F(\theta)$')
    ax2.set_xlabel(r'Angle $\theta$ (arcsec or relative units)')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(theta[0], theta[-1])
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    return fig, (ax1, ax2)





def plot_cii_dynamics(roc_map: dict, 
                      profile_moments: pd.DataFrame = None, 
                      secondary_curves: dict = None,
                      quantiles: dict = None,
                      slices: list = None,         # <-- Время (с) для срезов профилей
                      palette: str = 'tab10',      # <-- имя палитры по умолчанию (цвета для этих профилей)
                      theta_lim: tuple = None,     # <-- НОВЫЙ АРГУМЕНТ: лимиты по оси Y (углы)
                      smooth_window: int = 11,     # <-- Окно сглаживания Савицкого-Голея (должно быть нечетным)
                      smooth_poly: int = 3         # <-- Степень полинома для сглаживания
):
    """
    Строит многоуровневую визуализацию X-ray Rocking Map.
    
    Параметры:
    - roc_map: словарь с данными 2D карты ('time_s', 'intensity', 'theta_axis'/'s_axis').
    - profile_moments: DataFrame с интегральными метриками (должен содержать 'time_s' и 'area').
    - secondary_curves: словарь вида {'имя': {'x': array, 'y': array}} для нижнего графика.
    """
    # 1. Извлекаем базовые данные 2D карты
    time_s = roc_map["time_s"]
    Z = roc_map["intensity"].T
    
    if "theta_axis" in roc_map:
        y = roc_map["theta_axis"]
        ylabel_map = "Angle (arcsec)"
    else:
        y = roc_map["s_axis"]
        ylabel_map = r"$\sin(\omega t + \varphi)$"

    # --- НАЧАЛО ДИАГНОСТИКИ ---
    print("\n" + "="*40)
    print("ДИАГНОСТИКА ОСИ УГЛОВ (Y)")
    print(f"Первые 5 значений: {y[:5]}")
    print(f"Последние 5 значений: {y[-5:]}")
    
    # Проверяем шаг (разницу между соседними углами)
    dy = np.diff(y)
    print(f"Шаг (первые 5 dy): {dy[:5]}")
    
    # Анализируем массив
    is_descending = y[0] > y[-1]
    is_uniform = np.allclose(dy, dy[0], rtol=1e-3)
    
    print(f"Массив идет по убыванию? -> {is_descending}")
    print(f"Шаг строго равномерный? -> {is_uniform}")
    print("="*40 + "\n")
    # --- КОНЕЦ ДИАГНОСТИКИ ---
    
    # 2. Вычисляем интервалы для imshow (чтобы центры пикселей совпадали с time_s)
    dt = np.diff(time_s)
    edges = np.empty(len(time_s) + 1)
    edges[1:-1] = 0.5 * (time_s[:-1] + time_s[1:])
    edges[0] = time_s[0] - 0.5 * dt[0]
    edges[-1] = time_s[-1] + 0.5 * dt[-1]

    # 3. Настраиваем динамическую сетку (GridSpec)
    has_top = profile_moments is not None
    has_bot = secondary_curves is not None
    has_slices = slices is not None

    # Задаем пропорции высоты этажей:             Top(0.8), Map(4.0), Bot(1.2)
    height_ratios = []
    if has_top: height_ratios.append(0.8)
    height_ratios.append(4.0)                    # Карта есть всегда
    if has_bot: height_ratios.append(1.2)

    # Динамическая настройка колонок
    ncols = 3 if has_slices else 2
    width_ratios = [15, 6, 0.4] if has_slices else [15, 0.3]
    fig_width = 12 if has_slices else 10
                                                             # layout="constrained" автоматически подгонит отступы, чтобы тексты не слипались
    fig = plt.figure(figsize=(12, sum(height_ratios) * 1.5)) # Убираем layout="constrained", так как он будет пытаться втиснуть отступы  layout="constrained")
    gs = fig.add_gridspec(
        nrows=len(height_ratios), 
        ncols=ncols, 
        width_ratios=width_ratios,       # 15 частей на графики, 0.4 части на ширину colorbar
        height_ratios=height_ratios,
        hspace=0.03,                  # нулевой зазор между графиками по вертикали
        wspace=0.03                   # Минимальный зазор перед колорбаром
    )

    axes_list = []                    # Для проброса вертикальных линий
    row_idx = 0

    # ---------------------------------------------------------
    # ВЕРХНИЙ ЭТАЖ: Интегральная интенсивность (опционально) (только 0-я колонка)
    # ---------------------------------------------------------
    ax_top = None
    if has_top:
        ax_top = fig.add_subplot(gs[row_idx, 0])
        ax_top.plot(profile_moments["time_s"], profile_moments["area"], 
                    color="black", lw=1.2)
        ax_top.tick_params(axis="x", labelbottom=False, direction="in")       # прячем подписи X и делаем деления внутрь
        ax_top.tick_params(axis="y", direction="in")
        ax_top.yaxis.set_major_locator(MaxNLocator(prune='lower', nbins=4))   # принудительно обрезаем самое нижнее значение (Prune), чтобы нижняя цифра оси Y верхнего графика (например, 10000) не наложилась на верхнюю цифра оси Y карты
        ax_top.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_top.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax_top.set_ylabel("Integrated intensity")
        ax_top.set_title("Rocking curve dynamics", pad=12)                    # Ставим заголовок на САМЫЙ ВЕРХ, а не на карту
        ax_top.grid(True, axis='y', linestyle="--", alpha=0.25)               # горизонтальная сетка
        # Здесь можно вызвать style_line_axes(ax_top)
        axes_list.append(ax_top)
        row_idx += 1

    # ---------------------------------------------------------
    # СРЕДНИЙ ЭТАЖ: Rocking Map (основной) + Профили + Colorbar
    # ---------------------------------------------------------
    ax_map = fig.add_subplot(gs[row_idx, 0], sharex=ax_top if has_top else None)  
    
    # ИСПРАВЛЕННЫЙ БЛОК: Используем pcolormesh вместо imshow
    # shading="auto" сам вычислит границы пикселей, так что заранее считать edges больше не нужно
    im = ax_map.pcolormesh(
        time_s, y, Z, 
        cmap="viridis", 
        shading="auto", 
        rasterized=True  # Убирает артефакты сетки при экспорте в PDF/SVG и сильно ускоряет рендер
    )
    
    ax_map.set_ylabel(ylabel_map)
    if theta_lim is not None:
        ax_map.set_ylim(theta_lim)
    
    # --- ИЗОКВАНТИЛИ И ЦЕНТРОИД ---
    if quantiles is not None:
        ax_map.plot(time_s, quantiles[0.50], color='white', lw=1.2, linestyle='-', alpha=0.9, label='Median')
        ax_map.plot(time_s, quantiles[0.25], color='white', lw=1.0, linestyle='--', alpha=0.7, label=r'$FW_{50}$')  # lw=0.8
        ax_map.plot(time_s, quantiles[0.75], color='white', lw=1.0, linestyle='--', alpha=0.7)
        ax_map.plot(time_s, quantiles[0.10], color='silver', lw=1.0, linestyle='-.', alpha=0.8, label=r'$FW_{80}$')
        ax_map.plot(time_s, quantiles[0.90], color='silver', lw=1.0, linestyle='-.', alpha=0.8)
        ax_map.plot(time_s, quantiles[0.05], color='lightgray', lw=1.0, linestyle=':', alpha=0.6, label=r'$FW_{90}$')
        ax_map.plot(time_s, quantiles[0.95], color='lightgray', lw=1.0, linestyle=':', alpha=0.6)
        # --- ЦЕНТРОИД (Если переданы profile_moments и там есть 'centroid') ---
        if has_top and "centroid" in profile_moments.columns:
            ax_map.plot( profile_moments["time_s"], profile_moments["centroid"], 
                        color='crimson', lw=1.2, linestyle='-.',        # Штрихпунктир, чтобы отличался от медианы
                        alpha=0.9, label='Centroid')
        ax_map.legend(loc='upper left', 
                      fontsize=10, 
                      markerscale=1.2,    # <-- Увеличили значки/линии в легенде
                      framealpha=0.4,     # Чуть плотнее фон, чтобы текст читался лучше
                      facecolor='black', 
                      labelcolor='white')
    # ------------------------------

    # --- ДОБАВЛЕНИЕ БОКОВЫХ ПРОФИЛЕЙ И УКАЗАТЕЛЕЙ ---
    if has_slices:
        ax_prof = fig.add_subplot(gs[row_idx, 1], sharey=ax_map)
        ax_cbar = fig.add_subplot(gs[row_idx, 2])
        # --- ДИНАМИЧЕСКАЯ КАТЕГОРИАЛЬНАЯ ПАЛИТРА ---
        cmap = plt.get_cmap(palette)       # Получаем палитру по ее строковому имени из аргумента
        if hasattr(cmap, 'colors'):        # Универсальное извлечение цветов (работает для tab10, Dark2, Set1 и др.)
            palette_colors = cmap.colors   # Извлекаем цвета по очереди. (cmap.colors содержит список RGB-кортежей палитры).
        else:
            palette_colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(slices))]
        slice_colors = [palette_colors[i % len(palette_colors)] for i in range(len(slices))]    # i % len(cmap.colors) гарантирует, что мы пойдем по второму кругу, если срезов слишком много.
        
        if smooth_window % 2 == 0: smooth_window += 1              # Гарантируем, что окно сглаживания нечетное

        print("\n" + "-"*40)
        print("ИНДЕКСЫ СРЕЗОВ ДЛЯ ПРАВОЙ ПАНЕЛИ:")
        
        for t_val, color in zip(slices, slice_colors):             # Треугольная засечка: увеличенный размер и БЕЛАЯ ОКАНТОВКА
            ax_map.plot(t_val, 0, marker='^', markersize=12, 
                        color=color, markeredgecolor='white', markeredgewidth=1.2,
                        transform=ax_map.get_xaxis_transform(), clip_on=False, zorder=10)
            
            idx = np.abs(time_s - t_val).argmin()
            print(f"Время t = {t_val} s  ->  Индекс массива (срез): {idx}")
            raw_profile = Z[:, idx]
            
            # --- Фильтр Савицкого-Голея ---
            smoothed_profile = savgol_filter(raw_profile, window_length=smooth_window, polyorder=smooth_poly)
            smoothed_profile = np.clip(smoothed_profile, 0, None)                                                # Убираем возможные артефакты фильтра (отрицательные значения на фоне шума)
            ax_prof.plot(smoothed_profile, y, color=color, linewidth=1.5, label=f'{time_s[idx]:.1f} s')
            
        ax_prof.set_xlabel("Intensity", fontsize=10)
        ax_prof.tick_params(axis='y', labelleft=False, direction="in")
        ax_prof.grid(True, linestyle='--', alpha=0.3)
        ax_prof.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_prof.set_title("Representative rocking curves", pad=12, fontsize=11)
    else:
        ax_cbar = fig.add_subplot(gs[row_idx, 1])      # Ячейка специально под колорбар

    cbar = fig.colorbar(im, cax=ax_cbar)              # Отрисовка Colorbar
    cbar.set_label("Counts", rotation=270, labelpad=15)

    if not has_bot:                                    # Если нижнего графика нет, подписи оси X будут на карте
        ax_map.set_xlabel("Time (s)")
    else:
        ax_map.tick_params(axis="both", direction="in", labelbottom=False)
    # Здесь можно вызвать: style_map_axes(ax_map)
    axes_list.append(ax_map)
    row_idx += 1
    

    # ---------------------------------------------------------
    # НИЖНИЙ ЭТАЖ: Secondary curves (Нагрузка / Ширины) (опционально) (только 0-я колонка)
    # ---------------------------------------------------------
    ax_bot = None
    if has_bot:
        ax_bot = fig.add_subplot(gs[row_idx, 0], sharex=ax_map)
        for label, curve in secondary_curves.items():
            color = curve.get("color", None)                 # Если передан цвет в словаре, используем его
            ax_bot.plot(curve["x"], curve["y"], lw=1.5, alpha=0.8, label=label, color=color)
            
        ax_bot.set_ylabel("Pressure, MPa")            # "Load / Metric"
        ax_bot.set_xlabel("Time (s)")
        ax_bot.grid(True, axis='y', linestyle="--", alpha=0.4)
        # Здесь можно вызвать style_line_axes(ax_bot)
        axes_list.append(ax_bot)

    # ---------------------------------------------------------
    # ФИНАЛЬНЫЙ ШТРИХ: Сквозные вертикальные линии сканов
    # ---------------------------------------------------------
    for ax in axes_list:
        ax.margins(x=0)        # прижимаем данные вплотную к краям (убираем белые поля слева и справа)
        for t_edge in edges:   # рисуем границы сканов на ВСЕХ существующих графиках
            line_color = "white" if ax == ax_map else "0.6"
            ax.axvline(t_edge, color=line_color, linestyle=":", linewidth=0.6, alpha=0.5)

    # ax_map.set_title(f"Rocking curve dynamics", pad=10)

    return fig