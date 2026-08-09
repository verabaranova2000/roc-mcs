from typing import Callable, Optional, Mapping, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as pe

from roc_mcs.processing.utils import normalize01

def text_append(
    ax,
    x, y,
    *parts,
    transform=None,
    part_kwargs=None,
    **kwargs,
):
    """
    Вспомогательная функция для красивой сборки текста
    с разными стилями частей.
    
    Рисует строку из нескольких частей, автоматически располагая
    каждую следующую сразу после предыдущей.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : float
        Координаты начала строки.
    *parts : str
        Последовательные части строки.
    transform : matplotlib Transform, optional
        По умолчанию ax.transAxes.
    part_kwargs : dict[int, dict], optional
        Индивидуальные параметры для отдельных частей.
        Например:
            part_kwargs={
                1: dict(color="crimson"),
                3: dict(fontweight="bold")
            }
    **kwargs
        Общие параметры text().
    """
    if transform is None:
        transform = ax.transAxes

    if part_kwargs is None:
        part_kwargs = {}

    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    texts = []
    x_cur = x

    for i, part in enumerate(parts):
        style = kwargs.copy()
        style.update(part_kwargs.get(i, {}))

        t = ax.text(
            x_cur, y,
            part,
            transform=transform,
            **style,
        )
        texts.append(t)

        fig.canvas.draw()

        bbox = t.get_window_extent(renderer=renderer)
        x_cur = transform.inverted().transform((bbox.x1, bbox.y0))[0]
    return texts




def plot_calibration_diagnostics(
    score_surface: np.ndarray,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    score_vs_B_at_best_A: np.ndarray,
    a_best: float,
    b_best: float,
    minima_path_df: pd.DataFrame,
    theta_motor: np.ndarray,
    profile_motor: np.ndarray,
    theta_reconstructed: np.ndarray,
    profile_reconstructed: np.ndarray,
    *,
    figsize: tuple[int, int] = (11, 14),
    cmap: str = "viridis",
    ink_blue: str = "#1b365d",
    crimson: str = "crimson",
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """
    Построить диагностический график калибровки.

    Parameters
    ----------
    score_surface:
        2D массив значений S(A, B). Ожидается форма
        (len(a_grid), len(b_grid)).

    a_grid, b_grid:
        Векторы значений для осей A и B.

    score_vs_B_at_best_A:
        1D массив S(A*, B) для верхней панели. Имеет вид "_M_".
        Длина должна совпадать с len(b_grid).

    a_best, b_best:
        Координаты глобального минимума.

    minima_path_df:
        DataFrame с колонками:
        - "A"
        - "B_best"      (используется для траектории локальных минимумов)
        - "score_best"  (используется для правой панели)
        
    theta_motor, profile_motor:
        Профиль от моторного сканирования.

    theta_reconstructed, profile_reconstructed:
        Профиль, восстановленный по данным MCS.

    Returns
    -------
    fig, axes
        matplotlib Figure и словарь осей.
    """
    # --------- Проверки входа ---------
    score_surface = np.asarray(score_surface)
    a_grid = np.asarray(a_grid)
    b_grid = np.asarray(b_grid)
    score_vs_B_at_best_A = np.asarray(score_vs_B_at_best_A)
    theta_motor = np.asarray(theta_motor)
    profile_motor = np.asarray(profile_motor)
    theta_reconstructed = np.asarray(theta_reconstructed)
    profile_reconstructed = np.asarray(profile_reconstructed)

    if score_surface.ndim != 2:
        raise ValueError("score_surface должен быть 2D массивом.")
    if score_surface.shape != (len(a_grid), len(b_grid)):
        raise ValueError(
            "score_surface должен иметь форму (len(a_grid), len(b_grid))."
        )
    if len(score_vs_B_at_best_A) != len(b_grid):
        raise ValueError("score_vs_B_at_best_A должен иметь длину len(b_grid).")

    required_minima_cols = {"A", "B_best", "score_best"}
    if not required_minima_cols.issubset(minima_path_df.columns):
        raise ValueError(f"minima_path_df должен содержать колонки {required_minima_cols}.")

    # --------- Служебные вычисления ---------
    finite = np.isfinite(score_surface)
    if not np.any(finite):
        raise ValueError("score_surface не содержит конечных значений.")

    vmin = np.percentile(score_surface[finite], 10)
    vmax = np.max(score_surface[finite])

    cols = np.where(finite.any(axis=0))[0]                                         # ограничить отображаемый диапазон по B только той областью, где существуют конечные значения (не inf).
    if len(cols) == 0:
        raise ValueError("Не удалось определить диапазон отображения по оси B.")
    bmin_display = b_grid[cols[0]]
    bmax_display = b_grid[cols[-1]]

    valid_slice = np.isfinite(score_vs_B_at_best_A)
    if not np.any(valid_slice):
        raise ValueError("score_vs_B_at_best_A не содержит конечных значений.")
    slice_left = b_grid[np.where(valid_slice)[0][0]]
    slice_right = b_grid[np.where(valid_slice)[0][-1]]

    # --------- Разметка figure ---------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4, 3,                                        # Сетка 4x3
        width_ratios=[3.5, 1.2, 0.15],               # [Карта / Верх, Правая панель, Colorbar]
        height_ratios=[1.2, 3.5, 0.3, 3.5],          # [Верх, Карта, Пустая строка, Нижний график]
        wspace=0.04,                                 # Зазор по горизонтали
        hspace=0.04,                                 # Зазор по вертикали
    )

    ax_top = fig.add_subplot(gs[0, 0])                       # Верхняя панель (проекция B)
    ax_map = fig.add_subplot(gs[1, 0], sharex=ax_top)        # Основная карта (слева внизу)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_map)      # Правая панель (проекция A, общая ось Y с картой)
    ax_empty = fig.add_subplot(gs[0, 1])                     # Правый верхний угол (останется пустым)
    ax_cbar = fig.add_subplot(gs[1, 2])                      # Узкая ось для цветовой шкалы
    ax_bottom = fig.add_subplot(gs[3, 0:2])                  # Занимает 4-ю строку и растягивается на 2 колонки (под картой и правой панелью)

    axes = {
        "top": ax_top,
        "map": ax_map,
        "right": ax_right,
        "empty": ax_empty,
        "cbar": ax_cbar,
        "bottom": ax_bottom,
    }

    # --------- Верхняя панель ---------
    ax_top.plot(b_grid, score_vs_B_at_best_A, color=ink_blue, lw=1.2)
    ax_top.grid(True, linestyle="-", alpha=0.25, linewidth=0.6)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.set_ylabel(r"$\mathcal{S}(A^*,B)$")
    ax_top.text(0.02, 0.97, r"(a)", transform=ax_top.transAxes,
                ha="left", va="top", fontweight="bold", fontsize=11)

    ymax_top = ax_top.get_ylim()[1]
    ax_top.annotate(
        rf"$B^*={b_best:.2f}$",
        xy=(b_best, ymax_top),
        xytext=(5, -5),
        textcoords="offset points",
        fontsize=10,
        color=crimson,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.5, alpha=0.8, pad=1.5),
    )

    # --------- Карта ---------
    im = ax_map.imshow(
        score_surface,
        origin="lower",
        aspect="auto",
        extent=[b_grid.min(), b_grid.max(), a_grid.min(), a_grid.max()],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax_map.set_xlim(bmin_display, bmax_display)
    # - траектория внутренних минимумов (белая с черной обводкой) -
    ax_map.plot(                               
        minima_path_df["B_best"],
        minima_path_df["A"],
        color="white",
        lw=1.2,
        label="Траектория минимумов",
        path_effects=[
            pe.Stroke(linewidth=2.5, foreground="black", alpha=0.8),
            pe.Normal(),
        ],
    )
    # - точка глобального минимума -
    ax_map.scatter(                            
        [b_best],
        [a_best],
        s=80,
        facecolors=crimson,
        edgecolors="black",
        linewidths=1,
        zorder=5,
        label=r"Глобальный минимум $(A^*, B^*)$",
    )

    ax_map.set_xlabel(r"Сдвиг, $B$ (угл. сек.)")
    ax_map.set_ylabel(r"Масштаб, $A$ (угл. сек./канал)")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Score Density")

    # --------- Правая панель ---------
    ax_right.plot(minima_path_df["score_best"], minima_path_df["A"], color=ink_blue, lw=1.2)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)                # Убираем дублирование подписей оси Y, так как она общая с картой
    ax_right.grid(True, linestyle="-", alpha=0.25, linewidth=0.6)
    ax_right.set_xlabel(r"$\mathcal{S}_{\min}(A)$")
    ax_right.text(0.02, 0.97, r"(b)", transform=ax_right.transAxes,
                  ha="left", va="top", fontweight="bold", fontsize=11)
    # - аннотация оптимума -
    xmax_right = ax_right.get_xlim()[1]
    ax_right.annotate(
        rf"$A^*={a_best:.2f}$",
        xy=(xmax_right, a_best),
        xytext=(-5, 5),
        textcoords="offset points",
        fontsize=10,
        color=crimson,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.5, alpha=0.8, pad=1.5),
    )

    # --------- Нижняя панель ---------
    ax_bottom.plot(
        theta_motor,
        normalize01(profile_motor),
        color=ink_blue,
        lw=1.5,
        label="КДО при моторном сканировании",
        zorder=2,
    )
    ax_bottom.scatter(
        theta_reconstructed,
        normalize01(profile_reconstructed),
        color=crimson,
        s=12,
        alpha=0.8,
        label="КДО, восстановленная по данным MCS",
        zorder=3,
    )

    text_append(
        ax_bottom,
        0.02, 0.97,
        "Сопоставление профилей при ",
        r"$(A^*,\,B^*)$",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        part_kwargs={1: dict(color=crimson)},
    )

    ax_bottom.set_xlabel(r"Угол, $\theta$ (угл. сек.)")
    ax_bottom.set_ylabel("Нормированная интенсивность")
    ax_bottom.grid(True, linestyle="-", alpha=0.25, linewidth=0.6)
    ax_bottom.legend(loc="upper right", frameon=True, edgecolor="black",
                     facecolor="#f8f9fa", fontsize=9)
    # ax_bottom.spines["top"].set_visible(False)
    # ax_bottom.spines["right"].set_visible(False)

    # --------- Пояснения ---------
    ax_empty.axis("off")
    ax_empty.text(
        0.05, 0.95,
        r"Карта функционала расхождения $\mathcal{S}(A,B)$",
        transform=ax_empty.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )

    x_sym = 0.07
    x_txt = 0.28
    y1, y2, y3, y4 = 0.72, 0.54, 0.34, 0.16
    # - траектория внутренних минимумов -
    line = Line2D([x_sym, 0.24], [y1, y1],
                  transform=ax_empty.transAxes, color="white", lw=1.6)
    line.set_path_effects([
        pe.Stroke(linewidth=3.2, foreground="black", alpha=0.95),
        pe.Normal(),
    ])
    ax_empty.add_line(line)
    ax_empty.text(x_txt, y1, "линия локальных минимумов",
                  transform=ax_empty.transAxes, ha="left", va="center", fontsize=10)
    # - выбранный минимум -
    ax_empty.scatter([0.155], [y2], transform=ax_empty.transAxes,
                     s=70, facecolors=crimson, edgecolors="black", linewidths=1.0, zorder=5)
    text_append(
        ax_empty,
        x_txt, y2,
        "выбранный минимум ",
        r"$(A^*,\,B^*)$",
        fontsize=10,
        ha="left",
        va="center",
        part_kwargs={1: dict(color=crimson)},
    )
    # - пояснение к панели (a) -
    ax_empty.text(x_sym, y3, r"(a)", transform=ax_empty.transAxes,
                  ha="left", va="center", fontsize=9.8, fontweight="bold")
    text_append(
        ax_empty,
        x_txt, y3,
        "Срез ",
        r"$\mathcal{S}($",
        r"$A^*$",
        r"$,B)$",
        fontsize=9.8,
        ha="left",
        va="center",
        part_kwargs={2: dict(color=crimson, fontweight="bold")},
    )
    # - пояснение к панели (b) -
    ax_empty.text(x_sym, y4, r"(b)", transform=ax_empty.transAxes,
                  ha="left", va="center", fontsize=9.8, fontweight="bold")
    ax_empty.text(
        x_txt, y4,
        r"Проекция $\mathcal{S}_{\min}\!\left(A\right)=\min_B \mathcal{S}(A,B)$",
        transform=ax_empty.transAxes,
        ha="left",
        va="center",
        fontsize=9.8,
    )

    # --------- Связи между панелями ---------
    con_vert = ConnectionPatch(                                     # вертикальная линия вверх к ax_top
        xyA=(b_best, a_best), coordsA=ax_map.transData,
        xyB=(b_best, 1.0), coordsB=ax_top.get_xaxis_transform(),
        color=crimson, linestyle=":", lw=1.5, alpha=0.8, zorder=4,
    )
    fig.add_artist(con_vert)

    con_horiz = ConnectionPatch(                                    # горизонтальная линия вправо к ax_right (используем get_yaxis_transform() правой панели)
        xyA=(b_best, a_best), coordsA=ax_map.transData,
        xyB=(1.0, a_best), coordsB=ax_right.get_yaxis_transform(),
        color=crimson, linestyle=":", lw=1.5, alpha=0.8, zorder=4,
    )
    fig.add_artist(con_horiz)

    # --------- Границы среза ---------
    y_slice = a_best
    # - горизонтальный срез на карте -
    ax_map.plot(                      
        [slice_left, slice_right],
        [y_slice, y_slice],
        color="white",
        linestyle="--",
        lw=1.2,
        alpha=0.4,
        zorder=3,
    )
    # - вертикальные линии от краев кривой верхнего графика -
    for x_edge in (slice_left, slice_right):
        con_edge = ConnectionPatch(
            xyA=(x_edge, y_slice), coordsA=ax_map.transData,           # конец линии: на белой линии среза карты
            xyB=(x_edge, 0.0), coordsB=ax_top.get_xaxis_transform(),   # начало линии: низ верхнего графика
            color="white",
            linestyle="--",
            lw=1.2,
            alpha=0.4,
            zorder=10,               # гарантированно поверх всех слоев и данных
            clip_on=False,           # запрещаем Matplotlib обрезать линию между графиками
        )
        fig.add_artist(con_edge)

    return fig, axes