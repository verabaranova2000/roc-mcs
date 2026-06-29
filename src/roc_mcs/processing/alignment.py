import numpy as np

def find_phase(counts, n_channels, window=15):
    """
    Автоматически подбирает фазовый сдвиг, при котором две ветви ROC-кривой
    после преобразования I(k) -> I(s) наилучшим образом совмещаются.
    
    Для каждой пробной фазы φ строится фазовая координата s = sin(2π(k + φ)/N), 
    точки сортируются по s, после чего оценивается расхождение кривой 
    с её сглаженной аппроксимацией.

    Функционал качества (эмпирический):
        score = Var(I - I_smooth)

    Оптимальная фаза выбирается как:
        φ_opt = argmin(score)
    
    Параметры:
    -----
    counts : array-like
        Массив интенсивностей по каналам.
    n_channels : int
        Число каналов спектра.
    window : int
        Длина окна сглаживания для оценки критерия (moving average).
    
    Возвращает:
    -----
    best_phi : int
        Оптимальный фазовый сдвиг.
    scores : ndarray
        Значения критерия для всех проверенных фаз.
    """  
    ch = np.arange(n_channels)
    scores = []

    for phi in range(n_channels):
        s = np.sin(2*np.pi*(ch + phi)/n_channels)
        order = np.argsort(s)
        y = counts[order]
        y_smooth = np.convolve(y, np.ones(window)/window, mode="same")
        score = np.var(y - y_smooth)
        scores.append(score)

    scores = np.asarray(scores)
    best_phi = np.argmin(scores)
    phi = best_phi % (n_channels // 2) # + n_channels // 2  # 🔴 канонизация (убираем двузначность)
    # print('[DEBAG find_phase 🔴] best_phi =', best_phi, 'phi =', phi)
    return phi, scores 


def extract_branch(counts, phi, n_channels, branch="up"):
    """
    Извлечение монотонной ветви фазово-выравненного MCA-сигнала.

    Параметры
    ----------
    counts : array-like
        Счётчики по каналам MCA.
    phi : float
        Фазовый сдвиг (в каналах), определяющий положение синусоидальной развертки.
    n_channels : int
        Число каналов в измерении.
    branch : {"up", "down"}, optional
        Выбор ветви синусоидального развертывания:
        - "up"   : cos(θ) > 0 (возрастающая ветвь)
        - "down" : cos(θ) < 0 (убывающая ветвь)

    Возвращает
    ----------
    s : ndarray
        Координата sin(θ), отсортированная по возрастанию.
    y : ndarray
        Интенсивность, отсортированная в соответствии с s.

    Примечание
    ----------
    cos(θ) используется для разделения фазового периода на монотонные участки,
    sin(θ) задаёт параметрическую координату развертки.
    """    
    ch = np.arange(n_channels)
    ang = 2 * np.pi * (ch + phi) / n_channels

    if branch == "up":
        mask = np.cos(ang) > 0
    else:
        mask = np.cos(ang) < 0

    s = np.sin(ang)[mask]
    y = counts[mask]
    order = np.argsort(s)
    return s[order], y[order]