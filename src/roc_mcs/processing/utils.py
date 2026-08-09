import numpy as np
from scipy.signal import find_peaks, peak_prominences


def normalize01(y):
    y = np.asarray(y, dtype=float)
    y = y - np.percentile(y, 5)   # устойчивее, чем min
    y = np.clip(y, 0, None)
    ymax = np.max(y)
    return y / ymax if ymax > 0 else y



def normalize01_v1(y, n_top=5):
    y = np.asarray(y, dtype=float)
    
    # 1. Вычитаем фон (здесь 5-й перцентиль работает отлично)
    y = y - np.percentile(y, 5) 
    y = np.clip(y, 0, None)
    
    if len(y) == 0:
        return y
        
    # 2. Находим робастный максимум (защита от 1-2 шумовых выбросов)
    # Берем n_top самых больших значений (или меньше, если массив короткий)
    k = min(n_top, len(y))
    top_values = np.sort(y)[-k:]   
    # Медиана топ-значений отсечет случайные одиночные спайки
    ymax = np.median(top_values)

    # 3. Нормируем
    if ymax > 0:
        y_norm = y / ymax
    else:
        y_norm = y  
    return y_norm


def normalize01(y, p_low=5, p_high=99.0):
    """
    Робастная нормировка 1D-сигнала в [0, 1].

    Идея:
    - снимаем фон по нижнему процентилю;
    - находим главный пик без сглаживания всего сигнала;
    - масштаб оцениваем по области пика, а не по абсолютному max всего массива.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y

    z = y - np.percentile(y, p_low)
    z = np.clip(z, 0, None)

    if not np.any(z):
        return z

    peaks, _ = find_peaks(z)
    if len(peaks) == 0:
        scale = np.percentile(z, p_high)
        return z / scale if scale > 0 else z

    main_peak = peaks[np.argmax(z[peaks])]
    prom, left_bases, right_bases = peak_prominences(z, [main_peak])

    lo = int(left_bases[0])
    hi = int(right_bases[0])

    if hi <= lo:
        scale = np.percentile(z, p_high)
    else:
        region = z[lo:hi + 1]
        scale = np.percentile(region, p_high)

    return z / scale if scale > 0 else z
