import numpy as np
from datetime import datetime
from pathlib import Path


def read_f32(raw, start, end):
    return float(np.frombuffer(raw[start:end], dtype="<f4")[0])

def parse_header(file):
    """ Парсер заголовка """
    raw = np.fromfile(file, dtype=np.uint8)

    # text
    dt_str = raw[20:36].tobytes().decode()
    dt = datetime.strptime(dt_str, "%H:%M:%S%m%d%Y") # время старта сбора данных с точностью только до секунды

    instrument_len = int(raw[64])         # ❗ не доказано
    instrument = raw[65:88].tobytes().decode(errors="ignore").rstrip("\x00")

    sample_len = int(raw[128])            # ❗ не доказано
    sample = raw[129:133].tobytes().decode(errors="ignore")
    
    # numeric views
    u16 = np.frombuffer(raw[:256], dtype="<u2")
    u32 = np.frombuffer(raw[:256], dtype="<u4")

    return {
        "file_info": {
            "datetime": dt,
            "instrument": instrument,
            "sample": sample,
        },

        "acquisition": {
            "pass_current": int(u32[3]),   # ✔ подтверждено изменением файла
            "pass_length": int(u16[5]),    # ✔ подтверждено изменением файла
            "channel": 0,                  # пока просто как отображение GUI
        },

        "pass_control": {
            "pass_count_preset": int(u32[4]),          # ✔ подтверждено изменением файла
            "threshold_V": read_f32(raw, 52, 56),      # ✔ подтверждено изменением файла
            "bin_width_us": read_f32(raw, 222, 226),   # ✔ подтверждено изменением файла
        },            

        "input_control": {
            "sca_lower_V": read_f32(raw, 218, 222),      # ✔ подтверждено изменением файла                       
            "sca_upper_V": read_f32(raw, 214, 218),      # ✔ подтверждено изменением файла          
            "disc_threshold_V": read_f32(raw, 210, 214), # ✔ подтверждено изменением файла
            "disc_edge": None,          # rising/falling
            "input_mode": None,         # SCA / Disc
            "impedance": None,          # 50 Ohm / 1k Ohm
        },
        "format": {
            "header_signature": int(u16[0]),
            "header_size": int(u16[2]),
            "tail_tag": raw[248:256].tobytes().decode(errors="ignore"),
        },
        "header_unknown_blocks": {
            "bytes_6_9": raw[6:10].copy(),
            "bytes_36_51": raw[36:52].copy(),
            "bytes_56_63": raw[56:64].copy(),
            "bytes_88_127": raw[88:128].copy(),
            "bytes_133_191": raw[133:192].copy(),
            "bytes_192_210": raw[192:210].copy(),
            "bytes_226_247": raw[226:248].copy(),
        }
    }


# =========== Парсер файла =================
def load_mcs(file):
    """
    Загружает данные из файла ORTEC MCS.

    Заголовок считывается из первых 256 байт файла.
    Спектр считывается как массив uint32 начиная со смещения
    256 байт. Временная шкала рассчитывается по числу активных
    каналов и ширине временного бина (bin width).

    Возвращает словарь с заголовком, спектром и временной шкалой.
    """
    file = Path(file)
    raw = np.fromfile(file, dtype=np.uint8)

    header = parse_header(file)
    n_channels = header["acquisition"]["pass_length"]
    dt_us = header["pass_control"]["bin_width_us"]

    counts_raw = np.frombuffer(raw, dtype="<u4", offset=256, count=n_channels)
    x_us = np.arange(n_channels, dtype=np.float64) * dt_us   # преобразование индексов каналов в время
    x_s = x_us * 1e-6                                        # здесь X в секундах, шаг = 10 µs
    
    return {
        "file_name": file.name,
        "path": str(file),
        "header": header,

        "counts": counts_raw,

        "x_us": x_us,
        "x_s": x_s,

        "n_channels": n_channels,
        "storage_channels": len(counts_raw),
    }    


def find_mcs_files(folder):
    """
    Возвращает список MCS-файлов, отсортированных по времени измерения.
    """
    files = list(Path(folder).glob("*.mcs"))
    files.sort(key=lambda f: load_mcs(f)["header"]["file_info"]["datetime"])
    return files