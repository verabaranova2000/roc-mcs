# roc-mcs

Пакет для обработки MCS-данных и построения ROC-карт (rocking curve maps).

## Установка

Клонировать репозиторий:

```bash
git clone https://github.com/verabaranova2000/roc-mcs.git
cd roc-mcs
```

Установить пакет в режиме разработки:

```bash
pip install -e .
```

Обновление локальной копии:

```bash
git pull
```

---

## Запуск

Поддерживаются два режима запуска:

### 1. Ручной режим

```bash
roc-mcs --input-folder <PATH> \
        --amplitude <AMPLITUDE_MVPP> \
        [--reference-amplitude <REFERENCE_AMPLITUDE_MVPP>] \
        [--reference-angle <REFERENCE_ANGLE_ARCSEC>] \
        [--output-folder <OUTPUT_FOLDER>]
```

Пример:

```bash
roc-mcs --input-folder "D:\experiment" --amplitude 600 --reference-amplitude 400 --reference-angle 180
```

### 2. Запуск через YAML-конфигурацию

```bash
roc-mcs --config config.yaml
```

#### Формат `config.yaml`

```yaml
input_folder: "D:/experiment"

amplitude: 600
reference_amplitude: 400
reference_angle: 180

output_folder: null
```

Значение `null` означает, что каталог результатов не задан явно и будет создан автоматически как `<input_folder>/results`.

YAML-ключи соответствуют аргументам командной строки по правилу:

```text
--name-of-argument → name_of_argument
```

Параметры:

| Параметр                | Описание                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| `--input-folder`              | Каталог с файлами `.mcs`                                                        |
| `--amplitude`           | Амплитуда колебаний пьезоактуатора, mVpp                                        |
| `--reference-amplitude` | Опорная амплитуда пьезоактуатора для калибровки (по умолчанию: 400 mVpp)            |
| `--reference-angle`     | Угловой диапазон, соответствующий опорной амплитуде (по умолчанию: ±180 arcsec)  |
| `--output-folder`      | Каталог сохранения результатов. Если не задан, результаты сохраняются в `<input-folder>/results` |

---


## Результаты обработки

После выполнения создаются:

* `roc_map.png` — ROC-карта;
* `rocking_curve_dynamics.xlsx` — таблица интенсивностей и метаданных эксперимента.

