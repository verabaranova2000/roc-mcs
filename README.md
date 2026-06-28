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
        [--output-folder <OUTPUT_FOLDER>] \
        [--fit-models <FIT_MODELS>] \
        [--trajectory-models <TRAJECTORY_MODELS>] \
        [--diagnostics <DIAGNOSTICS>] \
        [--save-artifact]
```

Примеры:
```bash
roc-mcs --amplitude 600 --diagnostics phase
```

```bash
roc-mcs --input-folder "D:\experiment" --amplitude 600 --reference-amplitude 400 --reference-angle 180 --fit-models gauss,pvoigt --diagnostics phase,metrics
```
При отсутствии параметра `--input-folder` используется текущий рабочий каталог.

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

fit_models: 
  - gauss
  - pvoigt
trajectory_models:
  - gauss
  - pvoigt  
diagnostics: 
  - phase
save_artifact: true  
output_folder: null
```

YAML-ключи соответствуют аргументам командной строки по правилу:

```text
--name-of-argument → name_of_argument
```

Параметры:

| Параметр                | Описание                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| `--input-folder`              | Каталог с файлами `.mcs`. Если не указан, используется текущий рабочий каталог.                                                        |
| `--amplitude`           | Амплитуда колебаний пьезоактуатора, mVpp                                        |
| `--reference-amplitude` | Опорная амплитуда пьезоактуатора для калибровки (по умолчанию: 400 mVpp)            |
| `--reference-angle`     | Угловой диапазон, соответствующий опорной амплитуде (по умолчанию: ±180 arcsec)  |
| `--output-folder`      | Каталог сохранения результатов. Если не задан (т.е. `null`), результаты сохраняются в `<input-folder>/results` |
| `--fit-models`      | Список моделей подгонки ROC-кривых `<output-folder>/qc` |
| `--trajectory-models` | Подмножество `fit_models`, для которого дополнительно выполняется анализ эволюции параметров (ridge-регрессия и фильтр Калмана) |
| `--diagnostics`      | Список диагностик для сохранения в подкаталоге `<output-folder>/qc` |
| `--save-artifact` | Сохранить полный объект эксперимента `experiment_artifact.pkl` для последующего анализа в Python |

---


## Результаты обработки

После выполнения создаются:

* `roc_map.png` — ROC-карта;
* `rocking_curve_dynamics.xlsx` — таблица интенсивностей, метаданных и параметров моделей;
* `fit/`
  * `model_evolution_<model>.png` — эволюция параметров моделей;
  * `residual_maps.png` — разностные ROC-карты для сравнения моделей; 
  * `trajectory_<model>.png` — эволюция параметров модели во времени (если модель включена в `--trajectory-models` и для неё выполняется Kalman/Ridge анализ).   
* `qc/*.png` — диагностические графики, если они были включены;
* `experiment_artifact.pkl` — сериализованный объект ExperimentArtifact с результатами обработки, параметрами моделей и метаданными эксперимента.

---

## Загрузка сохранённого результата

Сохранённый эксперимент можно восстановить в Python:

```python
import pickle

with open("experiment_artifact.pkl", "rb") as f:
    artifact_loaded = pickle.load(f)
```