# Лабораторный компьютер: Python-окружение, зависимости и запуск TRS Apps

Инструкция для репозитория `C:\Users\User\roc-mcs` на лабораторном компьютере Windows. Команды рассчитаны на PowerShell, открытый в корне репозитория.

## 1. Что удалось восстановить из диагностического лога

Из сохранённого лога известны следующие факты:

- корень репозитория: `C:\Users\User\roc-mcs`;
- при запуске старой `calibration_app.py` использовался системный Python: `C:\Users\User\AppData\Local\Programs\Python\Python37\python.exe`;
- файл `C:\Users\User\roc-mcs\.venv\Scripts\python.exe` существовал в момент диагностики;
- файл `.venv\Scripts\python.exe` не равнозначен доказательству, что старый запуск действительно использовал это окружение: `sys.executable` показывает Python 3.7.

**Наиболее вероятный путь виртуального окружения — `C:\Users\User\roc-mcs\.venv\Scripts\`.** Версию Python внутри него необходимо проверить командами ниже. Не используй системный Python 3.7 только потому, что он указан в старом логе.

## 2. Сначала проверить окружение (ничего не устанавливает и не меняет)

Открой VS Code на лабораторном компьютере, открой папку `C:\Users\User\roc-mcs`, затем Terminal → New Terminal. Убедись, что терминал PowerShell находится в корне репозитория:

```powershell
Set-Location 'C:\Users\User\roc-mcs'
Get-Location
Test-Path .\pyproject.toml
Test-Path .\.venv\Scripts\python.exe
Test-Path .\.venv\Scripts\pythonw.exe
```

Ожидаемый результат: `Get-Location` показывает корень `roc-mcs`, а два последних `Test-Path` — `True`. Если `pyproject.toml` показывает `False`, остановись: команды `pip install -e .` из этого каталога не сработают; сначала найди корень проекта, где лежит `pyproject.toml`.

Проверь конкретно Python внутри `.venv`, не активируя случайный системный Python:

```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py --version
& $py -c "import sys, struct; print('executable =', sys.executable); print('bits =', struct.calcsize('P') * 8); print('prefix =', sys.prefix); print('base_prefix =', sys.base_prefix)"
& $py -m pip --version
```

`bits` должен быть `64`. Если версия ниже 3.12, не устанавливай в это окружение последнюю версию JAX вслепую: текущий JAX на PyPI требует Python >= 3.12. Современный JAX также официально отмечает CPU на Windows x86-64 как экспериментальную конфигурацию бинарных wheels. См. раздел «JAX» ниже.

Дополнительно можно посмотреть найденные на компьютере интерпретаторы:

```powershell
py -0p
where.exe python
where.exe pythonw
```

`where.exe python` показывает варианты в `PATH`, но **не гарантирует**, что они относятся к нужному окружению. Для запуска проекта надёжнее использовать полный путь `$py`.

## 3. Зафиксировать Python для ярлыков TRS Apps

В новом launcher сначала автоматически проверяются `.venv` и `venv`. Чтобы явно закрепить нужное окружение, создай в корне `C:\Users\User\roc-mcs` два текстовых файла. В каждом должна быть одна строка с полным путём, без кавычек и без комментариев.

### `python_path.txt`

Содержимое:

```text
C:\Users\User\roc-mcs\.venv\Scripts\python.exe
```

### `pythonw_path.txt`

Содержимое:

```text
C:\Users\User\roc-mcs\.venv\Scripts\pythonw.exe
```

Проверить, что оба файла существуют, можно так:

```powershell
Test-Path 'C:\Users\User\roc-mcs\.venv\Scripts\python.exe'
Test-Path 'C:\Users\User\roc-mcs\.venv\Scripts\pythonw.exe'
```

Если второй путь отсутствует, не вписывай случайный `pythonw.exe`: проверь содержимое `Scripts` командой `Get-ChildItem .\.venv\Scripts\python*.exe` и сначала восстанови корректное окружение.

Создать файлы из PowerShell без Блокнота можно так:

```powershell
@'
C:\Users\User\roc-mcs\.venv\Scripts\python.exe
'@ | Set-Content -Encoding ascii .\python_path.txt

@'
C:\Users\User\roc-mcs\.venv\Scripts\pythonw.exe
'@ | Set-Content -Encoding ascii .\pythonw_path.txt
```

Это пример именно для лабораторного логина `User`. Если имя пользователя или путь репозитория на компьютере изменится, поправь пути соответственно. Файлы `python_path.txt` и `pythonw_path.txt` локальные; не добавляй персональные пути с лабораторного компьютера в Git. Если эти файлы оказались отслеживаемыми Git, внеси их в `.gitignore`, оставив в репозитории только `.example`.

## 4. Установка и обновление зависимостей проекта

Всегда выполняй команды из корня, где находится `pyproject.toml`, и явно используй Python выбранного окружения:

```powershell
Set-Location 'C:\Users\User\roc-mcs'
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
& $py -m pip install -e .
& $py -m pip check
```

- `pip install -e .` устанавливает проект в editable-режиме: импорты `roc_mcs` указывают на текущую рабочую копию в `src`.
- Это устанавливает или обновляет зависимости, объявленные в `pyproject.toml`, но не удаляет все посторонние пакеты из окружения.
- После `git pull` повтори `& $py -m pip install -e .`, чтобы применить изменения зависимостей из конфигурации проекта.
- Для диагностики проверь импорт и физический путь пакета:

```powershell
& $py -c "import sys, roc_mcs; print('Python:', sys.executable); print('roc_mcs:', roc_mcs.__file__)"
```

Если `pyproject.toml` не менялся, а нужно лишь проверить состояние, достаточно `& $py -m pip check`. Не обновляй все пакеты без разбора перед реальным экспериментом: новая версия `numpy`, `scipy`, `PyQt6` или JAX может повлиять на совместимость. Сначала обнови и проверь в отдельном окне, а перед опытом сохрани `& $py -m pip freeze > requirements-lab-snapshot.txt`.

## 5. JAX и 64-битная точность

Здесь важно различать две вещи:

1. **64-битный Python** — архитектура интерпретатора; проверь `bits = 64`.
2. **64-битные типы JAX** — настройка `jax_enable_x64`; она управляет тем, могут ли массивы JAX использовать `float64`/`int64`. Сам факт установки JAX эту настройку не включает.

В актуальной документации JAX CPU на Windows x86-64 поддерживается, но соответствующие бинарные wheels обозначены как экспериментальные. Актуальный пакет `jax` на PyPI на дату подготовки этого файла требует Python >= 3.12. Поэтому старый Python 3.7 из диагностического лога не подходит для установки современного JAX. Источники: [официальная установка JAX](https://docs.jax.dev/en/latest/installation.html), [требования JAX на PyPI](https://pypi.org/project/jax/), [документация `jax_enable_x64`](https://docs.jax.dev/en/latest/default_dtypes.html).

### 5.1. Если `.venv` использует 64-битный Python 3.12 или новее

После установки зависимостей проекта попробуй установить CPU-вариант JAX:

```powershell
& $py -m pip install -U jax
& $py -m pip check
```

На Windows не устанавливай GPU/CUDA extra, если специально не настраиваешь поддерживаемую GPU-платформу. Для обычного CPU-расчёта нужен базовый `jax`.

Проверь импорт, backend и фактический dtype:

```powershell
& $py -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp; x = jnp.asarray([1.0]); print('JAX:', jax.__version__); print('backend:', jax.default_backend()); print('x64:', jax.config.read('jax_enable_x64')); print('dtype:', x.dtype)"
```

Ожидается `backend: cpu`, `x64: True`, `dtype: float64`. Если установка завершается ошибкой или `pip check` сообщает конфликты, не игнорируй их — сохрани полный вывод.

### 5.2. Если `.venv` старше Python 3.12

Не пытайся насильно устанавливать последний JAX в старое окружение. Сначала проверь совместимость проекта с Python 3.12, особенно закреплённые в `pyproject.toml` версии NumPy/SciPy, PyQt6 и h5py. Не удаляй старую `.venv` до проверки.

Если установлен 64-битный Python 3.12, создай отдельную тестовую среду рядом со старой:

```powershell
Set-Location 'C:\Users\User\roc-mcs'
py -0p
py -3.12 -m venv .venv312
$py312 = (Resolve-Path .\.venv312\Scripts\python.exe).Path
& $py312 -m pip install --upgrade pip
& $py312 -m pip install -e .
& $py312 -m pip install -U jax
& $py312 -m pip check
& $py312 -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp; print(jax.__version__, jax.default_backend(), jnp.asarray([1.0]).dtype)"
```

`py -3.12` работает, только если Python 3.12 установлен и зарегистрирован в Windows Python Launcher. Если команда не найдена, сначала установи подходящий 64-битный Python и проверь `py -0p`.

После установки проверь приложение на тестовых данных: импорт всех GUI-модулей, открытие HDF5, калибровку, экспорт и работу MCS/CSV-стримеров. **Не переключай ярлыки на `.venv312`, пока эта проверка не пройдёт.** Если тестовая среда совместима, укажи её в обоих `python*_path.txt`:

```text
C:\Users\User\roc-mcs\.venv312\Scripts\python.exe
```

и в `pythonw_path.txt`:

```text
C:\Users\User\roc-mcs\.venv312\Scripts\pythonw.exe
```

### 5.3. Включение x64 в коде

Флаг необходимо задавать в начале вычислительного модуля, до создания JAX-массивов и до запуска компилируемых (`jax.jit`) вычислений:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
```

Для проекта лучше иметь одну общую точку настройки вычислительной точности и не включать/выключать этот флаг в разных функциях. После изменения перезапусти Python/Jupyter: уже созданные массивы и JIT-кэши не следует считать перепроверенными автоматически.

## 6. Проверка ярлыков и запуск приложений

### Диагностика перед обычным запуском

Из корня репозитория:

```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -c "import sys; print(sys.executable); import PyQt6; print('PyQt6 OK'); import numpy, h5py; print('NumPy/h5py OK')"
& $py -m pip check
```

Если проект использует JAX, выполни и тест из раздела 5.1.

### TRS Apps — обычный запуск

Дважды щёлкни `Start_TRS_Apps.vbs` или ярлык на него. Он использует `pythonw.exe`, поэтому консольное окно не появляется. Входная точка сама подготавливает `sys.path` для каталога `src`.

### TRS Apps — запуск для диагностики

Дважды щёлкни `Start_TRS_Apps_Debug.bat`. При ошибке окно консоли останется открытым, чтобы можно было скопировать traceback. Если окно быстро закрывается, запускай BAT из уже открытого PowerShell:

```powershell
.\Start_TRS_Apps_Debug.bat
```

### ROI Explorer отдельно

- обычный запуск: `Start_ROI_Explorer.vbs`;
- диагностический запуск: `Start_ROI_Explorer_Debug.bat`.

### Live Scans отдельно

Если у проекта есть отдельный launcher `Start_Live_Scans.vbs`, он должен использовать тот же `pythonw_path.txt` и каталог `.venv`. В противном случае используй TRS Apps и запускай Live Scans из вертикальной панели внутри него.

### Где смотреть журнал ошибок

Для невидимого запуска проверь:

```text
%LOCALAPPDATA%\roc_mcs\logs\trs_apps_startup.log
%LOCALAPPDATA%\roc_mcs\logs\roi_explorer_startup.log
```

Для первичной диагностики всегда предпочитай `.bat`, а не `.vbs`.

## 7. Подготовка VS Code

1. Открой папку `C:\Users\User\roc-mcs` в VS Code.
2. Нажми `Ctrl+Shift+P` → `Python: Select Interpreter`.
3. Выбери `C:\Users\User\roc-mcs\.venv\Scripts\python.exe` (или проверенное `.venv312`, если ты сознательно перешла на него).
4. Открой новый терминал; проверь `python -c "import sys; print(sys.executable)"`.
5. Запускай установку зависимостей через `python -m pip`, а не просто `pip`: так pip гарантированно относится к выбранному интерпретатору.

Активация окружения в PowerShell необязательна, если команды используют полный путь. Если хочется активировать его вручную:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

Если выполнение скриптов запрещено политикой PowerShell, не меняй системную политику наугад; используй команды через полный путь `$py`, это не требует активации.

## 8. После обновления репозитория из Git

```powershell
Set-Location 'C:\Users\User\roc-mcs'
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
# 1. Забрать изменения, согласованные в Git:
git pull
# 2. Переустановить проект в editable-режиме и применить декларативные зависимости:
& $py -m pip install -e .
# 3. Проверить совместимость зависимостей:
& $py -m pip check
# 4. Сохранить информацию о Python и ключевых библиотеках:
& $py -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.architecture())"
& $py -m pip freeze > .\requirements-lab-snapshot.txt
```

Не делай `git pull` во время реального эксперимента и не обновляй зависимости непосредственно перед измерением. Сначала тестируй изменения вне рабочего запуска.

## 9. Минимальный чек-лист перед лабораторным экспериментом

- [ ] Ярлык открывает именно ожидаемое окружение; `sys.executable` проверен.
- [ ] `python -m pip check` завершается без конфликтов.
- [ ] `roc_mcs.__file__` указывает на нужный репозиторий.
- [ ] Для JAX тест показывает `float64` там, где это требуется.
- [ ] Калибровка открывается и использует ожидаемые HDF5/MCS и параметры.
- [ ] Экспорт на копии старого опыта создаёт `results` с нужными файлами.
- [ ] Наблюдение за каталогом проверено на проигрывателе/копии файлов, без подключения оборудования.
- [ ] Для реального управления прессом отдельно проверены COM-порты, аварийный останов и процедура безопасного завершения.
- [ ] Сохранена версия зависимостей (`requirements-lab-snapshot.txt`).

---

**Главное правило:** всегда знать, какой именно `python.exe` запускает программу. В старом логе это системный Python 3.7; для новых ярлыков мы явно выбираем `.venv\Scripts\pythonw.exe` / `.venv\Scripts\python.exe` и проверяем версию до установки JAX.
