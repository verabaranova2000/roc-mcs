@echo off
setlocal
cd /d "%~dp0"

set "APP=%~dp0src\roc_mcs\gui\trs_apps\trs_apps_app.py"
set "PY="

REM 1. Явно заданный интерпретатор проекта.
if exist "%~dp0python_path.txt" (
    for /f "usebackq eol=# delims=" %%i in ("%~dp0python_path.txt") do if not defined PY set "PY=%%~i"
    if defined PY if not exist "%PY%" (
        echo [TRS Apps] WARNING: python_path.txt points to a missing Python:
        echo   %PY%
        set "PY="
    )
)

REM 2. Окружения проекта: на лабораторном ПК .venv314 будет выбран автоматически.
if not defined PY if exist "%~dp0.venv314\Scripts\python.exe" set "PY=%~dp0.venv314\Scripts\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"

REM 3. Стандартная установка Python 3.14 для текущего пользователя Windows.
if not defined PY if exist "%LocalAppData%\Programs\Python\Python314\python.exe" set "PY=%LocalAppData%\Programs\Python\Python314\python.exe"

REM 4. Любой Python из PATH — последняя автоматическая попытка.
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [TRS Apps] ERROR: Python was not found.
    echo Add python_path.txt with the full path to python.exe, or install Python 3.12+.
    echo.
    pause
    exit /b 1
)
if not exist "%APP%" (
    echo [TRS Apps] ERROR: Entry point not found:
    echo   %APP%
    echo Check that this BAT is in the repository root.
    echo.
    pause
    exit /b 1
)

echo.
echo [TRS Apps] Checking Python environment...
set "PYTHONUNBUFFERED=1"
set "PYTHONFAULTHANDLER=1"
"%PY%" -c "import sys; print('[TRS Apps] Python:', sys.executable); print('[TRS Apps] Version:', sys.version.split()[0]); print('[TRS Apps] Prefix:', sys.prefix)"
if errorlevel 1 (
    echo [TRS Apps] ERROR: Selected Python cannot be started.
    pause
    exit /b 1
)

REM Проверка обязательных зависимостей 3D.
"%PY%" -c "import sys, os, plotly; from pathlib import Path; from importlib.metadata import version; from PyQt6.QtCore import QLibraryInfo; from PyQt6.QtWebEngineWidgets import QWebEngineView; from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings; from PyQt6.QtWebChannel import QWebChannel; pv=tuple(int(x) for x in plotly.__version__.split('.')[:2]); qname=version('PyQt6-WebEngine'); qv=tuple(int(x) for x in qname.split('.')[:2]); js=Path(plotly.__file__).resolve().parent/'package_data'/'plotly.min.js'; qtbin=Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.LibraryExecutablesPath)); qtdata=Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.DataPath)); resources=Path(os.environ.get('QTWEBENGINE_RESOURCES_PATH',str(qtdata/'resources'))); process=Path(os.environ.get('QTWEBENGINEPROCESS_PATH',str(qtbin/'QtWebEngineProcess.exe'))); resource_file=resources/'qtwebengine_resources.pak'; icu_file=resources/'icudtl.dat'; print('[TRS Apps] Plotly:', plotly.__version__, 'at', plotly.__file__); print('[TRS Apps] Qt WebEngine:', qname); print('[TRS Apps] QtWebChannel:', QWebChannel.__module__); print('[TRS Apps] Qt executable dir:', qtbin); print('[TRS Apps] QtWebEngineProcess:', process, 'exists=', process.is_file()); print('[TRS Apps] Qt WebEngine resources:', resources, 'pak=', resource_file.is_file(), 'ICU=', icu_file.is_file()); print('[TRS Apps] Plotly.js:', js, 'exists=', js.is_file(), 'bytes=', js.stat().st_size if js.is_file() else 0); assert sys.version_info >= (3, 12), 'Python >= 3.12 is required'; assert pv >= (7, 1) and pv < (8, 0), 'Plotly >= 7.1,<8 is required by pyproject.toml'; assert qv >= (6, 11) and qv < (7, 0), 'PyQt6-WebEngine >= 6.11,<7 is required by pyproject.toml'; assert js.is_file() and js.stat().st_size >= 100000, 'Plotly local JavaScript asset is missing or truncated'; assert process.is_file(), 'QtWebEngineProcess.exe is missing from the selected Qt installation'; assert resource_file.is_file() and icu_file.is_file(), 'Qt WebEngine resource pack or icudtl.dat is missing from the selected Qt installation'"
if errorlevel 1 (
    echo.
    echo [TRS Apps] ERROR: Required 3D dependencies are missing or too old in this Python.
    echo Required: Python >= 3.12, Plotly >= 7.1 and < 8, PyQt6-WebEngine >= 6.11 and < 7.
    echo Install/update the project with:
    echo   "%PY%" -m pip install -e .
    echo.
    pause
    exit /b 1
)

echo [TRS Apps] Dependency preflight passed in a separate Python process.
echo [TRS Apps] This does NOT preload Qt WebEngine into the GUI process.
echo [TRS Apps] The GUI process will print its own Qt WebEngine bootstrap result.
echo [TRS Apps] Starting...
echo.
"%PY%" "%APP%" %*
if errorlevel 1 pause
endlocal
