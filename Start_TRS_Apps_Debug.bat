@echo off
setlocal
cd /d "%~dp0"

set "APP=%~dp0src\roc_mcs\gui\trs_apps\trs_apps_app.py"
set "PY="

REM 1. Явно заданный интерпретатор проекта.
if exist "%~dp0python_path.txt" (
    for /f "usebackq eol=# delims=" %%i in ("%~dp0python_path.txt") do if not defined PY set "PY=%%i"
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
"%PY%" -c "import sys; print('[TRS Apps] Python:', sys.executable); print('[TRS Apps] Version:', sys.version.split()[0])"
if errorlevel 1 (
    echo [TRS Apps] ERROR: Selected Python cannot be started.
    pause
    exit /b 1
)

REM Проверка обязательных зависимостей 3D.
"%PY%" -c "import sys, plotly; from importlib.metadata import version; from PyQt6.QtWebEngineWidgets import QWebEngineView; pv=tuple(int(x) for x in plotly.__version__.split('.')[:2]); qname=version('PyQt6-WebEngine'); qv=tuple(int(x) for x in qname.split('.')[:2]); print('[TRS Apps] Plotly:', plotly.__version__); print('[TRS Apps] Qt WebEngine:', qname); assert sys.version_info >= (3, 12), 'Python >= 3.12 is required'; assert pv >= (7, 1) and pv < (8, 0), 'Plotly >= 7.1,<8 is required by pyproject.toml'; assert qv >= (6, 11) and qv < (7, 0), 'PyQt6-WebEngine >= 6.11,<7 is required by pyproject.toml'"
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

echo [TRS Apps] Starting...
echo.
"%PY%" "%APP%" %*
if errorlevel 1 pause
endlocal
