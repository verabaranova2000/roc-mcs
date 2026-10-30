@echo off
setlocal
cd /d "%~dp0"

set "APP=%~dp0src\roc_mcs\gui\trs_apps\trs_apps_app.py"
set "PY="

REM 1. Явный локальный override
if exist "%~dp0python_path.txt" (
    for /f "usebackq eol=# delims=" %%i in ("%~dp0python_path.txt") do (
        if not defined PY set "PY=%%i"
    )
    if defined PY if not exist "%PY%" (
        echo [TRS Apps] WARNING: python_path.txt points to a missing Python:
        echo   %PY%
        echo [TRS Apps] Ignoring python_path.txt and trying automatic detection.
        set "PY="
    )
)

REM 2. Локальное .venv
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
)

REM 3. Локальное venv
if not defined PY if exist "%~dp0venv\Scripts\python.exe" (
    set "PY=%~dp0venv\Scripts\python.exe"
)

REM 4. Системный Python из PATH
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

REM 5. Python вообще не найден
if not defined PY (
    echo [TRS Apps] ERROR: Python was not found.
    echo.
    echo Expected one of:
    echo   %~dp0.venv\Scripts\python.exe
    echo   %~dp0venv\Scripts\python.exe
    echo   python from PATH
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

REM Проверяем обязательные зависимости 3D
"%PY%" -c "import plotly; from PyQt6.QtWebEngineWidgets import QWebEngineView; print('[TRS Apps] Plotly:', plotly.__version__); print('[TRS Apps] Qt WebEngine: OK')"
if errorlevel 1 (
    echo.
    echo [TRS Apps] ERROR: Required 3D dependencies are missing in this Python.
    echo [TRS Apps] Install/update the project with:
    echo.
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