@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON="

rem Prefer the interpreter explicitly configured for this ROC-MCS checkout.
if exist "%ROOT%python_path.txt" (
    for /f "usebackq delims=" %%P in ("%ROOT%python_path.txt") do (
        if not defined PYTHON if exist "%%P" set "PYTHON=%%P"
    )
)
if not defined PYTHON if exist "%ROOT%.venv314\Scripts\python.exe" set "PYTHON=%ROOT%.venv314\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%venv\Scripts\python.exe" set "PYTHON=%ROOT%venv\Scripts\python.exe"

if defined PYTHON (
    "%PYTHON%" "%ROOT%launch_roi_vtk_h5_viewer.py" %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.14 "%ROOT%launch_roi_vtk_h5_viewer.py" %*
    ) else (
        python "%ROOT%launch_roi_vtk_h5_viewer.py" %*
    )
)
if errorlevel 1 (
    echo.
    echo Viewer failed to start. See the error above.
    pause
)
endlocal
