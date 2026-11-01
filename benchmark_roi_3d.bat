@echo off
setlocal
rem Run from the repository root. Uses this checkout's src tree, not another install.
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
python -m roc_mcs.gui.roi_explorer.volume_rendering_benchmark %*
exit /b %errorlevel%
