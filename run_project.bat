@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_project.ps1" %*
if errorlevel 1 (
  echo.
  echo EfficientSAM demo failed. Check the error message above.
  pause
  exit /b %errorlevel%
)
echo.
echo EfficientSAM demo finished.
pause
