@echo off
echo ============================================================
echo   Face Search EXE Builder
echo ============================================================

:: Check for PyInstaller
.\venv\Scripts\python.exe -m pip install pyinstaller

echo [Builder] Starting PyInstaller...
.\venv\Scripts\python.exe -m PyInstaller --clean FaceSearch.spec

echo [Builder] Done! Check the 'dist' folder for FaceSearch.exe
pause
