@echo off
title Face Search AI - EXE Builder
echo [BUILDER] Preparing to compile Face Search AI into a standalone EXE...
echo [BUILDER] This may take several minutes due to AI model size.

:: --- Ensure PyInstaller is installed ---
echo [BUILDER] Verifying PyInstaller...
venv\Scripts\python.exe -m pip install pyinstaller >nul 2>&1

:: --- Run Build ---
echo [BUILDER] Starting compilation (Background Mode enabled)...
venv\Scripts\python.exe -m PyInstaller --clean FaceSearchAI.spec

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the output above.
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Build complete! 
echo [SUCCESS] Your standalone deployment folder is in: dist\FaceSearchAI
echo [SUCCESS] You can copy the "FaceSearchAI" folder to any other Windows system.
echo.
pause
