@echo off
:: ============================================================
::  face_search sidecar service launcher
::  Run this from the face_search\ directory.
::  Uses the parent project's venv automatically.
:: ============================================================

cd /d "%~dp0"

:: -- Locate Python from parent venv, then system PATH ---------
set "PARENT_VENV=..\face\venv\Scripts\python.exe"
set "LOCAL_VENV=venv\Scripts\python.exe"

if exist "%PARENT_VENV%" (
    set "PYTHON=%PARENT_VENV%"
    echo [Launcher] Using parent venv: %PARENT_VENV%
) else if exist "%LOCAL_VENV%" (
    set "PYTHON=%LOCAL_VENV%"
    echo [Launcher] Using local venv: %LOCAL_VENV%
) else (
    set "PYTHON=python"
    echo [Launcher] WARNING: No venv found – using system Python.
)

:: -- Start uvicorn --------------------------------------------
echo [Launcher] Starting face_search sidecar on port 8001...
"%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8001 --workers 1

pause
