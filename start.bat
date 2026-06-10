@echo off
:: ============================================================
::  Face Check Standalone Launcher
:: ============================================================

cd /d "%~dp0"

:: -- Locate Python from local venv, then system PATH ---------
set "LOCAL_VENV=venv\Scripts\python.exe"

if exist "%LOCAL_VENV%" (
    set "PYTHON=%LOCAL_VENV%"
    echo [Launcher] Using local venv: %LOCAL_VENV%
) else (
    set "PYTHON=python"
    echo [Launcher] WARNING: No venv found – using system Python.
)

:: -- Start Face Check -----------------------------------------
echo [Launcher] Starting Face Check Standalone...
"%PYTHON%" face_check.py

pause
