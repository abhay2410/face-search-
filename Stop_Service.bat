@echo off
title Face Search AI - Stop Service
echo [SYSTEM] Searching for background Face Search processes...
:: Find the PID of the python process running app.py
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    echo [SYSTEM] Stopping process PID: %%a
    taskkill /F /PID %%a
)
echo [SYSTEM] Service stopped.
pause
