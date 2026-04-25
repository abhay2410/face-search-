@echo off
title Face Search AI Dashboard
color 0B
cls
echo [SYSTEM] Launching Face Search AI (Standalone)...
:: Priority: Use local venv for standalone mode.
powershell -ExecutionPolicy Bypass -Command "$baseDir = '%~dp0'; Set-Location $baseDir; $python = Join-Path $baseDir 'venv\Scripts\python.exe'; if (-not (Test-Path $python)) { Write-Host '[ERROR] No local virtual environment found!' -FG Red; pause; exit }; Write-Host \"[SYSTEM] Using Local Python: $python\"; Start-Process 'http://127.0.0.1:8001'; & $python app.py; if ($LASTEXITCODE -ne 0) { Write-Host 'The system encountered an error and stopped.' -FG Red; pause }"
pause
