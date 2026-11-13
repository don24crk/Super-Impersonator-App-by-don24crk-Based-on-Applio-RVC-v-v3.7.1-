@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Ultimate RVC Core

set "ENGINE_DIR=C:\Users\User\Downloads\Super Impersonator app by don24crk\Engine"
set "PYTHON_DIR=C:\Users\User\ultimate-rvc\.venv\Scripts"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "FFMPEG_BIN_DIR=%ENGINE_DIR%"
set "GRADIO_SERVER_PORT=7860"

if not exist "%PYTHON_EXE%" (
    echo [FEHLER] Python nicht gefunden: "%PYTHON_EXE%"
    pause
    exit /b 1
)

if not exist "%ENGINE_DIR%\infer-web.py" (
    echo [FEHLER] infer-web.py nicht gefunden: "%ENGINE_DIR%\infer-web.py"
    pause
    exit /b 1
)

set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%FFMPEG_BIN_DIR%;%PATH%"

cd /d "%ENGINE_DIR%"
if errorlevel 1 (
    echo [FEHLER] Engine-Verzeichnis konnte nicht geoeffnet werden.
    pause
    exit /b 1
)

echo [INFO] Starte Ultimate RVC Core auf http://127.0.0.1:7860
"%PYTHON_EXE%" "%ENGINE_DIR%\infer-web.py"

set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo [FEHLER] Die App wurde mit Fehlercode %APP_EXIT_CODE% beendet.
)
pause
exit /b %APP_EXIT_CODE%
