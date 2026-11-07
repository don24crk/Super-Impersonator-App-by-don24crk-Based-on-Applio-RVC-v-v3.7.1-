@echo off
title Super Impersonator App v3.0 - Bootloader
echo ========================================================
echo   Super Impersonator App v3.0 - MATRIX CONFIG INIT
echo ========================================================

:: Python 3.12 Pfade definieren und priorisieren
set "PYTHON_DIR=C:\Users\User\AppData\Local\Programs\Python\Python312"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"

:: FFmpeg & FFprobe Pfad global registrieren
set "FFMPEG_BIN_DIR=C:\Users\User\Downloads\Super Impersonator app by don24crk\Engine"
set "PATH=%FFMPEG_BIN_DIR%;%PATH%"

:: Hardware- & Software-Checks vorab
if exist "%PYTHON_DIR%\python.exe" (
    echo [OK] Python 3.12 erfolgreich geladen.
) else (
    echo [FEHLER] Python nicht gefunden unter: %PYTHON_DIR%
    pause
    exit
)

if exist "%FFMPEG_BIN_DIR%\ffmpeg.exe" (
    echo [OK] ffmpeg.exe im Engine-Verzeichnis gefunden.
) else (
    echo [WARNUNG] ffmpeg.exe wurde unter %FFMPEG_BIN_DIR% nicht gefunden!
)

if exist "%FFMPEG_BIN_DIR%\ffprobe.exe" (
    echo [OK] ffprobe.exe im Engine-Verzeichnis gefunden.
) else (
    echo [WARNUNG] ffprobe.exe wurde unter %FFMPEG_BIN_DIR% nicht gefunden!
)

:: Wechsel in das Hauptverzeichnis des Skripts
cd /d "%~dp0"

:: Übergibt die Kontrolle an das Start-Skript
call launch-web.bat

pause
