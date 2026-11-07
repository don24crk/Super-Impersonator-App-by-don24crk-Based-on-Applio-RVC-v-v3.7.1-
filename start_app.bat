@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Super Impersonator App v3.0 - Studio Loader

echo =======================================================================
echo Starte Super Impersonator App...
echo Lade Python-Umgebung und initialisiere Hardware-Matrix...
echo =======================================================================

rem Engine-Verzeichnis
set "ENGINE_DIR=C:\Users\User\Downloads\Super Impersonator app by don24crk\Engine"
set "APP_SCRIPT=%ENGINE_DIR%\infer-web.py"

rem Python-Interpreter
set "APPLIO_PYTHON=C:\Users\User\Downloads\ApplioV3.7.1\.venv\Scripts\python.exe"
set "PYTHON312=C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe"

if not exist "%ENGINE_DIR%" (
    echo [FEHLER] Engine-Verzeichnis nicht gefunden:
    echo "%ENGINE_DIR%"
    pause
    exit /b 1
)

cd /d "%ENGINE_DIR%"
if errorlevel 1 (
    echo [FEHLER] Engine-Verzeichnis konnte nicht geoeffnet werden.
    pause
    exit /b 1
)

if not exist "%APP_SCRIPT%" (
    echo [FEHLER] infer-web.py wurde nicht gefunden:
    echo "%APP_SCRIPT%"
    pause
    exit /b 1
)

rem Applio-Umgebung zuerst verwenden, da dort vermutlich die RVC-Abhaengigkeiten liegen.
if exist "%APPLIO_PYTHON%" (
    echo [OK] Applio .venv-Python gefunden. Verwende diese Umgebung...
    "%APPLIO_PYTHON%" "%APP_SCRIPT%"
) else if exist "%PYTHON312%" (
    echo [OK] Python 3.12 gefunden. Verwende:
    echo "%PYTHON312%"
    "%PYTHON312%" "%APP_SCRIPT%"
) else (
    echo [WARNUNG] Applio-Python und Python 3.12 nicht am angegebenen Pfad gefunden.
    echo Versuche Python ueber PATH...
    python "%APP_SCRIPT%"
)

set "APP_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%APP_EXIT_CODE%"=="0" (
    echo [FEHLER] Die App wurde mit Fehlercode %APP_EXIT_CODE% beendet.
    echo Pruefe die Python-Abhaengigkeiten und die oben angezeigte Fehlermeldung.
) else (
    echo Die App wurde beendet.
)

pause
exit /b %APP_EXIT_CODE%
