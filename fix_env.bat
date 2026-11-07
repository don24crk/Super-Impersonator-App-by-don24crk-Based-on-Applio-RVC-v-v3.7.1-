@echo off
REM ==================================================================
REM  fix_env.bat  --  repariert die NumPy/SciPy-Passung
REM
REM  Behebt:  module 'numpy' has no attribute 'ulong'
REM           Could not import the Applio/RVC voice converter class
REM
REM  Standard : SciPy auf 1.11.4 zurueckstufen (NumPy bleibt 1.26.4)
REM  --numpy2 : stattdessen NumPy 2.2.6 + SciPy 1.15.1 installieren
REM
REM  Aufruf:  fix_env.bat
REM           fix_env.bat --numpy2
REM           fix_env.bat "C:\...\ApplioV3.7.1\env"
REM           fix_env.bat "C:\...\env" --numpy2
REM ==================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "SI_APPLIO=C:\Users\User\Downloads\ApplioV3.7.1"
set "SI_APP=C:\Users\User\Downloads\Super Impersonator app by don24crk"
set "SI_ENGINE=%SI_APP%\Engine"

set "MODE=numpy1"
set "TARGET="
for %%A in (%*) do (
    if /I "%%~A"=="--numpy2" ( set "MODE=numpy2" ) else ( set "TARGET=%%~A" )
)

echo ============================================================
echo   Repariere NumPy/SciPy   (Modus: %MODE%)
echo ============================================================

if not "%TARGET%"=="" (
    call :fixone "%TARGET%"
    goto :verify
)

for %%E in (
  "%SI_APPLIO%\env"
  "%SI_APPLIO%\.venv"
  "%SI_APPLIO%\venv"
  "%SI_ENGINE%\env"
  "%SI_ENGINE%\.venv"
  "%SI_APP%\.venv"
) do (
    if exist "%%~E\Scripts\python.exe" call :fixone "%%~E"
)

:verify
echo.
echo ============================================================
echo   Verifikation: echter Import-Test
echo ============================================================
call "%~dp0_select_env.bat"
if errorlevel 1 (
    echo.
    echo [X] Immer noch keine funktionierende Umgebung.
    echo.
    echo     Wenn scipy.sparse jetzt geht, aber torch/rvc fehlt:
    echo       - torch nachinstallieren:
    echo         "%%E\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo     Anderer Versuch:
    echo       fix_env.bat --numpy2
    pause
    exit /b 2
)

echo.
echo ============================================================
echo   FERTIG. Funktionierende Umgebung:
echo     %SI_ENVDIR%
echo   Starte die App jetzt mit:  start_app.bat
echo ============================================================
pause
exit /b 0

:fixone
set "E=%~1"
if not exist "%E%\Scripts\python.exe" exit /b 0
echo.
echo ------------------------------------------------------------
echo ENV: %E%
"%E%\Scripts\python.exe" -c "import numpy,scipy;print('  vorher : numpy',numpy.__version__,'| scipy',scipy.__version__)" 2>nul

if /I "%MODE%"=="numpy2" (
    "%E%\Scripts\python.exe" -m pip install --quiet --no-cache-dir "numpy==2.2.6" "scipy==1.15.1"
) else (
    "%E%\Scripts\python.exe" -m pip install --quiet --no-cache-dir "scipy==1.11.4"
)

"%E%\Scripts\python.exe" -c "import numpy,scipy,scipy.sparse;print('  nachher: numpy',numpy.__version__,'| scipy',scipy.__version__,'| scipy.sparse OK')" 2>nul
if errorlevel 1 (
    echo   [!] scipy.sparse weiterhin defekt - versuche NumPy 2.x
    "%E%\Scripts\python.exe" -m pip install --quiet --no-cache-dir "numpy==2.2.6" "scipy==1.15.1"
    "%E%\Scripts\python.exe" -c "import numpy,scipy,scipy.sparse;print('  nachher: numpy',numpy.__version__,'| scipy',scipy.__version__,'| scipy.sparse OK')" 2>nul
)
exit /b 0
