@echo off
REM ==================================================================
REM  install_shim.bat  --  DAUERHAFTE Loesung fuer
REM                        "module 'numpy' has no attribute 'ulong'"
REM
REM  Installiert einen Startup-Shim in die site-packages der Umgebung.
REM  Dadurch existiert np.ulong BEVOR scipy importiert wird.
REM  NumPy selbst bleibt unveraendert -> torch bleibt kompatibel.
REM
REM  Aufruf:  install_shim.bat
REM           install_shim.bat "C:\...\ApplioV3.7.1\env"
REM           install_shim.bat --remove
REM ==================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "SI_APPLIO=C:\Users\User\Downloads\ApplioV3.7.1"
set "SI_APP=C:\Users\User\Downloads\Super Impersonator app by don24crk"
set "SI_ENGINE=%SI_APP%\Engine"

if /I "%~1"=="--remove" (
    for %%E in (
      "%SI_APPLIO%\env" "%SI_APPLIO%\.venv" "%SI_APPLIO%\venv"
      "%SI_ENGINE%\env" "%SI_ENGINE%\.venv" "%SI_APP%\.venv"
    ) do (
        if exist "%%~E\Scripts\python.exe" (
            echo -- entferne Shim aus %%~E
            "%%~E\Scripts\python.exe" "install_shim.py" --env "%%~E" --remove
        )
    )
    pause & exit /b 0
)

if not "%~1"=="" (
    echo === Shim installieren in %~1 ===
    "%~1\Scripts\python.exe" "install_shim.py" --env "%~1"
    pause & exit /b %ERRORLEVEL%
)

echo ============================================================
echo   Installiere NumPy-Shim in alle Umgebungen
echo ============================================================
for %%E in (
  "%SI_APPLIO%\env" "%SI_APPLIO%\.venv" "%SI_APPLIO%\venv"
  "%SI_ENGINE%\env" "%SI_ENGINE%\.venv" "%SI_APP%\.venv"
) do (
    if exist "%%~E\Scripts\python.exe" (
        echo.
        echo ------------------------------------------------------------
        echo ENV: %%~E
        "%%~E\Scripts\python.exe" "install_shim.py" --env "%%~E"
    )
)

echo.
echo ============================================================
echo   Danach:  start_app.bat
echo ============================================================
pause
