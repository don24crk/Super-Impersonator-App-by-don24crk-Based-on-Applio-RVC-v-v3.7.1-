@echo off
REM ==================================================================
REM  _select_env.bat  --  helper, called by start_app.bat
REM
REM  Sets  SI_PYTHON  /  SI_ENVDIR  to the first interpreter that
REM  passes the real import probe (numpy + scipy.sparse + torch + rvc).
REM  Exits 1 if none pass.
REM ==================================================================
setlocal EnableExtensions EnableDelayedExpansion

set "SI_APPLIO=C:\Users\User\Downloads\ApplioV3.7.1"
set "SI_APP=C:\Users\User\Downloads\Super Impersonator app by don24crk"
set "SI_ENGINE=%SI_APP%\Engine"

echo === Probiere Umgebungen (echter Import-Test) ===

for %%E in (
  "%SI_APPLIO%\env"
  "%SI_APPLIO%\.venv"
  "%SI_APPLIO%\venv"
  "%SI_ENGINE%\env"
  "%SI_ENGINE%\.venv"
  "%SI_APP%\.venv"
) do (
    if exist "%%~E\Scripts\python.exe" (
        echo.
        echo -- %%~E
        "%%~E\Scripts\python.exe" "%~dp0_probe_env.py" --applio "%SI_APPLIO%" --engine "%SI_ENGINE%"
        if !errorlevel! equ 0 (
            endlocal & set "SI_PYTHON=%%~E\Scripts\python.exe" & set "SI_ENVDIR=%%~E" & exit /b 0
        )
    )
)

echo.
echo [X] Keine Umgebung hat den Import-Test bestanden.
exit /b 1
