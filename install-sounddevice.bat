@echo off
setlocal
set "PYTHON=C:\Users\User\ultimate-rvc\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python executable not found:
    echo   %PYTHON%
    echo Check your Ultimate RVC installation path.
    pause
    exit /b 1
)

echo Bootstrapping pip in the Ultimate RVC environment...
"%PYTHON%" -m ensurepip --upgrade
if errorlevel 1 (
    echo.
    echo ERROR: ensurepip could not install pip.
    echo If this Python build does not include ensurepip, install pip using:
    echo   "%PYTHON%" get-pip.py
    echo Download get-pip.py from https://bootstrap.pypa.io/get-pip.py
    pause
    exit /b 1
)

echo.
echo Installing sounddevice...
"%PYTHON%" -m pip install sounddevice
if errorlevel 1 (
    echo.
    echo ERROR: sounddevice installation failed.
    pause
    exit /b 1
)

echo.
echo sounddevice installed successfully.
pause
exit /b 0
