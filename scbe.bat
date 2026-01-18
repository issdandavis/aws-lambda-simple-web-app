@echo off
REM SCBE-AETHERMOORE CLI Launcher (Windows)
REM Usage: scbe.bat

cd /d "%~dp0"

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python scbe-cli.py %*
