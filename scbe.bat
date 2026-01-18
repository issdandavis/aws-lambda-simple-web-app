@echo off
REM SCBE-AETHERMOORE Unified Launcher (Windows)
REM
REM Usage:
REM   scbe.bat           - Interactive CLI (default)
REM   scbe.bat cli       - Interactive CLI with tutorial
REM   scbe.bat agent     - Polly AI Agent
REM   scbe.bat demo      - Run demo
REM   scbe.bat memory    - AI Memory Shard demo
REM   scbe.bat api       - Start REST API server

cd /d "%~dp0"

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Parse command (default to cli)
set MODE=%1
if "%MODE%"=="" set MODE=cli

if "%MODE%"=="cli" goto cli
if "%MODE%"=="tutorial" goto cli
if "%MODE%"=="agent" goto agent
if "%MODE%"=="polly" goto agent
if "%MODE%"=="demo" goto demo
if "%MODE%"=="memory" goto memory
if "%MODE%"=="shard" goto memory
if "%MODE%"=="integrated" goto integrated
if "%MODE%"=="api" goto api
if "%MODE%"=="server" goto api
if "%MODE%"=="web" goto web
if "%MODE%"=="help" goto help
if "%MODE%"=="--help" goto help
if "%MODE%"=="-h" goto help

echo Unknown command: %MODE%
echo Run 'scbe.bat help' for usage
exit /b 1

:cli
python scbe-cli.py %2 %3 %4 %5
goto end

:agent
python scbe-agent.py %2 %3 %4 %5
goto end

:demo
python demo.py %2 %3 %4 %5
goto end

:memory
python demo_memory_shard.py %2 %3 %4 %5
goto end

:integrated
python demo_integrated_memory_shard.py %2 %3 %4 %5
goto end

:api
python -m scbe_production.api %2 %3 %4 %5
goto end

:web
echo Opening web demo...
start "" "web\index.html"
goto end

:help
echo SCBE-AETHERMOORE Launcher
echo.
echo Usage: scbe.bat [command]
echo.
echo Commands:
echo   cli, tutorial    Interactive CLI with tutorial (default)
echo   agent, polly     Polly AI Agent
echo   demo             Run basic demo
echo   memory, shard    AI Memory Shard demo
echo   integrated       Full integrated demo
echo   api, server      Start REST API server
echo   web              Open web demo in browser
echo   help             Show this help
echo.
goto end

:end
