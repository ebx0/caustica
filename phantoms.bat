@echo off
rem ---------------------------------------------------------------------------
rem  caustica phantom launcher - double-click me, or run:  phantoms.bat [action]
rem  Actions: gui | build | dataset | catalog | tissues | info | fetch
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

rem Prefer the project venv; fall back to whatever python is on PATH.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

rem Was this double-clicked from Explorer? Then keep the window open at the end.
set "KEEPOPEN="
echo %cmdcmdline% | find /i "%~nx0" >nul 2>nul && set "KEEPOPEN=1"

"%PY%" -m apps.phantom_launcher %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo   exited with code %RC%
    if "%PY%"=="python" (
        echo   if that was a ModuleNotFoundError: create the venv first,
        echo      python -m venv .venv ^&^& .venv\Scripts\pip install -e .[dev]
    )
    set "KEEPOPEN=1"
)
if defined KEEPOPEN pause
endlocal & exit /b %RC%
