@echo off
REM ===================================================================
REM  Sale Deed AI - double-click to start.
REM
REM  Finds the project's Python (virtual environment first, then the
REM  system installation) and hands over to launcher.py, which does
REM  everything else.
REM ===================================================================

setlocal
title Sale Deed AI

REM Work from this file's own folder, so a desktop shortcut works.
cd /d "%~dp0"

set "PYEXE="
set "PYANY="

REM Interpreters are selected by CAPABILITY, not by version number. More than
REM one Python is normally installed - this project's Surya OCR component needs
REM its own 3.12, and installing it changes what bare "python" resolves to.
REM Picking the newest, or the first found, silently launches an interpreter
REM without PySide6 and the failure reads as a broken application.

REM 1. Project virtual environment, under any of the usual names.
for %%V in (".venv" "venv" "env") do (
    if not defined PYEXE if exist "%%~V\Scripts\python.exe" (
        "%%~V\Scripts\python.exe" -c "import PySide6" >nul 2>&1 && set "PYEXE=%%~V\Scripts\python.exe"
    )
)

REM 2. Any registered interpreter that can actually run the application.
REM    Newest first, since that is the more likely home of the dependencies.
if not defined PYEXE (
    for %%V in (3.14 3.13 3.12) do (
        if not defined PYEXE (
            py -%%V -c "import sys" >nul 2>&1 && set "PYANY=py -%%V"
            py -%%V -c "import PySide6" >nul 2>&1 && set "PYEXE=py -%%V"
        )
    )
)

REM 3. Whatever "python" means on PATH, again checked before use.
if not defined PYEXE (
    python -c "import sys" >nul 2>&1 && set "PYANY=python"
    python -c "import PySide6" >nul 2>&1 && set "PYEXE=python"
)

REM A Python with no PySide6 is still better than none: launcher.py reports
REM exactly which packages are missing and the command that installs them.
if not defined PYEXE if defined PYANY (
    echo.
    echo   No Python with the application's packages was found.
    echo   Continuing with %PYANY% so the checks can tell you what is missing.
    echo.
    set "PYEXE=%PYANY%"
)

if not defined PYEXE (
    echo.
    echo   Python was not found.
    echo.
    echo   Install Python 3.12 or newer, then run this file again:
    echo       winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

%PYEXE% "launcher.py" %*
set "RC=%ERRORLEVEL%"

REM Only hold the window open on failure - a clean exit should just close.
if not "%RC%"=="0" (
    echo.
    echo   Sale Deed AI exited with code %RC%.
    echo   The startup log is in:  runtime\logs\launcher.log
    echo.
    pause
)

endlocal & exit /b %RC%
