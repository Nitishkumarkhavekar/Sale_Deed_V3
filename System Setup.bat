@echo off
REM ===================================================================
REM  Sale Deed AI - System Setup
REM
REM  Double-click this file on a new Windows machine. It detects what is
REM  already installed, installs only what is missing, configures the
REM  application, validates every component, and starts it.
REM
REM  Safe to run again: completed steps are skipped and nothing that was
REM  already on the machine is removed or reinstalled.
REM
REM    "System Setup.bat"                everything, then launch
REM    "System Setup.bat" --report-only  detect and report, change nothing
REM    "System Setup.bat" --no-launch    set up but do not start
REM    "System Setup.bat" --skip-tests   skip the test suite
REM
REM  This file stays a shim on purpose. All logic lives in
REM  src/tools/system_setup.py, because batch is unreadable and untestable at
REM  any size worth writing.
REM ===================================================================

setlocal EnableDelayedExpansion
title Sale Deed AI - System Setup

REM Work from this file's own folder, so a desktop shortcut works.
cd /d "%~dp0"

echo.
echo   Sale Deed AI - System Setup
echo   Preparing this machine. Nothing already installed will be replaced.
echo.

REM -------------------------------------------------------------------
REM  Find an interpreter.
REM
REM  Chosen by CAPABILITY, not by version number. This project deliberately
REM  needs two Pythons - 3.13 for the application and 3.12 for OCR - and
REM  installing the second changes what a bare "python" resolves to. Picking
REM  the newest, or the first found, silently selects an interpreter without
REM  the application's packages, and the failure reads as a broken program
REM  rather than a wrong interpreter.
REM
REM  On a clean machine nothing has PySide6 yet, so any working Python is
REM  accepted and the setup script installs the packages into it.
REM -------------------------------------------------------------------

set "PYEXE="
set "PYANY="

REM  PYANY keeps the FIRST working interpreter, not the last. Without the
REM  guard every iteration overwrote it, so on a clean machine - where none of
REM  them has PySide6 yet - the fallback ended up as whichever version was
REM  probed last. On a box with 3.12, 3.13 and 3.14 installed that selected
REM  3.14, and the packages were installed into an interpreter this project
REM  has never been tested on. The list below is in preference order and the
REM  guard is what makes that order mean anything.
for %%V in (3.13 3.12 3.14) do (
    if not defined PYEXE (
        if not defined PYANY (
            py -%%V -c "import sys" >nul 2>&1 && set "PYANY=py -%%V"
        )
        py -%%V -c "import PySide6" >nul 2>&1 && set "PYEXE=py -%%V"
    )
)

if not defined PYEXE (
    if not defined PYANY (
        python -c "import sys" >nul 2>&1 && set "PYANY=python"
    )
    python -c "import PySide6" >nul 2>&1 && set "PYEXE=python"
)

REM No configured interpreter yet is the normal first-run state.
if not defined PYEXE if defined PYANY set "PYEXE=%PYANY%"

if not defined PYEXE (
    echo   Python was not found on this machine.
    echo.
    echo   Install Python 3.13, then run this file again:
    echo.
    echo       winget install Python.Python.3.13
    echo.
    echo   Tick "Add python.exe to PATH" if the installer offers it.
    echo.
    pause
    exit /b 1
)

echo   Using: %PYEXE%
echo.

%PYEXE% "src\tools\system_setup.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo   Setup stopped with code %RC%.
    echo.
    echo   Read:  docs\INSTALLATION_REPORT.md
    echo   Logs:  runtime\logs\setup\installation.log
    echo.
    echo   Fix the problem above and run this file again - completed steps
    echo   are skipped, so it will not redo work.
    echo.
    pause
)

endlocal & exit /b %RC%
