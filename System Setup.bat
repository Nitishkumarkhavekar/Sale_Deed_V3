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
REM  Engines: llama.cpp serves the quantised model on any card and is what
REM  runs on a laptop. On a machine with 16 GB or more of VRAM the setup also
REM  installs vLLM - into its own virtualenv, because it pins a transformers
REM  version Surya cannot use - and the server then selects it automatically.
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

REM -------------------------------------------------------------------
REM  No Python at all: install it, rather than telling the operator to.
REM
REM  This is the one step that cannot live in system_setup.py, because that
REM  script is Python. On a genuinely clean Windows machine nothing else in
REM  this installer can run until this succeeds.
REM
REM  winget is present on Windows 10 1809+ and Windows 11. Where it is absent
REM  - an old build, or a machine with the Store stripped - there is no
REM  scriptable install path, so the manual instruction is still printed.
REM -------------------------------------------------------------------

if not defined PYEXE (
    echo   Python was not found. Installing Python 3.13 ...
    echo.

    where winget >nul 2>&1
    if errorlevel 1 (
        echo   winget is not available on this machine, so Python cannot be
        echo   installed automatically.
        echo.
        echo   Install Python 3.13 by hand, then run this file again:
        echo       https://www.python.org/downloads/
        echo.
        echo   Tick "Add python.exe to PATH" if the installer offers it.
        echo.
        pause
        exit /b 1
    )

    REM --scope machine needs administrator; fall back to a per-user install,
    REM which needs none and is enough for everything this application does.
    winget install --id Python.Python.3.13 --scope machine ^
        --accept-package-agreements --accept-source-agreements --silent
    if errorlevel 1 (
        echo   Machine-wide install did not succeed; trying a per-user install.
        winget install --id Python.Python.3.13 ^
            --accept-package-agreements --accept-source-agreements --silent
    )

    REM  Re-probe through `py`, the launcher Windows installs into System32.
    REM  It is on PATH immediately, whereas python.exe usually is not until a
    REM  new shell picks up the changed PATH - so probing `python` here would
    REM  report failure on a perfectly good install.
    for %%V in (3.13 3.12 3.14) do (
        if not defined PYEXE (
            py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
        )
    )
    if not defined PYEXE (
        python -c "import sys" >nul 2>&1 && set "PYEXE=python"
    )

    if not defined PYEXE (
        echo.
        echo   Python was installed but is not visible to this window yet.
        echo   That is normal - PATH changes reach a shell only when it starts.
        echo.
        echo   Close this window and run "System Setup.bat" again. It will
        echo   pick up from here; nothing already done is repeated.
        echo.
        pause
        exit /b 1
    )
    echo   Python installed: %PYEXE%
    echo.
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
