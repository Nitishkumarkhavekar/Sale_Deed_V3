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
REM  Find an interpreter to BOOTSTRAP with.
REM
REM  Only to build the virtual environment. The application's own packages go
REM  into ".venv" a few lines below and are never installed system-wide, so
REM  this interpreter needs nothing but a working "venv" module.
REM
REM  It used to be chosen by whether it already had PySide6, which was the
REM  right question when packages lived in the system Python and is the wrong
REM  one now: on a clean machine no interpreter has PySide6, and on this one
REM  the answer would point at the very installation the venv exists to stop
REM  depending on.
REM
REM  Version order still matters. This project deliberately needs two Pythons -
REM  3.13 for the application and 3.12 for OCR - and installing the second
REM  changes what a bare "python" resolves to. The list is in preference order
REM  and the guard is what makes that order mean anything: without it every
REM  iteration overwrites the previous, and a box with 3.12, 3.13 and 3.14
REM  ends up building the venv from 3.14, which this project has never been
REM  tested on.
REM -------------------------------------------------------------------

set "PYEXE="
set "PYANY="

for %%V in (3.13 3.12 3.14) do (
    if not defined PYANY (
        py -%%V -c "import venv" >nul 2>&1 && set "PYANY=py -%%V"
    )
)

if not defined PYANY (
    python -c "import venv" >nul 2>&1 && set "PYANY=python"
)

set "PYEXE=%PYANY%"

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

echo   Bootstrap interpreter: %PYEXE%
echo.

REM -------------------------------------------------------------------
REM  The application's virtual environment.
REM
REM  Everything the application needs lives here and nowhere else, so a
REM  machine's system Python is left exactly as it was found and two projects
REM  on one machine cannot pull each other's package versions around.
REM
REM  Built from this file's own folder - the working directory was set from
REM  %~dp0 at the top - so the path is whatever the project was copied to. No
REM  drive letter is written down anywhere.
REM
REM  This is the application's environment only. OCR keeps its own under
REM  models\SuryaOCR, because Surya pins a transformers version the rest of
REM  the project cannot use, and vLLM keeps a third for the same reason. Those
REM  are built by system_setup.py and are deliberately not merged into this
REM  one - merging them is what makes an OCR upgrade break extraction.
REM -------------------------------------------------------------------

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo   Creating the application environment in .venv ...
    %PYEXE% -m venv ".venv"
    if errorlevel 1 (
        echo.
        echo   The virtual environment could not be created.
        echo.
        echo   Most often this is a Python installed without the "venv"
        echo   module, or no permission to write inside:
        echo       %CD%
        echo.
        echo   Try running this file from a folder you own, or reinstall
        echo   Python from https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    echo   Created: .venv
    echo.
)

REM  Present is not the same as working: a half-finished creation, or a folder
REM  copied from another machine, leaves a .venv whose python.exe cannot run.
REM  Checked before use so the failure names the environment rather than
REM  surfacing later as a missing package.
"%VENV_PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   .venv exists but its Python does not run. It is probably damaged,
    echo   or was copied from another machine - a virtual environment records
    echo   absolute paths and cannot be moved.
    echo.
    echo   Delete the .venv folder and run this file again.
    echo.
    pause
    exit /b 1
)

echo   Using: %VENV_PY%
echo.

REM  Run the setup INSIDE the environment. Every install, migration and check
REM  it performs uses sys.executable, so this one line is what puts the whole
REM  installation into .venv rather than into the machine's Python.
"%VENV_PY%" "src\tools\system_setup.py" %*
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
