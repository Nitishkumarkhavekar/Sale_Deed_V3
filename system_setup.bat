@echo off
REM ===================================================================
REM  Sale Deed AI - System Setup (single entry point)
REM
REM  Forwards to "System Setup.bat", which is the same installer under the
REM  name the desktop shortcut and the documentation already use. Both names
REM  work and both do exactly the same thing; this one exists because
REM  "system_setup.bat" is what a script or a runbook would reasonably type,
REM  and a name with a space in it is awkward to pass around.
REM
REM  All arguments are forwarded, and the exit code is preserved, so this is
REM  usable in an automated install:
REM
REM    system_setup.bat                 everything, then launch
REM    system_setup.bat --report-only   detect and report, change nothing
REM    system_setup.bat --no-launch     set up but do not start
REM    system_setup.bat --skip-tests    skip the test suite
REM ===================================================================

setlocal

REM  %~dp0 ends with a backslash and is quoted throughout, so the project may
REM  live on any drive and in any folder, including one with spaces.
if not exist "%~dp0System Setup.bat" (
    echo.
    echo   ERROR: "System Setup.bat" was not found next to this file.
    echo.
    echo   Both files must stay in the project root:
    echo       %~dp0
    echo.
    echo   Re-copy the project rather than moving individual files.
    echo.
    pause
    exit /b 1
)

call "%~dp0System Setup.bat" %*
endlocal & exit /b %ERRORLEVEL%
