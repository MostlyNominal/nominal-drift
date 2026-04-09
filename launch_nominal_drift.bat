@echo off
echo ========================================
echo  Nominal Drift - Local AI Workbench
echo ========================================
echo.

cd /d %~dp0

REM Try py launcher first (standard Windows Python launcher)
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found Python via 'py' launcher
    py -3 -m pip show streamlit >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [INFO] Installing dependencies...
        py -3 -m pip install -e . 2>&1
    )
    echo [INFO] Launching Nominal Drift GUI...
    py -3 -m streamlit run nominal_drift/gui/app.py --server.headless true
    goto :end
)

REM Try python on PATH
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Verify it is real Python, not the Windows Store alias
    python -c "import sys; sys.exit(0)" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Found Python on PATH
        python -m pip show streamlit >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo [INFO] Installing dependencies...
            python -m pip install -e . 2>&1
        )
        echo [INFO] Launching Nominal Drift GUI...
        python -m streamlit run nominal_drift/gui/app.py --server.headless true
        goto :end
    )
)

REM Try python3 on PATH (Git Bash / MSYS2)
where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found python3 on PATH
    python3 -m streamlit run nominal_drift/gui/app.py --server.headless true
    goto :end
)

echo.
echo ========================================
echo  ERROR: Python not found!
echo ========================================
echo.
echo  Nominal Drift requires Python 3.11 or newer.
echo.
echo  Install Python from:
echo    https://www.python.org/downloads/
echo.
echo  IMPORTANT: During installation, check the box:
echo    [x] Add Python to PATH
echo.
echo  After installing Python, re-run this launcher.
echo.

:end
pause
