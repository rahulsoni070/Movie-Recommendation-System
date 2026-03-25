@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   Movie Recommendation System - Windows Setup
echo ============================================================
echo.

:: ----------------------------------------------------------------
:: 1. Find Python (prefer the 'py' launcher, fall back to 'python')
:: ----------------------------------------------------------------
set PYTHON_CMD=

py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    echo [OK] Found Python launcher (py)
    goto :CHECK_VERSION
)

python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    echo [OK] Found Python
    goto :CHECK_VERSION
)

echo [ERROR] Python not found.
echo.
echo Please install Python 3.10 or higher from https://www.python.org/downloads/
echo IMPORTANT: On the installer screen, check "Add Python to PATH".
echo.
echo If you installed Python from the Microsoft Store, please uninstall it
echo and reinstall from python.org instead.
pause
exit /b 1

:CHECK_VERSION
for /f "tokens=2" %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VERSION=%%v
echo [INFO] Python version: %PY_VERSION%

:: ----------------------------------------------------------------
:: 2. Remove any leftover broken venv and create a fresh one
:: ----------------------------------------------------------------
if exist venv (
    echo [INFO] Removing existing venv...
    rmdir /s /q venv
)

echo [INFO] Creating virtual environment...
%PYTHON_CMD% -m venv venv

if not exist venv\Scripts\activate.bat (
    echo.
    echo [ERROR] Virtual environment creation failed.
    echo.
    echo This is a known issue when Python is installed from the Microsoft Store.
    echo.
    echo FIX: Disable the Microsoft Store Python aliases:
    echo   1. Open Windows Settings (Win + I)
    echo   2. Go to: Apps ^> Advanced app settings ^> App execution aliases
    echo   3. Turn OFF both "python.exe" and "python3.exe" toggles
    echo   4. Re-run this script
    echo.
    echo Alternatively, install Python from https://www.python.org/downloads/
    echo and make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Virtual environment created.

:: ----------------------------------------------------------------
:: 3. Activate venv and install dependencies
:: ----------------------------------------------------------------
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [INFO] Installing dependencies (this may take a minute)...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

:: ----------------------------------------------------------------
:: 4. Generate sample movie model data
:: ----------------------------------------------------------------
echo [INFO] Generating sample movie model...
python manage.py generate_sample_data
if %errorlevel% neq 0 (
    echo [ERROR] Failed to generate sample data.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------
:: 5. Run database migrations
:: ----------------------------------------------------------------
echo [INFO] Running database migrations...
python manage.py migrate --run-syncdb
if %errorlevel% neq 0 (
    echo [ERROR] Failed to run migrations.
    pause
    exit /b 1
)

:: ----------------------------------------------------------------
:: 6. Done
:: ----------------------------------------------------------------
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo To start the application, run:
echo.
echo     venv\Scripts\activate.bat
echo     python manage.py runserver
echo.
echo Then open http://127.0.0.1:8000 in your browser.
echo.

set /p START_NOW="Start the server now? [Y/n]: "
if /i "!START_NOW!" == "n" goto :END

echo.
echo Starting server... Press Ctrl+C to stop.
echo Open http://127.0.0.1:8000 in your browser.
echo.
python manage.py runserver

:END
endlocal
