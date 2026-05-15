#!/usr/bin/env bash
# Movie Recommendation System - Linux/macOS Setup Script
set -e

echo
echo "============================================================"
echo "  Movie Recommendation System - Setup"
echo "============================================================"
echo

# ----------------------------------------------------------------
# 1. Find Python 3.10+
# ----------------------------------------------------------------
PYTHON_CMD=""

for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)')
        if [ "$version" -ge 310 ] 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10 or higher not found."
    echo
    echo "Install Python 3.10+:"
    echo "  macOS:  brew install python3"
    echo "  Ubuntu: sudo apt install python3"
    echo "  Others: https://www.python.org/downloads/"
    exit 1
fi

echo "[OK] Using: $($PYTHON_CMD --version)"

# ----------------------------------------------------------------
# 2. Create virtual environment
# ----------------------------------------------------------------
if [ -d venv ]; then
    echo "[INFO] Removing existing venv..."
    rm -rf venv
fi

echo "[INFO] Creating virtual environment..."
$PYTHON_CMD -m venv venv
echo "[OK] Virtual environment created."

# ----------------------------------------------------------------
# 3. Activate and install dependencies
# ----------------------------------------------------------------
echo "[INFO] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[INFO] Installing dependencies (this may take a minute)..."
pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed."

# ----------------------------------------------------------------
# 4. Generate sample movie model data
# ----------------------------------------------------------------
echo "[INFO] Generating sample movie model..."
python manage.py generate_sample_data

# ----------------------------------------------------------------
# 5. Run database migrations
# ----------------------------------------------------------------
echo "[INFO] Running database migrations..."
python manage.py migrate --run-syncdb

# ----------------------------------------------------------------
# 6. Done
# ----------------------------------------------------------------
echo
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo
echo "To start the application next time:"
echo
echo "    source venv/bin/activate"
echo "    python manage.py runserver"
echo
echo "Then open http://127.0.0.1:8000 in your browser."
echo

read -r -p "Start the server now? [Y/n]: " START_NOW
if [[ "${START_NOW,,}" != "n" ]]; then
    echo
    echo "Starting server... Press Ctrl+C to stop."
    echo "Open http://127.0.0.1:8000 in your browser."
    echo
    python manage.py runserver
fi
