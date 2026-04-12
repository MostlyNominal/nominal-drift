#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Find Python 3 interpreter
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: Python not found."
    echo "Install Python 3.11+ from https://www.python.org/downloads/"
    exit 1
fi

echo "Using: $($PY --version)"

# Auto-install if not yet installed
$PY -c "import streamlit" 2>/dev/null || {
    echo "[INFO] Installing dependencies..."
    $PY -m pip install -e .
}

echo "[INFO] Launching Nominal Drift GUI..."
$PY -m streamlit run nominal_drift/gui/app.py --server.headless true
