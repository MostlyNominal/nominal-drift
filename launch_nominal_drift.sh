#!/bin/bash
cd "$(dirname "$0")"
python -m streamlit run nominal_drift/gui/app.py
