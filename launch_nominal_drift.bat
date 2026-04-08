@echo off
echo ========================================
echo  Nominal Drift - Local AI Workbench
echo ========================================
cd /d %~dp0
python -m streamlit run nominal_drift/gui/app.py
pause
