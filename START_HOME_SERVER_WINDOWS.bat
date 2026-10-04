@echo off
setlocal
cd /d "%~dp0"
set GEOAI_HOST=0.0.0.0
set GEOAI_PORT=7871
python app.py
pause
