@echo off
REM Verified with Python 3.13.5; Python 3.13 is recommended.
setlocal
cd /d "%~dp0"
python src\run_analysis.py
if errorlevel 1 exit /b %errorlevel%
endlocal
