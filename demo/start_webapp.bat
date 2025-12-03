@echo off
echo ============================================================
echo Stock Sentiment Analysis - Web Application
echo ============================================================
echo.
echo Starting the web server...
echo.

cd /d "%~dp0"
python app.py

pause
