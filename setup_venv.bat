@echo off
echo =====================================================
echo   Sentiment Analysis Pipeline - Virtual Environment Setup
echo =====================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment!
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements!
    pause
    exit /b 1
)

echo.
echo Downloading SpaCy model...
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo Warning: SpaCy model download failed.
    echo Entity recognition may not work properly.
)

echo.
echo Testing installation...
python test_setup.py

echo.
echo =====================================================
echo   Setup completed in virtual environment!
echo =====================================================
echo.
echo To activate the virtual environment in future sessions:
echo   venv\Scripts\activate
echo.
echo To run the application:
echo   streamlit run main.py
echo.
echo Don't forget to:
echo 1. Get your NewsAPI key from https://newsapi.org/
echo 2. Set environment variable: set NEWSAPI_KEY=your_key_here
echo.
pause
