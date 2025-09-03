@echo off
echo =====================================================
echo   Starting Sentiment Analysis Pipeline
echo =====================================================
echo.

echo Checking if virtual environment exists...
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run setup_venv.bat first to create the environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Checking NewsAPI key...
if "%NEWSAPI_KEY%"=="" (
    echo Warning: NEWSAPI_KEY environment variable not set!
    echo You can still run the app, but news fetching won't work.
    echo Get your free API key from https://newsapi.org/
    echo Then set it with: set NEWSAPI_KEY=your_key_here
    echo.
)

echo Starting Streamlit application...
echo Open your browser and go to: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run main.py

pause
