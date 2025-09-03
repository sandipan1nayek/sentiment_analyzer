@echo off
echo =====================================================
echo   Sentiment Analysis Pipeline - Windows Setup
echo =====================================================
echo.

echo Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing packages!
    pause
    exit /b 1
)

echo.
echo Downloading SpaCy model...
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo Warning: SpaCy model download failed. Entity recognition may not work.
    echo You can try again later with: python -m spacy download en_core_web_sm
)

echo.
echo Testing setup...
python test_setup.py
if errorlevel 1 (
    echo Some tests failed. Please check the output above.
    pause
    exit /b 1
)

echo.
echo =====================================================
echo   Setup completed successfully!
echo =====================================================
echo.
echo Next steps:
echo 1. Get your NewsAPI key from https://newsapi.org/
echo 2. Set environment variable: set NEWSAPI_KEY=your_key_here
echo 3. Run the app: streamlit run main.py
echo.
pause
