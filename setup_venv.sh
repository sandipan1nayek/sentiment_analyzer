#!/bin/bash
echo "====================================================="
echo "  Sentiment Analysis Pipeline - Virtual Environment Setup"
echo "====================================================="
echo

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment!"
    exit 1
fi

echo
echo "Activating virtual environment..."
source venv/bin/activate

echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements!"
    exit 1
fi

echo
echo "Downloading SpaCy model..."
python -m spacy download en_core_web_sm
if [ $? -ne 0 ]; then
    echo "Warning: SpaCy model download failed."
    echo "Entity recognition may not work properly."
fi

echo
echo "Testing installation..."
python test_setup.py

echo
echo "====================================================="
echo "  Setup completed in virtual environment!"
echo "====================================================="
echo
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo
echo "To run the application:"
echo "  streamlit run main.py"
echo
echo "Don't forget to:"
echo "1. Get your NewsAPI key from https://newsapi.org/"
echo "2. Set environment variable: export NEWSAPI_KEY=your_key_here"
echo
