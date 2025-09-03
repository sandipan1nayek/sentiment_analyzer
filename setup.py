"""
Setup script for Sentiment Analysis Pipeline
Run this script to install dependencies and download required models
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print(f"Error message: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Sentiment Analysis Pipeline...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        return False
    
    # Download SpaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading SpaCy English model"):
        print("⚠️ SpaCy model download failed. Entity recognition will not work.")
        print("You can try downloading manually later with: python -m spacy download en_core_web_sm")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Get a free API key from https://newsapi.org/")
    print("2. Set your API key as environment variable: NEWSAPI_KEY=your_key_here")
    print("   Or update the NEWSAPI_KEY in config.py")
    print("3. Run the app: streamlit run main.py")
    print("\n🔗 Useful links:")
    print("- NewsAPI: https://newsapi.org/")
    print("- Streamlit docs: https://docs.streamlit.io/")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
