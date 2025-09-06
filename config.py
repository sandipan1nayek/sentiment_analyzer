"""
Configuration file for sentiment analysis pipeline
Smart API key management with demo mode and persistent storage
"""

import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime, timedelta

# Load environment variables from .env file (for local development)
load_dotenv()

# Demo mode configuration - smart detection based on API key availability
FORCE_DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'

def get_api_key():
    """
    Smart API key retrieval with multiple sources and persistence
    Priority: Session -> Streamlit Secrets -> Environment -> User Input
    """
    # 1. Check if API key is already in session state (persistent across interactions)
    if 'user_api_key' in st.session_state and st.session_state.user_api_key:
        return st.session_state.user_api_key
    
    # 2. Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'NEWSAPI_KEY' in st.secrets:
            api_key = st.secrets['NEWSAPI_KEY']
            if api_key and api_key.strip() and api_key != 'your_newsapi_key_here':
                # Store in session for persistence
                st.session_state.user_api_key = api_key
                return api_key
    except Exception:
        pass
    
    # 3. Fall back to environment variables (for local development)
    api_key = os.getenv('NEWSAPI_KEY')
    if api_key and api_key.strip() and api_key != 'your_newsapi_key_here':
        # Store in session for persistence
        st.session_state.user_api_key = api_key
        return api_key
    
    # 4. Return None if no API key found (will trigger demo mode or user input)
    return None

def is_demo_mode():
    """Check if running in demo mode"""
    # Force demo mode if explicitly set
    if FORCE_DEMO_MODE:
        return True
    
    # Otherwise, demo mode only if no valid API key
    api_key = get_api_key()
    return api_key is None

def get_demo_status():
    """Get demo mode information for UI display"""
    api_key = get_api_key()
    force_demo = FORCE_DEMO_MODE
    
    if force_demo:
        return {
            'is_demo': True,
            'message': '🎭 Demo Mode (Forced)',
            'description': 'Demo mode is force-enabled. Using realistic sample data.',
            'color': 'info'
        }
    elif api_key:
        return {
            'is_demo': False,
            'message': '🔴 Live Mode Active',
            'description': 'Using real-time data from NewsAPI. API key is configured.',
            'color': 'success'
        }
    else:
        return {
            'is_demo': True,
            'message': '🎭 Demo Mode Active',
            'description': 'Using realistic sample data to showcase functionality. No API key required!',
            'color': 'info'
        }

def clear_api_key():
    """Clear stored API key (for switching modes)"""
    if 'user_api_key' in st.session_state:
        del st.session_state.user_api_key
    # Force refresh of demo mode status
    if 'demo_mode_cache' in st.session_state:
        del st.session_state.demo_mode_cache

def set_api_key(api_key):
    """Set API key in session state for persistence"""
    if api_key and api_key.strip() and api_key != 'your_newsapi_key_here':
        st.session_state.user_api_key = api_key.strip()
        # Force refresh of demo mode status
        if 'demo_mode_cache' in st.session_state:
            del st.session_state.demo_mode_cache
        return True
    return False

def validate_api_key(api_key):
    """Basic API key validation"""
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty"
    
    # Basic format check (NewsAPI keys are typically 32 characters)
    if len(api_key.strip()) < 10:
        return False, "API key appears to be too short"
    
    # Could add more validation here (like test API call)
    return True, "API key format looks valid"

# Configuration settings
NEWSAPI_BASE_URL = 'https://newsapi.org/v2/everything'
DEFAULT_TIME_RANGE = 7
MIN_TEXT_LENGTH = 50
SENTIMENT_THRESHOLD = 0.05
MAX_ITEMS_PER_SOURCE = 50

# Sentiment Analysis Thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Time Range Options
TIME_RANGES = {
    "1 Day": 1,
    "7 Days": 7,
    "30 Days": 30
}

# Data cleaning patterns
CLEANING_PATTERNS = {
    'url_pattern': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'mention_pattern': r'@\w+',
    'hashtag_pattern': r'#\w+',
    'emoji_pattern': r'[^\w\s,.]',
    'extra_whitespace': r'\s+'
}

# SpaCy model
SPACY_MODEL = 'en_core_web_sm'

# Streamlit configuration
PAGE_CONFIG = {
    "page_title": "Sentiment Analysis Platform",
    "page_icon": "📊",
    "layout": "wide"
}
