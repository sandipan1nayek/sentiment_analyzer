"""
Configuration file for sentiment analysis pipeline
"""

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# API Configuration
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')  # Get free key from newsapi.org
NEWSAPI_BASE_URL = 'https://newsapi.org/v2/everything'

# Validate required environment variables
if not NEWSAPI_KEY:
    raise ValueError("NEWSAPI_KEY not found in environment variables. Please create a .env file with your API key.")

# Twitter Configuration (snscrape doesn't need API keys)
TWITTER_SEARCH_LIMIT = 100  # Number of tweets to fetch per query

# Sentiment Analysis Thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Time Range Options
TIME_RANGES = {
    "1 Day": 1,
    "7 Days": 7,
    "30 Days": 30
}

# Default settings
DEFAULT_TIME_RANGE = 7
MAX_RESULTS_PER_SOURCE = 100

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
    "page_title": "Sentiment Analysis Pipeline",
    "page_icon": "📊",
    "layout": "wide"
}
