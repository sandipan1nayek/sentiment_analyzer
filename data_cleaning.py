"""
Data cleaning module for text preprocessing
"""

import pandas as pd
import re
import html
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Professional data cleaning class"""
    
    def __init__(self):
        """Initialize the data cleaner"""
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean individual text"""
        return clean_text(text)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean entire dataframe"""
        return clean_data(df)
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean entire dataframe - alias for clean_data"""
        return clean_data(df)
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        return preprocess_for_sentiment(text)

def clean_text(text: str) -> str:
    """
    Clean individual text by removing URLs, mentions, hashtags, emojis, and extra whitespace
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove Twitter mentions and hashtags (but keep the content after #)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
    
    # Remove RT indicator
    text = re.sub(r'^RT\s+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip and return
    return text.strip()

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate texts while keeping the most recent timestamp
    
    Args:
        df (pd.DataFrame): DataFrame with text data
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    # Sort by timestamp (most recent first) and remove duplicates based on text
    df_sorted = df.sort_values('timestamp', ascending=False)
    df_deduped = df_sorted.drop_duplicates(subset=['text'], keep='first')
    
    # Sort back by timestamp (oldest first)
    df_deduped = df_deduped.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Removed {len(df) - len(df_deduped)} duplicate texts")
    return df_deduped

def filter_quality_text(df: pd.DataFrame, min_length: int = 10, max_length: int = 500) -> pd.DataFrame:
    """
    Filter texts based on quality criteria
    
    Args:
        df (pd.DataFrame): DataFrame with text data
        min_length (int): Minimum text length
        max_length (int): Maximum text length
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if df.empty:
        return df
    
    initial_count = len(df)
    
    # Filter by length
    df_filtered = df[
        (df['text'].str.len() >= min_length) & 
        (df['text'].str.len() <= max_length)
    ].copy()
    
    # Remove texts that are mostly special characters or numbers
    df_filtered = df_filtered[
        df_filtered['text'].str.contains(r'[a-zA-Z]', regex=True)
    ].copy()
    
    logger.info(f"Filtered {initial_count - len(df_filtered)} low-quality texts")
    return df_filtered.reset_index(drop=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning function that applies all cleaning steps
    
    Args:
        df (pd.DataFrame): Raw DataFrame with columns ['timestamp', 'text', 'source']
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for cleaning")
        return df
    
    logger.info(f"Starting data cleaning for {len(df)} records")
    
    # Create a copy to avoid modifying original
    df_cleaned = df.copy()
    
    # Clean text
    df_cleaned['text'] = df_cleaned['text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df_cleaned = df_cleaned[df_cleaned['text'].str.strip() != ''].copy()
    
    # Remove duplicates
    df_cleaned = remove_duplicates(df_cleaned)
    
    # Filter by quality
    df_cleaned = filter_quality_text(df_cleaned)
    
    logger.info(f"Cleaning completed. {len(df_cleaned)} records remaining from {len(df)} original records")
    
    return df_cleaned

def preprocess_for_sentiment(text: str) -> str:
    """
    Additional preprocessing specifically for sentiment analysis
    
    Args:
        text (str): Cleaned text
        
    Returns:
        str: Text prepared for sentiment analysis
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove excessive punctuation (keep some for sentiment context)
    text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
    text = re.sub(r'[!]{2,}', '!', text)    # Multiple exclamations to one
    text = re.sub(r'[?]{2,}', '?', text)    # Multiple questions to one
    
    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    # Test the cleaning functions
    test_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'text': [
            'Check out this amazing article! https://example.com #AI @username 🔥🔥',
            'RT @someone: This is a retweet with emojis 😊🚀 and extra   spaces'
        ],
        'source': ['NewsAPI', 'Twitter']
    })
    
    cleaned_data = clean_data(test_data)
    print("Original:")
    print(test_data['text'].tolist())
    print("\nCleaned:")
    print(cleaned_data['text'].tolist())
