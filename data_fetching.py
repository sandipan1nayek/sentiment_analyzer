"""
Data fetching module for news and Twitter data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import config
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For now, we'll use sample Twitter data instead of snscrape due to compatibility issues
SAMPLE_TWEETS = [
    "Just bought a {topic}! Absolutely loving it so far! #amazing #excited",
    "The new {topic} announcement has me concerned... not sure about this direction",
    "Finally got my hands on {topic}. Pretty good but nothing revolutionary",
    "{topic} is everywhere these days. Getting a bit tired of all the hype honestly",
    "Wow, {topic} just changed the game completely! This is incredible technology",
    "Not impressed with {topic} at all. Expected much more for the price",
    "Been using {topic} for a week now. Solid product, would recommend to others",
    "The {topic} reviews are mixed but I'm optimistic about the future potential",
    "Just saw the latest {topic} update. Some interesting features but still buggy",
    "Amazing experience with {topic} today! Customer service was fantastic too"
]

def fetch_news_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """
    Fetch news headlines from NewsAPI
    
    Args:
        topic (str): Topic to search for
        days_back (int): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'text', 'source']
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Prepare API request
        params = {
            'q': topic,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': config.MAX_RESULTS_PER_SOURCE,
            'apiKey': config.NEWSAPI_KEY
        }
        
        # Make API request
        response = requests.get(config.NEWSAPI_BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] != 'ok':
            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame(columns=['timestamp', 'text', 'source'])
        
        # Process articles
        articles = []
        for article in data['articles']:
            if article['title']:  # Only include articles with titles
                articles.append({
                    'timestamp': pd.to_datetime(article['publishedAt']),
                    'text': article['title'],
                    'source': 'NewsAPI'
                })
        
        logger.info(f"Fetched {len(articles)} news articles for topic: {topic}")
        return pd.DataFrame(articles)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news data: {e}")
        return pd.DataFrame(columns=['timestamp', 'text', 'source'])
    except Exception as e:
        logger.error(f"Unexpected error in fetch_news_data: {e}")
        return pd.DataFrame(columns=['timestamp', 'text', 'source'])

def fetch_twitter_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """
    Generate sample Twitter-like data for demonstration
    (Real Twitter API requires extensive authentication)
    
    Args:
        topic (str): Topic to search for
        days_back (int): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'text', 'source']
    """
    try:
        logger.info(f"Generating sample Twitter data for topic: {topic}")
        
        # Generate sample timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        tweets = []
        num_tweets = min(config.TWITTER_SEARCH_LIMIT, 20)  # Limit sample data
        
        for i in range(num_tweets):
            # Generate random timestamp within range
            time_diff = end_date - start_date
            random_seconds = random.randint(0, int(time_diff.total_seconds()))
            timestamp = start_date + timedelta(seconds=random_seconds)
            
            # Select random tweet template and fill in topic
            tweet_template = random.choice(SAMPLE_TWEETS)
            tweet_text = tweet_template.format(topic=topic)
            
            tweets.append({
                'timestamp': timestamp,
                'text': tweet_text,
                'source': 'Twitter'
            })
        
        logger.info(f"Generated {len(tweets)} sample tweets for topic: {topic}")
        return pd.DataFrame(tweets)
        
    except Exception as e:
        logger.error(f"Error generating sample Twitter data: {e}")
        return pd.DataFrame(columns=['timestamp', 'text', 'source'])

def fetch_all_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """
    Fetch data from all sources and combine
    
    Args:
        topic (str): Topic to search for
        days_back (int): Number of days to look back
        
    Returns:
        pd.DataFrame: Combined DataFrame with all data
    """
    logger.info(f"Starting data fetch for topic: {topic}, days back: {days_back}")
    
    # Fetch from both sources
    news_df = fetch_news_data(topic, days_back)
    twitter_df = fetch_twitter_data(topic, days_back)
    
    # Combine dataframes
    combined_df = pd.concat([news_df, twitter_df], ignore_index=True)
    
    if not combined_df.empty:
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Total data points fetched: {len(combined_df)}")
    else:
        logger.warning("No data fetched from any source")
    
    return combined_df

if __name__ == "__main__":
    # Test the functions
    test_topic = "Tesla"
    test_data = fetch_all_data(test_topic, 3)
    print(f"Fetched {len(test_data)} total data points")
    if not test_data.empty:
        print(test_data.head())
