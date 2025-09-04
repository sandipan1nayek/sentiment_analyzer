"""
Professional Data fetching module for real news and social media data
Production-grade sentiment analysis with multiple data sources
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import config
import time
import re
from urllib.parse import quote

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalDataFetcher:
    """Professional-grade data fetcher with multiple real sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_reddit_data(self, topic: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch real Reddit posts and comments about the topic"""
        try:
            search_url = f"https://www.reddit.com/search.json"
            params = {
                'q': topic,
                'sort': 'new',
                'limit': 50,
                't': 'week' if days_back <= 7 else 'month'
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                posts = []
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post['data']
                    text_content = ""
                    
                    # Combine title and text for richer content
                    if post_data.get('title'):
                        text_content = post_data['title']
                    if post_data.get('selftext') and len(post_data['selftext']) > 10:
                        text_content += f" {post_data['selftext'][:300]}"
                    
                    if text_content and len(text_content) > 20:
                        posts.append({
                            'timestamp': pd.to_datetime(post_data['created_utc'], unit='s'),
                            'text': text_content,
                            'source': 'Reddit',
                            'engagement': post_data.get('score', 0),
                            'url': f"https://reddit.com{post_data.get('permalink', '')}"
                        })
                
                logger.info(f"Fetched {len(posts)} Reddit posts for topic: {topic}")
                return pd.DataFrame(posts)
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
        
        return pd.DataFrame(columns=['timestamp', 'text', 'source', 'engagement', 'url'])
    
    def fetch_hackernews_data(self, topic: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch Hacker News stories and comments"""
        try:
            search_url = "https://hn.algolia.com/api/v1/search"
            params = {
                'query': topic,
                'tags': 'story',
                'hitsPerPage': 50,
                'numericFilters': f'created_at_i>{int((datetime.now() - timedelta(days=days_back)).timestamp())}'
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                stories = []
                
                for hit in data.get('hits', []):
                    if hit.get('title') and len(hit['title']) > 10:
                        stories.append({
                            'timestamp': pd.to_datetime(hit['created_at']),
                            'text': hit['title'],
                            'source': 'HackerNews',
                            'engagement': hit.get('points', 0),
                            'url': f"https://news.ycombinator.com/item?id={hit['objectID']}"
                        })
                
                logger.info(f"Fetched {len(stories)} HackerNews stories for topic: {topic}")
                return pd.DataFrame(stories)
            
        except Exception as e:
            logger.error(f"Error fetching HackerNews data: {e}")
        
        return pd.DataFrame(columns=['timestamp', 'text', 'source', 'engagement', 'url'])
    
    def fetch_comprehensive_news(self, topic: str, days_back: int = 7) -> pd.DataFrame:
        """Enhanced NewsAPI fetching with comprehensive coverage"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Multiple search strategies for comprehensive coverage
            search_queries = [
                topic,
                f'"{topic}"',  # Exact phrase
                f'{topic} AND (news OR update OR analysis)',
                f'{topic} OR {topic.lower()}'
            ]
            
            all_articles = []
            seen_titles = set()
            
            for query in search_queries:
                params = {
                    'q': query,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 30,
                    'apiKey': config.NEWSAPI_KEY
                }
                
                response = self.session.get(config.NEWSAPI_BASE_URL, params=params)
                time.sleep(0.2)  # Rate limiting
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        for article in data.get('articles', []):
                            title = article.get('title', '').strip()
                            description = article.get('description', '').strip()
                            
                            # Skip duplicates and low-quality content
                            if title and title not in seen_titles and len(title) > 10:
                                seen_titles.add(title)
                                
                                # Combine title and description for richer analysis
                                full_text = title
                                if description and len(description) > 10:
                                    full_text += f". {description}"
                                
                                all_articles.append({
                                    'timestamp': pd.to_datetime(article['publishedAt']).tz_localize(None),
                                    'text': full_text,
                                    'source': f"News-{article['source']['name']}",
                                    'engagement': 0,
                                    'url': article.get('url', '')
                                })
            
            df = pd.DataFrame(all_articles)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} comprehensive news articles for topic: {topic}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive news: {e}")
            return pd.DataFrame(columns=['timestamp', 'text', 'source', 'engagement', 'url'])

# Initialize the professional fetcher
professional_fetcher = ProfessionalDataFetcher()

def fetch_news_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """Professional news data fetching"""
    return professional_fetcher.fetch_comprehensive_news(topic, days_back)

def fetch_twitter_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """Fetch social media data from multiple real sources"""
    reddit_data = professional_fetcher.fetch_reddit_data(topic, days_back)
    hn_data = professional_fetcher.fetch_hackernews_data(topic, days_back)
    
    # Combine social media sources
    social_data = pd.concat([reddit_data, hn_data], ignore_index=True)
    
    if not social_data.empty:
        logger.info(f"Fetched {len(social_data)} social media posts for topic: {topic}")
    
    return social_data

def fetch_all_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """Professional data fetching from multiple real sources"""
    logger.info(f"Starting comprehensive data fetch for topic: {topic}, days back: {days_back}")
    
    # Fetch from multiple real sources
    news_df = professional_fetcher.fetch_comprehensive_news(topic, days_back)
    social_df = fetch_twitter_data(topic, days_back)
    
    # Combine all sources
    all_data = []
    if not news_df.empty:
        all_data.append(news_df)
    if not social_df.empty:
        all_data.append(social_df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure consistent column structure
        required_columns = ['timestamp', 'text', 'source']
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = ''
        
        # Add data quality metrics
        combined_df['text_length'] = combined_df['text'].str.len()
        combined_df['word_count'] = combined_df['text'].str.split().str.len()
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Total professional data points: {len(combined_df)}")
        logger.info(f"Sources: {combined_df['source'].value_counts().to_dict()}")
        
        return combined_df
    else:
        logger.warning("No data fetched from any source")
        return pd.DataFrame(columns=['timestamp', 'text', 'source', 'engagement', 'url'])

if __name__ == "__main__":
    # Professional testing
    test_topic = "Tesla"
    test_data = fetch_all_data(test_topic, 7)
    print(f"Fetched {len(test_data)} professional data points")
    if not test_data.empty:
        print("\nData sources:")
        print(test_data['source'].value_counts())
        print("\nSample data:")
        print(test_data[['timestamp', 'source', 'text']].head())
