"""
Premium Source Strategy for Comprehensive News Coverage
Implements tiered source prioritization with global + local + sector-specific media
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import config
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PremiumSourceStrategy:
    """
    🌍 Comprehensive source coverage with intelligent prioritization
    - Global mainstream media for international perspective
    - Indian media for local relevance  
    - Sector-specific media for domain expertise
    """
    
    def __init__(self):
        # Tier 1: Global Premium Sources (Highest Priority)
        self.global_premium_sources = [
            "reuters", "bloomberg", "financial-times", "the-wall-street-journal",
            "the-guardian-uk", "bbc-news", "the-new-york-times", "the-washington-post",
            "cnn", "cnbc", "forbes", "techcrunch", "wired"
        ]
        
        # Tier 2: Indian Mainstream Media (High Priority for Indian topics)
        self.indian_mainstream_sources = [
            "the-times-of-india", "the-hindu", "the-indian-express", 
            "hindustan-times", "ndtv", "india-today", "economic-times",
            "business-standard", "livemint", "moneycontrol"
        ]
        
        # Tier 3: Sector-Specific Sources (Medium Priority)
        self.sector_specific_sources = {
            "automotive": ["automotive-news", "electrek", "insideevs", "autocar-india"],
            "technology": ["techcrunch", "wired", "ars-technica", "the-verge"],
            "finance": ["bloomberg", "financial-times", "economic-times", "moneycontrol"],
            "politics": ["the-guardian-uk", "the-washington-post", "the-hindu", "ndtv"]
        }
        
        # Source distribution strategy
        self.source_distribution = {
            "global_premium": 40,    # 40% from top global sources
            "indian_mainstream": 35, # 35% from Indian media
            "sector_specific": 25    # 25% from domain experts
        }
        
        # Enhanced search domains for broader coverage
        self.search_domains = [
            "site:reuters.com",
            "site:bloomberg.com", 
            "site:ft.com",
            "site:wsj.com",
            "site:theguardian.com",
            "site:bbc.com",
            "site:nytimes.com",
            "site:washingtonpost.com",
            "site:cnn.com",
            "site:cnbc.com",
            "site:forbes.com",
            "site:timesofindia.indiatimes.com",
            "site:thehindu.com",
            "site:indianexpress.com",
            "site:hindustantimes.com",
            "site:ndtv.com",
            "site:indiatoday.in",
            "site:economictimes.indiatimes.com",
            "site:business-standard.com",
            "site:livemint.com",
            "site:moneycontrol.com"
        ]
    
    def fetch_comprehensive_premium_news(self, topic: str, days_back: int = 7) -> pd.DataFrame:
        """
        🌍 Fetch from premium sources with intelligent prioritization
        """
        logger.info(f"🌍 Fetching premium coverage for: '{topic}'")
        
        all_articles = []
        
        # Step 1: Global Premium Sources (Highest Priority)
        global_articles = self._fetch_from_source_tier(
            topic, days_back, self.global_premium_sources, "global_premium"
        )
        all_articles.extend(global_articles)
        
        # Step 2: Indian Mainstream (High Priority for Indian relevance)
        indian_articles = self._fetch_from_source_tier(
            topic, days_back, self.indian_mainstream_sources, "indian_mainstream"
        )
        all_articles.extend(indian_articles)
        
        # Step 3: Sector-Specific Sources
        sector_articles = self._fetch_sector_specific(topic, days_back)
        all_articles.extend(sector_articles)
        
        # Step 4: Enhanced Google News Search for missed sources
        google_articles = self._fetch_from_google_domains(topic, days_back)
        all_articles.extend(google_articles)
        
        # Process and prioritize results
        if all_articles:
            df = self._process_and_prioritize_articles(all_articles, topic)
            logger.info(f"✅ Premium coverage: {len(df)} articles from {df['source_tier'].value_counts().to_dict()}")
            return df
        
        return pd.DataFrame()
    
    def _fetch_from_source_tier(self, topic: str, days_back: int, sources: List[str], tier: str) -> List[Dict]:
        """Fetch from specific source tier - Enhanced approach"""
        articles = []
        
        # Use broader search and filter by source quality afterward
        try:
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params={
                    'q': f'"{topic}"',  # Exact phrase for precision
                    'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 50,  # Larger pool for source diversity
                    'apiKey': config.NEWSAPI_KEY
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    if self._is_premium_quality(article, topic):
                        source_name = article['source']['name'].lower()
                        
                        # Categorize by source quality
                        source_tier = self._categorize_source_by_quality(source_name)
                        
                        articles.append({
                            'timestamp': pd.to_datetime(article['publishedAt']).tz_localize(None),
                            'text': self._extract_rich_content(article),
                            'source': f"Premium-{article['source']['name']}",
                            'source_tier': source_tier,
                            'url': article.get('url', ''),
                            'priority_score': self._calculate_source_priority(article['source']['name'], source_tier)
                        })
            
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"Error in {tier} tier fetch: {e}")
        
        return articles
    
    def _fetch_sector_specific(self, topic: str, days_back: int) -> List[Dict]:
        """Fetch from sector-specific sources based on topic"""
        articles = []
        
        # Determine relevant sectors
        topic_lower = topic.lower()
        relevant_sectors = []
        
        if any(term in topic_lower for term in ['car', 'auto', 'ev', 'tesla', 'mahindra', 'tata motors']):
            relevant_sectors.append('automotive')
        if any(term in topic_lower for term in ['ai', 'tech', 'chatgpt', 'software', 'innovation']):
            relevant_sectors.append('technology')
        if any(term in topic_lower for term in ['stock', 'market', 'finance', 'economy', 'investment']):
            relevant_sectors.append('finance')
        if any(term in topic_lower for term in ['modi', 'government', 'policy', 'election', 'politics']):
            relevant_sectors.append('politics')
        
        # Fetch from relevant sector sources
        for sector in relevant_sectors:
            sector_sources = self.sector_specific_sources.get(sector, [])
            for source in sector_sources[:5]:  # Limit per sector
                try:
                    response = requests.get(
                        'https://newsapi.org/v2/everything',
                        params={
                            'q': f'"{topic}"',
                            'sources': source,
                            'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                            'to': datetime.now().strftime('%Y-%m-%d'),
                            'language': 'en',
                            'sortBy': 'relevancy',
                            'pageSize': 15,
                            'apiKey': config.NEWSAPI_KEY
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for article in data.get('articles', []):
                            if self._is_premium_quality(article, topic):
                                articles.append({
                                    'timestamp': pd.to_datetime(article['publishedAt']).tz_localize(None),
                                    'text': self._extract_rich_content(article),
                                    'source': f"Sector-{article['source']['name']}",
                                    'source_tier': f"sector_{sector}",
                                    'url': article.get('url', ''),
                                    'priority_score': self._calculate_source_priority(article['source']['name'], 'sector')
                                })
                    
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.warning(f"Error fetching from sector source {source}: {e}")
                    continue
        
        return articles
    
    def _fetch_from_google_domains(self, topic: str, days_back: int) -> List[Dict]:
        """Fetch additional coverage from specific domains using Google Custom Search"""
        articles = []
        
        # This would use Google Custom Search API for domain-specific searches
        # For now, we'll use additional NewsAPI calls with broader terms
        
        try:
            # Broader search to catch more sources
            response = requests.get(
                'https://newsapi.org/v2/everything',
                params={
                    'q': f'{topic} OR "{topic}"',  # Both exact and broad
                    'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 50,  # Larger pool for diversity
                    'apiKey': config.NEWSAPI_KEY
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    if self._is_premium_quality(article, topic):
                        articles.append({
                            'timestamp': pd.to_datetime(article['publishedAt']).tz_localize(None),
                            'text': self._extract_rich_content(article),
                            'source': f"Extended-{article['source']['name']}",
                            'source_tier': 'extended',
                            'url': article.get('url', ''),
                            'priority_score': self._calculate_source_priority(article['source']['name'], 'extended')
                        })
        
        except Exception as e:
            logger.warning(f"Error in extended search: {e}")
        
        return articles
    
    def _is_premium_quality(self, article: dict, topic: str) -> bool:
        """Enhanced quality check for premium content"""
        if not article or not article.get('title'):
            return False
        
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        
        # Must mention the topic
        topic_lower = topic.lower()
        if topic_lower not in f"{title} {description}":
            return False
        
        # Quality indicators
        if len(title) < 15:
            return False
        
        # Skip promotional content
        spam_indicators = ['advertisement', 'sponsored', 'press release']
        if any(spam in f"{title} {description}" for spam in spam_indicators):
            return False
        
        return True
    
    def _extract_rich_content(self, article: dict) -> str:
        """Extract comprehensive content from article"""
        title = article.get('title', '').strip()
        description = article.get('description', '').strip()
        content = article.get('content', '').strip()
        
        # Combine all available content
        full_content = title
        if description and len(description) > 10:
            full_content += f". {description}"
        if content and len(content) > 20:
            # Take first part of content (often truncated by NewsAPI)
            content_excerpt = content[:200] if len(content) > 200 else content
            full_content += f" {content_excerpt}"
        
        return full_content.strip()
    
    def _calculate_source_priority(self, source_name: str, tier: str) -> float:
        """Calculate priority score based on source and tier"""
        base_scores = {
            'global_premium': 1.0,
            'indian_mainstream': 0.9,
            'sector': 0.8,
            'extended': 0.7
        }
        
        # Boost for specific premium sources
        premium_boost = {
            'reuters': 0.1,
            'bloomberg': 0.1,
            'bbc': 0.1,
            'financial times': 0.1,
            'the times of india': 0.08,
            'the hindu': 0.08,
            'economic times': 0.08
        }
        
        base_score = base_scores.get(tier, 0.5)
        source_lower = source_name.lower()
        
        for premium_source, boost in premium_boost.items():
            if premium_source in source_lower:
                base_score += boost
                break
        
        return min(base_score, 1.0)
    
    def _categorize_source_by_quality(self, source_name: str) -> str:
        """Categorize source by quality tier based on name"""
        source_lower = source_name.lower()
        
        # Global premium indicators
        global_indicators = [
            'reuters', 'bloomberg', 'bbc', 'financial times', 'ft.com',
            'wall street journal', 'wsj', 'guardian', 'new york times',
            'nytimes', 'washington post', 'cnn', 'cnbc', 'forbes'
        ]
        
        # Indian mainstream indicators  
        indian_indicators = [
            'times of india', 'toi', 'hindu', 'indian express', 
            'hindustan times', 'ndtv', 'india today', 'economic times',
            'business standard', 'mint', 'livemint', 'moneycontrol'
        ]
        
        # Sector-specific indicators
        sector_indicators = [
            'techcrunch', 'wired', 'verge', 'ars technica',
            'automotive news', 'electrek', 'autocar', 'overdrive'
        ]
        
        # Check categories
        for indicator in global_indicators:
            if indicator in source_lower:
                return 'global_premium'
        
        for indicator in indian_indicators:
            if indicator in source_lower:
                return 'indian_mainstream'
                
        for indicator in sector_indicators:
            if indicator in source_lower:
                return 'sector_specific'
        
        return 'extended'
    
    def _process_and_prioritize_articles(self, articles: List[Dict], topic: str) -> pd.DataFrame:
        """Process and prioritize all articles"""
        if not articles:
            return pd.DataFrame()
        
        df = pd.DataFrame(articles)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Sort by priority score and timestamp
        df = df.sort_values(['priority_score', 'timestamp'], ascending=[False, False])
        
        # Apply source distribution limits
        balanced_articles = []
        
        # Global premium (40%)
        global_articles = df[df['source_tier'] == 'global_premium'].head(40)
        balanced_articles.append(global_articles)
        
        # Indian mainstream (35%) 
        indian_articles = df[df['source_tier'] == 'indian_mainstream'].head(35)
        balanced_articles.append(indian_articles)
        
        # Sector + extended (25%)
        other_articles = df[~df['source_tier'].isin(['global_premium', 'indian_mainstream'])].head(25)
        balanced_articles.append(other_articles)
        
        # Combine and finalize
        if balanced_articles:
            final_df = pd.concat(balanced_articles, ignore_index=True)
            final_df = final_df.sort_values('priority_score', ascending=False)
            
            # Add category for compatibility
            final_df['category'] = 'news'
            
            return final_df.head(100)  # Maximum 100 articles
        
        return df.head(100)

# Initialize premium source strategy
premium_source_strategy = PremiumSourceStrategy()

def fetch_premium_news_coverage(topic: str, days_back: int = 7) -> pd.DataFrame:
    """
    🌍 Main function: Fetch comprehensive premium news coverage
    """
    return premium_source_strategy.fetch_comprehensive_premium_news(topic, days_back)

if __name__ == "__main__":
    # Test premium source strategy
    test_topic = "Mahindra"
    print(f"🌍 Testing Premium Source Strategy for: {test_topic}")
    
    premium_data = fetch_premium_news_coverage(test_topic, 7)
    
    if not premium_data.empty:
        print(f"\n✅ Premium Coverage: {len(premium_data)} articles")
        print(f"\n📊 Source Tier Distribution:")
        print(premium_data['source_tier'].value_counts())
        print(f"\n🏆 Top Sources:")
        print(premium_data['source'].value_counts().head(10))
        
        print(f"\n📰 Sample Headlines:")
        for idx, row in premium_data.head(5).iterrows():
            print(f"- [{row['source']}] {row['text'][:80]}...")
    else:
        print("❌ No premium data fetched")
