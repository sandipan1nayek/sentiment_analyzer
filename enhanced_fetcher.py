"""
Enhanced Multi-Source Data Fetcher for 360° Sentiment Analysis
Includes: International relations, VIP remarks, public sentiment, policy coverage
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple
import config
import time
import re
from urllib.parse import quote
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSourceIntelligenceFetcher:
    """
    Advanced multi-source data fetcher for comprehensive sentiment analysis
    Sources: News + Social Media + VIP Statements + International Coverage
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Source distribution for balanced analysis
        self.source_targets = {
            'news': 40,      # 40 Traditional news articles
            'social': 40,    # 40 Social media posts
            'vip': 20        # 20 VIP statements/official content
        }
        
        # Initialize relevance filter
        try:
            from relevance_filter import relevance_filter
            self.relevance_filter = relevance_filter
            logger.info("Advanced relevance filtering enabled")
        except ImportError:
            logger.warning("Relevance filter not available")
            self.relevance_filter = None
            
        # Initialize premium source strategy
        try:
            from premium_sources import premium_source_strategy
            self.premium_sources = premium_source_strategy
            logger.info("Premium source strategy enabled")
        except ImportError:
            logger.warning("Premium sources not available")
            self.premium_sources = None
            
        # Initialize comprehensive media coverage
        try:
            from comprehensive_media import comprehensive_coverage
            self.comprehensive_media = comprehensive_coverage
            logger.info("Comprehensive media coverage enabled")
        except ImportError:
            logger.warning("Comprehensive media not available")
            self.comprehensive_media = None
    
    def fetch_comprehensive_news(self, topic: str, days_back: int = 7) -> Dict:
        """
        🌍 NEW: Comprehensive news with fallback + Twitter integration
        Returns: {'data': articles_list, 'summary': info_string, 'domain_info': domain_analysis}
        """
        # Use comprehensive media coverage if available
        if self.comprehensive_media:
            logger.info("🌍 Using comprehensive media coverage with fallback strategy")
            comprehensive_results = self.comprehensive_media.fetch_comprehensive_coverage(topic, days_back)
            
            if comprehensive_results and comprehensive_results.get('data'):
                articles = comprehensive_results['data']
                domain_info = comprehensive_results.get('domain_info', {})
                coverage_stats = comprehensive_results.get('coverage_stats', {})
                coverage_analysis = comprehensive_results.get('coverage_analysis', {})
                
                # Enhanced summary with domain intelligence
                summary = f"Found {len(articles)} items from comprehensive coverage"
                summary += f" | Domain: {domain_info.get('primary_domain', 'general')}"
                summary += f" | News: {coverage_stats.get('premium_news', 0) + coverage_stats.get('fallback_news', 0)}"
                summary += f" | Twitter: {coverage_stats.get('twitter', 0)}"
                summary += f" | Social: {coverage_stats.get('other_social', 0)}"
                
                if coverage_analysis.get('fallback_activated'):
                    summary += " | ⚡ Fallback activated for better coverage"
                
                return {
                    'data': articles,
                    'summary': summary,
                    'domain_info': domain_info,
                    'coverage_stats': coverage_stats,
                    'coverage_analysis': coverage_analysis
                }
        
        # Fallback to original method
        df = self.fetch_comprehensive_intelligence(topic, days_back)
        
        if df.empty:
            return {
                'data': [],
                'summary': f"No articles found for '{topic}'"
            }
        
        # Convert to list format
        articles = []
        for _, row in df.iterrows():
            articles.append({
                'title': row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
                'source': row['source'],
                'url': row.get('url', ''),
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'text': row['text']
            })
        
        # Create summary
        source_counts = df['source'].value_counts()
        summary = f"Found {len(articles)} articles from {len(source_counts)} sources"
        
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            summary += f" (News: {category_counts.get('news', 0)}, Social: {category_counts.get('social', 0)}, VIP: {category_counts.get('vip', 0)})"
        
        return {
            'data': articles,
            'summary': summary
        }
    
    def fetch_comprehensive_intelligence(self, topic: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch comprehensive data from multiple sources for 360° analysis
        """
        logger.info(f"🚀 Starting comprehensive intelligence fetch for: '{topic}'")
        
        all_data = []
        
        # 1. Traditional News (40% - International + Local)
        news_data = self._fetch_enhanced_news(topic, days_back)
        if not news_data.empty:
            news_data['category'] = 'news'
            all_data.append(news_data)
            logger.info(f"📰 News articles: {len(news_data)}")
        
        # 2. Social Media & Public Sentiment (40%)
        social_data = self._fetch_social_intelligence(topic, days_back)
        if not social_data.empty:
            social_data['category'] = 'social'
            all_data.append(social_data)
            logger.info(f"💬 Social posts: {len(social_data)}")
        
        # 3. VIP Statements & Official Content (20%)
        vip_data = self._fetch_vip_statements(topic, days_back)
        if not vip_data.empty:
            vip_data['category'] = 'vip'
            all_data.append(vip_data)
            logger.info(f"🎤 VIP statements: {len(vip_data)}")
        
        # Combine all sources
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Apply advanced relevance filtering
            if self.relevance_filter:
                logger.info("🔍 Applying advanced relevance filtering...")
                combined_df = self.relevance_filter.filter_by_relevance(combined_df, topic, threshold=0.5)
                combined_df = self.relevance_filter.remove_noise(combined_df)
            
            # Ensure balanced representation
            combined_df = self._balance_sources(combined_df)
            
            # Sort by relevance and timestamp
            if 'relevance_score' in combined_df.columns:
                combined_df = combined_df.sort_values(['relevance_score', 'timestamp'], ascending=[False, False])
            else:
                combined_df = combined_df.sort_values('timestamp', ascending=False)
            
            logger.info(f"✅ Total intelligent data points: {len(combined_df)}")
            return combined_df.reset_index(drop=True)
        
        return pd.DataFrame()
    
    def _fetch_enhanced_news(self, topic: str, days_back: int) -> pd.DataFrame:
        """
        🌍 Enhanced news fetching with premium global + Indian + sector sources
        """
        
        # Use premium source strategy if available
        if self.premium_sources:
            logger.info("🌍 Using premium source strategy for comprehensive coverage")
            premium_data = self.premium_sources.fetch_comprehensive_premium_news(topic, days_back)
            if not premium_data.empty:
                logger.info(f"📰 Premium news coverage: {len(premium_data)} articles")
                return premium_data
        
        # Fallback to enhanced method
        articles = []
        
        # Intelligent query expansion
        if self.relevance_filter:
            queries = self.relevance_filter.expand_search_queries(topic)
        else:
            queries = [f'"{topic}"', topic, f'{topic} news']
        
        # Enhanced query contexts - INTELLIGENT CONTEXT SELECTION
        contexts = self._get_relevant_contexts(topic)
        
        for base_query in queries[:3]:  # Limit to top 3 base queries
            for context in contexts:  # Use only relevant contexts
                try:
                    search_query = f"{base_query}{context}".strip()
                    
                    response = self.session.get(
                        'https://newsapi.org/v2/everything',
                        params={
                            'q': search_query,
                            'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                            'to': datetime.now().strftime('%Y-%m-%d'),
                            'language': 'en',
                            'sortBy': 'relevancy',
                            'pageSize': 15,  # Increased for better coverage
                            'apiKey': config.NEWSAPI_KEY
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for article in data.get('articles', []):
                            if self._is_quality_article(article, topic):
                                articles.append({
                                    'timestamp': pd.to_datetime(article['publishedAt']).tz_localize(None),
                                    'text': self._extract_rich_text(article),
                                    'source': f"News-{article['source']['name']}",
                                    'url': article.get('url', ''),
                                    'context': context.strip() or 'general'
                                })
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error with query '{search_query}': {e}")
                    continue
        
        # Remove duplicates and return top articles
        if articles:
            df = pd.DataFrame(articles)
            df = df.drop_duplicates(subset=['text'])
            return df.head(self.source_targets['news'])
        
        return pd.DataFrame()
    
    def _get_relevant_contexts(self, topic: str) -> List[str]:
        """
        🧠 INTELLIGENT CONTEXT SELECTION based on entity type
        Only applies relevant contexts to avoid noise
        """
        topic_lower = topic.lower()
        
        # Detect entity type
        is_person = self._is_person_topic(topic_lower)
        is_company = self._is_company_topic(topic_lower) 
        is_technology = self._is_technology_topic(topic_lower)
        is_political = self._is_political_topic(topic_lower)
        
        contexts = ['']  # Always include basic query
        
        # Person/Political Figure contexts
        if is_person or is_political:
            contexts.extend([
                ' policy',         # Policy statements
                ' international',  # International relations  
                ' speech',         # Speeches
                ' statement'       # Official statements
            ])
        
        # Company/Organization contexts
        elif is_company:
            contexts.extend([
                ' stock',          # Stock performance
                ' CEO',            # Leadership news
                ' earnings',       # Financial results
                ' product'         # Product announcements
            ])
        
        # Technology/Product contexts
        elif is_technology:
            contexts.extend([
                ' technology',     # Technical developments
                ' research',       # Research advances
                ' application',    # Use cases
                ' development'     # Development news
            ])
        
        # General topic contexts (fallback)
        else:
            contexts.extend([
                ' news',           # General news
                ' update',         # Updates
                ' analysis'        # Analysis
            ])
        
        return contexts[:5]  # Limit to 5 most relevant contexts
    
    def _is_person_topic(self, topic: str) -> bool:
        """Check if topic is a person/political figure"""
        person_indicators = [
            'modi', 'biden', 'trump', 'musk', 'gates', 'bezos', 'cook',
            'zuckerberg', 'putin', 'xi', 'macron', 'merkel', 'obama'
        ]
        return any(indicator in topic for indicator in person_indicators)
    
    def _is_company_topic(self, topic: str) -> bool:
        """Check if topic is a company/organization"""
        company_indicators = [
            'tesla', 'apple', 'microsoft', 'google', 'meta', 'amazon', 
            'netflix', 'uber', 'airbnb', 'spacex', 'openai', 'nvidia'
        ]
        return any(indicator in topic for indicator in company_indicators)
    
    def _is_technology_topic(self, topic: str) -> bool:
        """Check if topic is technology/product related"""
        tech_indicators = [
            'ai', 'artificial intelligence', 'chatgpt', 'blockchain', 
            'bitcoin', 'cryptocurrency', 'machine learning', 'robotics',
            'quantum', 'vr', 'ar', 'metaverse', '5g', 'cloud computing'
        ]
        return any(indicator in topic for indicator in tech_indicators)
    
    def _is_political_topic(self, topic: str) -> bool:
        """Check if topic is political in nature"""
        political_indicators = [
            'election', 'government', 'congress', 'parliament', 'senate',
            'policy', 'law', 'regulation', 'democracy', 'vote'
        ]
        return any(indicator in topic for indicator in political_indicators)
    
    def _fetch_social_intelligence(self, topic: str, days_back: int) -> pd.DataFrame:
        """
        Fetch social media intelligence for public sentiment
        """
        social_data = []
        
        # Reddit discussions
        reddit_data = self._fetch_reddit_discussions(topic, days_back)
        if not reddit_data.empty:
            social_data.append(reddit_data)
        
        # HackerNews for tech topics
        hn_data = self._fetch_hackernews_discussions(topic, days_back)
        if not hn_data.empty:
            social_data.append(hn_data)
        
        # Combine social sources
        if social_data:
            combined = pd.concat(social_data, ignore_index=True)
            return combined.head(self.source_targets['social'])
        
        return pd.DataFrame()
    
    def _fetch_reddit_discussions(self, topic: str, days_back: int) -> pd.DataFrame:
        """Enhanced Reddit fetching with better query targeting"""
        try:
            # Enhanced search queries for Reddit
            search_queries = [
                topic,
                f'"{topic}"',
                f'{topic} news',
                f'{topic} discussion'
            ]
            
            all_posts = []
            
            for query in search_queries[:2]:  # Limit queries
                try:
                    response = self.session.get(
                        'https://www.reddit.com/search.json',
                        params={
                            'q': query,
                            'sort': 'relevance',
                            'limit': 25,
                            't': 'week' if days_back <= 7 else 'month'
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for post in data.get('data', {}).get('children', []):
                            post_data = post['data']
                            
                            # Enhanced text extraction
                            text_content = self._extract_reddit_text(post_data)
                            
                            if text_content and len(text_content) > 30:
                                all_posts.append({
                                    'timestamp': pd.to_datetime(post_data['created_utc'], unit='s'),
                                    'text': text_content,
                                    'source': 'Reddit',
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                    'engagement': post_data.get('score', 0)
                                })
                    
                    time.sleep(1)  # Reddit rate limiting
                    
                except Exception as e:
                    logger.warning(f"Reddit query error: {e}")
                    continue
            
            return pd.DataFrame(all_posts)
            
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
            return pd.DataFrame()
    
    def _fetch_hackernews_discussions(self, topic: str, days_back: int) -> pd.DataFrame:
        """Enhanced HackerNews fetching"""
        try:
            response = self.session.get(
                'https://hn.algolia.com/api/v1/search',
                params={
                    'query': topic,
                    'tags': 'story',
                    'hitsPerPage': 30,
                    'numericFilters': f'created_at_i>{int((datetime.now() - timedelta(days=days_back)).timestamp())}'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                stories = []
                
                for hit in data.get('hits', []):
                    title = hit.get('title', '').strip()
                    if title and len(title) > 20:
                        stories.append({
                            'timestamp': pd.to_datetime(hit['created_at']),
                            'text': title,
                            'source': 'HackerNews',
                            'url': f"https://news.ycombinator.com/item?id={hit['objectID']}",
                            'engagement': hit.get('points', 0)
                        })
                
                return pd.DataFrame(stories)
            
        except Exception as e:
            logger.error(f"HackerNews fetch error: {e}")
        
        return pd.DataFrame()
    
    def _fetch_vip_statements(self, topic: str, days_back: int) -> pd.DataFrame:
        """
        Fetch VIP statements, official announcements, and policy content
        """
        vip_data = []
        
        # VIP-focused search queries
        vip_queries = [
            f'"{topic}" statement',
            f'"{topic}" speech',
            f'"{topic}" announcement',
            f'"{topic}" interview',
            f'"{topic}" press conference'
        ]
        
        for query in vip_queries[:3]:  # Limit VIP queries
            try:
                response = self.session.get(
                    'https://newsapi.org/v2/everything',
                    params={
                        'q': query,
                        'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                        'to': datetime.now().strftime('%Y-%m-%d'),
                        'language': 'en',
                        'sortBy': 'relevancy',
                        'pageSize': 10,
                        'apiKey': config.NEWSAPI_KEY
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        if self._is_vip_content(article, topic):
                            vip_data.append({
                                'timestamp': pd.to_datetime(article['publishedAt']).tz_localize(None),
                                'text': self._extract_rich_text(article),
                                'source': f"VIP-{article['source']['name']}",
                                'url': article.get('url', ''),
                                'type': 'official'
                            })
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"VIP query error: {e}")
                continue
        
        return pd.DataFrame(vip_data).head(self.source_targets['vip'])
    
    def _is_quality_article(self, article: dict, topic: str) -> bool:
        """Enhanced quality check for articles"""
        if not article or not article.get('title'):
            return False
        
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        
        # Skip promotional content
        spam_indicators = ['advertisement', 'sponsored', 'buy now', 'discount']
        if any(spam in f"{title} {description}" for spam in spam_indicators):
            return False
        
        # Check minimum content quality
        if len(title) < 20:
            return False
        
        # Topic relevance (basic check)
        topic_lower = topic.lower()
        if topic_lower not in f"{title} {description}":
            return False
        
        return True
    
    def _is_vip_content(self, article: dict, topic: str) -> bool:
        """Check if content is VIP/official statement"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        vip_indicators = [
            'statement', 'announces', 'declares', 'speech', 'address',
            'press conference', 'interview', 'official', 'government',
            'minister', 'president', 'prime minister', 'says', 'tells'
        ]
        
        return any(indicator in text for indicator in vip_indicators)
    
    def _extract_rich_text(self, article: dict) -> str:
        """Extract rich text content from article"""
        title = article.get('title', '').strip()
        description = article.get('description', '').strip()
        
        # Combine for richer context
        if description and len(description) > 10:
            return f"{title}. {description}"
        return title
    
    def _extract_reddit_text(self, post_data: dict) -> str:
        """Extract meaningful text from Reddit post"""
        title = post_data.get('title', '').strip()
        selftext = post_data.get('selftext', '').strip()
        
        if selftext and len(selftext) > 20:
            # Combine title and content, limit length
            combined = f"{title}. {selftext[:300]}"
            return combined
        return title
    
    def _balance_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure balanced representation across source categories"""
        if df.empty or 'category' not in df.columns:
            return df
        
        balanced_data = []
        
        for category, target_count in self.source_targets.items():
            category_data = df[df['category'] == category]
            if not category_data.empty:
                # Take top items by relevance score if available
                if 'relevance_score' in category_data.columns:
                    category_data = category_data.nlargest(target_count, 'relevance_score')
                else:
                    category_data = category_data.head(target_count)
                balanced_data.append(category_data)
        
        if balanced_data:
            return pd.concat(balanced_data, ignore_index=True)
        return df


# Initialize the enhanced fetcher
intelligence_fetcher = MultiSourceIntelligenceFetcher()

# Create alias for compatibility
class EnhancedFetcher(MultiSourceIntelligenceFetcher):
    """Alias for compatibility with main application"""
    pass

# Legacy compatibility functions
def fetch_news_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """Fetch comprehensive news data"""
    return intelligence_fetcher._fetch_enhanced_news(topic, days_back)

def fetch_twitter_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """Fetch social media intelligence"""
    return intelligence_fetcher._fetch_social_intelligence(topic, days_back)

def fetch_all_data(topic: str, days_back: int = 7) -> pd.DataFrame:
    """
    🚀 MAIN FUNCTION: Fetch comprehensive 360° intelligence
    
    This function provides:
    - International news coverage
    - VIP statements and official content  
    - Public sentiment from social media
    - Policy analysis and reactions
    - Advanced relevance filtering
    """
    return intelligence_fetcher.fetch_comprehensive_intelligence(topic, days_back)

if __name__ == "__main__":
    # Test the enhanced system
    test_topic = "Narendra Modi"
    print(f"🚀 Testing comprehensive intelligence fetch for: {test_topic}")
    
    test_data = fetch_all_data(test_topic, 7)
    
    if not test_data.empty:
        print(f"\n✅ Total data points: {len(test_data)}")
        print(f"\n📊 Source distribution:")
        if 'category' in test_data.columns:
            print(test_data['category'].value_counts())
        print(f"\n📰 Source breakdown:")
        print(test_data['source'].value_counts().head(10))
        
        print(f"\n🎯 Sample results:")
        for idx, row in test_data.head(5).iterrows():
            print(f"- [{row['source']}] {row['text'][:100]}...")
    else:
        print("❌ No data fetched")
