"""
UNIVERSAL NEWSAPI FETCHER
Single API call system with enhanced universal source coverage

Key Features:
1. Single NewsAPI call per query (maximum efficiency)
2. Enhanced query construction using Wikidata contexts
3. Universal source coverage (ALL NewsAPI sources, not restricted)
4. Intelligent query optimization for maximum relevance

Process:
Wikidata Contexts → Enhanced Query Construction → Single Universal NewsAPI Call → Smart Filtering
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import config
import time
from urllib.parse import quote

logger = logging.getLogger(__name__)

class UniversalNewsAPIFetcher:
    """
    🌐 UNIVERSAL NEWSAPI FETCHER
    Single API call with maximum source coverage and intelligent query enhancement
    """
    
    def __init__(self):
        self.api_key = config.NEWSAPI_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UniversalNewsAnalyzer/1.0'
        })
        
        # Query optimization parameters
        self.max_articles_per_call = 100  # Maximum articles from single call
        self.relevance_threshold = 60     # Minimum relevance for inclusion
        
        logger.info("🌐 Universal NewsAPI Fetcher initialized")
    
    def fetch_universal_content(self, query: str, domain_contexts: Dict, days_back: int = 7) -> pd.DataFrame:
        """
        MAIN FETCHING FUNCTION
        Single enhanced NewsAPI call with universal source coverage
        """
        logger.info(f"🚀 Starting universal fetch for: {query}")
        
        try:
            # Step 1: Construct enhanced query using Wikidata contexts
            enhanced_query = self._construct_enhanced_query(query, domain_contexts)
            logger.info(f"📝 Enhanced query: {enhanced_query}")
            
            # Step 2: Execute single universal NewsAPI call
            articles = self._execute_universal_api_call(enhanced_query, days_back)
            logger.info(f"📡 Raw articles fetched: {len(articles)}")
            
            # Step 3: Smart filtering and relevance scoring
            filtered_articles = self._apply_smart_filtering(articles, query, domain_contexts)
            logger.info(f"✅ Relevant articles after filtering: {len(filtered_articles)}")
            
            # Step 4: Convert to DataFrame with enhanced metadata
            df = self._convert_to_dataframe(filtered_articles, query, domain_contexts)
            
            # Step 5: Final optimization and ranking
            if not df.empty:
                df = self._optimize_and_rank_results(df, query, domain_contexts)
            
            logger.info(f"🎯 Final results: {len(df)} high-quality articles")
            if not df.empty:
                sources = df['source'].value_counts()
                logger.info(f"📊 Source distribution: {dict(sources.head())}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in universal fetch: {e}")
            return pd.DataFrame(columns=['timestamp', 'text', 'source', 'relevance_score', 'url', 'title'])
    
    def _construct_enhanced_query(self, query: str, domain_contexts: Dict) -> str:
        """
        Construct enhanced query using Wikidata contexts for maximum relevance
        """
        # Get domain information
        primary_domain = domain_contexts.get('primary_domain', 'general')
        sentiment_contexts = domain_contexts.get('sentiment_contexts', [])
        search_enhancements = domain_contexts.get('search_enhancement_terms', [])
        
        # Base query with entity
        base_query = f'"{query}"'
        
        # Add high-value sentiment contexts (top 5)
        top_contexts = sentiment_contexts[:5]
        context_queries = [f'"{query}" {context}' for context in top_contexts]
        
        # Add domain enhancement terms
        top_enhancements = search_enhancements[:3]
        enhancement_queries = [f'"{query}" {enhancement}' for enhancement in top_enhancements]
        
        # Combine into comprehensive query using OR logic
        all_query_parts = [base_query] + context_queries + enhancement_queries
        
        # Construct final query with proper OR syntax
        enhanced_query = ' OR '.join(all_query_parts[:8])  # Limit to 8 parts to avoid query length issues
        
        return enhanced_query
    
    def _execute_universal_api_call(self, enhanced_query: str, days_back: int) -> List[Dict]:
        """
        Execute single NewsAPI call with universal source coverage
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Universal NewsAPI parameters (NO source restrictions)
            params = {
                'q': enhanced_query,
                'apiKey': self.api_key,
                'sortBy': 'relevancy',  # Sort by relevance for better quality
                'pageSize': self.max_articles_per_call,
                'language': 'en',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
                # NO 'sources' or 'domains' parameter = ALL SOURCES INCLUDED
            }
            
            logger.info(f"📡 Making universal API call (ALL sources included)")
            response = self.session.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    total_results = data.get('totalResults', 0)
                    
                    logger.info(f"✅ API Success: {len(articles)} articles from {total_results} total matches")
                    return articles
                else:
                    logger.error(f"❌ API Error: {data.get('message', 'Unknown error')}")
                    return []
            
            elif response.status_code == 429:
                logger.error("❌ API Rate limit exceeded")
                return []
            else:
                logger.error(f"❌ API HTTP Error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Exception in API call: {e}")
            return []
    
    def _apply_smart_filtering(self, articles: List[Dict], query: str, domain_contexts: Dict) -> List[Dict]:
        """
        Apply intelligent filtering to retain only relevant articles
        """
        if not articles:
            return []
        
        filtered_articles = []
        query_lower = query.lower()
        sentiment_contexts = domain_contexts.get('sentiment_contexts', [])
        
        for article in articles:
            try:
                # Basic article validation
                title = article.get('title', '').strip()
                description = article.get('description', '').strip()
                
                if not title or len(title) < 10:
                    continue
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    article, query_lower, sentiment_contexts
                )
                
                # Apply relevance threshold
                if relevance_score >= self.relevance_threshold:
                    article['relevance_score'] = relevance_score
                    filtered_articles.append(article)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error filtering article: {e}")
                continue
        
        # Sort by relevance score
        filtered_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return filtered_articles
    
    def _calculate_relevance_score(self, article: Dict, query_lower: str, sentiment_contexts: List[str]) -> float:
        """
        Calculate relevance score for an article
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = f"{title} {description}"
        
        relevance_score = 0.0
        
        # ENTITY RELEVANCE (40 points max)
        query_words = [word for word in query_lower.split() if len(word) > 2]
        
        # Exact query match in title (highest value)
        if query_lower in title:
            relevance_score += 35
        elif len(query_words) > 1 and all(word in title for word in query_words):
            relevance_score += 30
        elif any(word in title for word in query_words):
            relevance_score += 25
        
        # Query match in description
        if query_lower in description:
            relevance_score += 15
        elif any(word in description for word in query_words):
            relevance_score += 10
        
        # SENTIMENT CONTEXT RELEVANCE (30 points max)
        context_matches = sum(1 for context in sentiment_contexts if context.lower() in content)
        relevance_score += min(context_matches * 5, 30)
        
        # CONTENT QUALITY (20 points max)
        content_quality = 0
        if len(title) > 20 and description and len(description) > 30:
            content_quality += 15
        elif len(title) > 15:
            content_quality += 10
        
        # Bonus for detailed content
        if description and len(description) > 100:
            content_quality += 5
        
        relevance_score += min(content_quality, 20)
        
        # SENTIMENT INDICATORS (10 points max)
        sentiment_indicators = [
            'reaction', 'response', 'opinion', 'analysis', 'review', 'criticism',
            'praise', 'impact', 'effect', 'performance', 'announcement', 'statement'
        ]
        
        sentiment_matches = sum(1 for indicator in sentiment_indicators if indicator in content)
        relevance_score += min(sentiment_matches * 2, 10)
        
        # PENALTIES
        # Penalty for irrelevant content
        irrelevant_terms = [
            'advertisement', 'sponsored', 'shopping', 'discount', 'sale',
            'recipe', 'weather', 'horoscope', 'sports score'
        ]
        
        penalty = sum(15 for term in irrelevant_terms if term in content)
        relevance_score -= penalty
        
        return max(0, min(100, relevance_score))
    
    def _convert_to_dataframe(self, articles: List[Dict], query: str, domain_contexts: Dict) -> pd.DataFrame:
        """
        Convert filtered articles to DataFrame with enhanced metadata
        """
        if not articles:
            return pd.DataFrame(columns=['timestamp', 'text', 'source', 'relevance_score', 'url', 'title'])
        
        df_data = []
        
        for article in articles:
            try:
                # Combine title and description for analysis
                title = article.get('title', '').strip()
                description = article.get('description', '').strip()
                
                # Create comprehensive text content
                text_content = title
                if description and len(description) > 10:
                    text_content += f". {description}"
                
                # Parse timestamp
                published_at = article.get('publishedAt', '')
                try:
                    timestamp = pd.to_datetime(published_at).tz_localize(None)
                except:
                    timestamp = pd.to_datetime('now')
                
                # Extract source information
                source_info = article.get('source', {})
                source_name = source_info.get('name', 'Unknown')
                
                df_data.append({
                    'timestamp': timestamp,
                    'text': text_content,
                    'title': title,
                    'description': description,
                    'source': f"News-{source_name}",
                    'relevance_score': article.get('relevance_score', 0),
                    'url': article.get('url', ''),
                    'domain': domain_contexts.get('primary_domain', 'general'),
                    'query': query
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing article: {e}")
                continue
        
        return pd.DataFrame(df_data)
    
    def _optimize_and_rank_results(self, df: pd.DataFrame, query: str, domain_contexts: Dict) -> pd.DataFrame:
        """
        Final optimization and ranking of results
        """
        if df.empty:
            return df
        
        # Remove duplicates based on title similarity
        df = self._remove_duplicate_articles(df)
        
        # Add quality metrics
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Calculate final ranking score
        df['ranking_score'] = (
            df['relevance_score'] * 0.7 +  # Relevance is most important
            (df['text_length'] / 500) * 10 +  # Bonus for detailed content
            (df['word_count'] / 50) * 10    # Bonus for substantial content
        )
        
        # Sort by ranking score
        df = df.sort_values(['ranking_score', 'timestamp'], ascending=[False, False])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Add rank column
        df['content_rank'] = range(1, len(df) + 1)
        
        return df
    
    def _remove_duplicate_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate articles based on title similarity
        """
        if df.empty:
            return df
        
        # Simple duplicate removal based on title
        seen_titles = set()
        unique_indices = []
        
        for idx, title in enumerate(df['title']):
            title_clean = title.lower().strip()
            
            # Check for exact duplicates
            if title_clean not in seen_titles:
                seen_titles.add(title_clean)
                unique_indices.append(idx)
        
        return df.iloc[unique_indices].reset_index(drop=True)
    
    def get_source_coverage_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about source coverage
        """
        if df.empty:
            return {'total_sources': 0, 'total_articles': 0}
        
        source_stats = df['source'].value_counts()
        
        return {
            'total_sources': len(source_stats),
            'total_articles': len(df),
            'source_distribution': dict(source_stats.head(10)),
            'average_relevance': df['relevance_score'].mean(),
            'top_sources': list(source_stats.head(5).index)
        }

# Initialize the universal fetcher
universal_fetcher = UniversalNewsAPIFetcher()

def fetch_universal_news_single_call(query: str, domain_contexts: Dict, days_back: int = 7) -> pd.DataFrame:
    """
    Main function for universal news fetching with single API call
    """
    return universal_fetcher.fetch_universal_content(query, domain_contexts, days_back)

# Testing function
if __name__ == "__main__":
    # Test with mock domain contexts
    test_query = "Tesla"
    test_contexts = {
        'primary_domain': 'automotive',
        'entity_type': 'organization',
        'sentiment_contexts': [
            'financial performance', 'stock price', 'market reaction', 'earnings',
            'electric vehicle', 'competition', 'innovation', 'production'
        ],
        'search_enhancement_terms': ['automotive', 'business', 'financial', 'innovation', 'market']
    }
    
    print(f"🚀 Testing Universal NewsAPI Fetcher")
    print(f"Query: {test_query}")
    print(f"Domain: {test_contexts['primary_domain']}")
    
    result_df = fetch_universal_news_single_call(test_query, test_contexts, 7)
    
    print(f"\n📊 Results: {len(result_df)} articles")
    if not result_df.empty:
        print(f"Source distribution:")
        print(result_df['source'].value_counts().head())
        print(f"\nAverage relevance: {result_df['relevance_score'].mean():.1f}")
        print(f"Top articles:")
        for idx, row in result_df.head(3).iterrows():
            print(f"  {idx+1}. {row['title'][:80]}... (Relevance: {row['relevance_score']:.1f})")
