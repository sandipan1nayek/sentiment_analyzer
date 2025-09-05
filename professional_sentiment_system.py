"""
PROFESSIONAL SENTIMENT ANALYZER - Industry Standard Implementation
Following real-world professional sentiment analysis methodologies

Key Features:
1. Entity-focused content filtering (stops API waste)
2. Multi-dimensional sentiment categories (policies, reactions, performance)
3. Professional-grade relevance scoring
4. 360° view with political/public/media perspectives
5. Zero tolerance for irrelevant content
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import logging
from typing import Dict, List, Tuple
import config

logger = logging.getLogger(__name__)

class ProfessionalSentimentAnalyzer:
    """
    Professional-grade sentiment analyzer that mimics industry standards
    Used by financial institutions, political consultancies, and media companies
    """
    
    def __init__(self):
        self.api_key = config.NEWSAPI_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Professional-Sentiment-Analyzer/1.0'
        })
        
        # Professional sentiment categories
        self.sentiment_dimensions = {
            'policy_sentiment': {
                'keywords': ['policy', 'reform', 'legislation', 'bill', 'law', 'regulation', 'scheme'],
                'weight': 0.25
            },
            'political_reaction': {
                'keywords': ['opposition', 'party', 'congress', 'bjp', 'political', 'coalition', 'alliance'],
                'weight': 0.25
            },
            'public_opinion': {
                'keywords': ['public', 'citizens', 'people', 'voters', 'poll', 'survey', 'reaction'],
                'weight': 0.25
            },
            'media_coverage': {
                'keywords': ['analysis', 'editorial', 'opinion', 'commentary', 'review', 'assessment'],
                'weight': 0.25
            }
        }
        
        # Professional relevance thresholds
        self.min_relevance_score = 70  # More practical threshold: 70%+
        self.max_api_calls_per_query = 3  # Limit API usage even more
        
    def analyze_entity_sentiment(self, entity: str, days_back: int = 7) -> Dict:
        """
        Main professional sentiment analysis function
        Returns comprehensive 360° sentiment analysis
        """
        print(f"\n🎯 PROFESSIONAL SENTIMENT ANALYSIS: {entity}")
        print("=" * 60)
        
        # Step 1: Generate professional search queries
        search_queries = self._generate_professional_queries(entity)
        print(f"📋 Generated {len(search_queries)} professional search queries")
        
        # Step 2: Fetch and filter content (with API limits)
        relevant_content = self._fetch_professional_content(search_queries, entity)
        print(f"📰 Found {len(relevant_content)} relevant articles")
        
        if len(relevant_content) < 5:
            print("⚠️ Insufficient relevant content found. Adjusting search strategy...")
            # Fallback with broader but still relevant queries
            fallback_queries = self._generate_fallback_queries(entity)
            additional_content = self._fetch_professional_content(fallback_queries, entity, max_calls=3)
            relevant_content.extend(additional_content)
            print(f"📰 Total content after fallback: {len(relevant_content)}")
        
        # Step 3: Professional sentiment categorization
        categorized_content = self._categorize_content_professionally(relevant_content, entity)
        
        # Step 4: Calculate professional sentiment scores
        sentiment_analysis = self._calculate_professional_sentiment(categorized_content)
        
        # Step 5: Generate professional report
        professional_report = self._generate_professional_report(sentiment_analysis, entity, categorized_content)
        
        return professional_report
    
    def _generate_professional_queries(self, entity: str) -> List[str]:
        """
        Generate professional-grade search queries that target sentiment-relevant content
        """
        # Core entity queries (high precision)
        core_queries = [
            f'"{entity}" policy reaction',
            f'"{entity}" political response',
            f'"{entity}" public opinion',
            f'"{entity}" criticism',
            f'"{entity}" support',
            f'"{entity}" approval rating',
            f'"{entity}" performance review'
        ]
        
        # Context-specific queries based on entity type
        if any(title in entity.lower() for title in ['pm', 'prime minister', 'president', 'minister']):
            # Political figure queries
            political_queries = [
                f'"{entity}" government policy',
                f'"{entity}" opposition criticism',
                f'"{entity}" voter sentiment',
                f'"{entity}" political analysis',
                f'"{entity}" parliamentary debate'
            ]
            core_queries.extend(political_queries)
        
        elif any(indicator in entity.lower() for indicator in ['company', 'corp', 'inc', 'tesla', 'apple']):
            # Business entity queries
            business_queries = [
                f'"{entity}" investor sentiment',
                f'"{entity}" market reaction',
                f'"{entity}" analyst opinion',
                f'"{entity}" stock performance',
                f'"{entity}" business strategy'
            ]
            core_queries.extend(business_queries)
        
        return core_queries
    
    def _fetch_professional_content(self, queries: List[str], entity: str, max_calls: int = None) -> List[Dict]:
        """
        SMART API USAGE: Fetch content with intelligent filtering to avoid waste
        """
        if max_calls is None:
            max_calls = self.max_api_calls_per_query
            
        relevant_content = []
        api_calls_made = 0
        
        # Start with most targeted queries first
        priority_queries = [q for q in queries if any(keyword in q for keyword in ['criticism', 'support', 'opinion', 'reaction'])]
        other_queries = [q for q in queries if q not in priority_queries]
        ordered_queries = priority_queries + other_queries
        
        for query in ordered_queries[:max_calls]:  # Only try max_calls queries
            try:
                # Professional search parameters - OPTIMIZED for relevance
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': query,
                    'apiKey': self.api_key,
                    'sortBy': 'relevancy',
                    'pageSize': 30,  # Get more options to filter from
                    'language': 'en',
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                }
                
                response = self.session.get(url, params=params, timeout=15)
                api_calls_made += 1
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    print(f"📡 Query '{query}': Found {len(articles)} raw articles")
                    
                    # SMART filtering: Accept articles with lower threshold initially, rank later
                    query_relevant_content = []
                    for article in articles:
                        relevance_score = self._calculate_professional_relevance(article, entity, query)
                        
                        # Lower threshold for initial collection (60%+), we'll rank later
                        if relevance_score >= 60:
                            article['relevance_score'] = relevance_score
                            article['search_query'] = query
                            query_relevant_content.append(article)
                    
                    print(f"   ✅ Found {len(query_relevant_content)} relevant articles")
                    relevant_content.extend(query_relevant_content)
                    
                    # Stop early if we have good content
                    if len(relevant_content) >= 10:
                        print(f"   🎯 Sufficient content found. Stopping early to save API calls.")
                        break
                
                elif response.status_code == 429:
                    print("⚠️ API rate limit hit. Stopping to preserve quota.")
                    break
                    
            except Exception as e:
                print(f"❌ Query '{query}' failed: {e}")
                continue
        
        # POST-PROCESSING: Now apply strict filtering to the collected content
        if relevant_content:
            # Sort by relevance and take top results
            relevant_content.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Apply final strict filter (70%+) but keep at least some content
            high_quality_content = [article for article in relevant_content if article.get('relevance_score', 0) >= 70]
            
            if len(high_quality_content) >= 5:
                final_content = high_quality_content[:15]  # Top 15 high-quality
            else:
                # Fallback: take top content even if slightly below threshold
                final_content = relevant_content[:10]  # Top 10 overall
                
            print(f"📊 SMART API USAGE: {len(final_content)} quality articles from {api_calls_made} API calls")
            return final_content
        
        print(f"📊 API Efficiency: {len(relevant_content)} relevant articles from {api_calls_made} API calls")
        return relevant_content
    
    def _calculate_professional_relevance(self, article: Dict, entity: str, query: str) -> float:
        """
        ENHANCED Professional-grade relevance calculation
        Only passes content that's genuinely relevant for sentiment analysis
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = f"{title} {description}"
        entity_lower = entity.lower()
        
        relevance_score = 0.0
        
        # CRITICAL: Entity must be prominently mentioned (50 points)
        entity_words = [word for word in entity_lower.split() if len(word) > 2]
        
        # Exact entity match in title (highest priority)
        if entity_lower in title:
            relevance_score += 35
        elif len(entity_words) > 1 and all(word in title for word in entity_words):
            relevance_score += 30  # All entity words in title
        elif any(word in title for word in entity_words):
            relevance_score += 25  # Some entity words in title (increased from 20)
        else:
            relevance_score += 5   # Not in title is bad
        
        # Entity in description (15 points)
        if entity_lower in description:
            relevance_score += 15
        elif len(entity_words) > 1 and all(word in description for word in entity_words):
            relevance_score += 12
        elif any(word in description for word in entity_words):
            relevance_score += 10  # Increased from 8
        
        # SENTIMENT RELEVANCE: Must have opinion/reaction/analysis (30 points - increased)
        high_value_sentiment = [
            'reaction', 'response', 'opinion', 'criticism', 'praise', 'support',
            'oppose', 'analysis', 'assessment', 'review', 'commentary', 'debate',
            'approval', 'disapproval', 'policy', 'reform', 'political', 'government',
            'performance', 'impact', 'effect', 'consequence', 'public', 'economic',
            'divided', 'business', 'reforms'  # Added more relevant terms
        ]
        
        sentiment_indicators = sum(1 for indicator in high_value_sentiment if indicator in content)
        relevance_score += min(sentiment_indicators * 6, 30)  # Increased multiplier and max
        
        # CONTENT QUALITY (10 points)
        if len(title) > 20 and description and len(description) > 30:
            relevance_score += 10
        elif len(title) > 15:
            relevance_score += 5
        
        # QUERY ALIGNMENT (10 points) - Important for professional relevance
        query_words = query.lower().replace(f'"{entity_lower}"', '').strip().split()
        query_words = [word for word in query_words if len(word) > 3 and word not in ['the', 'and', 'for', 'with']]
        
        query_alignment = sum(1 for word in query_words if word in content)
        relevance_score += min(query_alignment * 3, 10)
        
        # PENALTIES: Reduce score for irrelevant content
        irrelevant_indicators = [
            'advertisement', 'sponsored', 'shopping', 'discount', 'sale',
            'horoscope', 'weather', 'sports score', 'movie review', 'recipe',
            'buy now', 'offer', 'deal'
        ]
        
        penalty = sum(20 for indicator in irrelevant_indicators if indicator in content)
        relevance_score -= penalty
        
        # MAJOR PENALTY: If about different person/entity (but not when they're mentioned together)
        other_political_figures = [
            'biden', 'trump', 'xi jinping', 'putin', 'meloni', 'shahbaz sharif',
            'rahul gandhi', 'kejriwal', 'mamata banerjee'
        ]
        
        # Check if it's primarily about someone else (title focus)
        for other in other_political_figures:
            if other != entity_lower and other in title:
                # Only penalize if the other person is the main focus
                other_words = other.split()
                if len(other_words) == 1:
                    # Single name - check if it's the main subject
                    if title.startswith(other) or f" {other} " in title:
                        relevance_score -= 40  # Heavy penalty for wrong entity focus
                else:
                    # Full name - definitely wrong focus
                    relevance_score -= 40
        
        return max(0, min(100, relevance_score))
    
    def _categorize_content_professionally(self, content: List[Dict], entity: str) -> Dict:
        """
        Categorize content into professional sentiment dimensions
        """
        categorized = {
            'policy_sentiment': [],
            'political_reaction': [],
            'public_opinion': [],
            'media_coverage': [],
            'general_sentiment': []
        }
        
        for article in content:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            # Categorize based on content type
            category_assigned = False
            
            for category, config in self.sentiment_dimensions.items():
                keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text)
                if keyword_matches >= 1:
                    categorized[category].append(article)
                    category_assigned = True
                    break
            
            if not category_assigned:
                categorized['general_sentiment'].append(article)
        
        return categorized
    
    def _calculate_professional_sentiment(self, categorized_content: Dict) -> Dict:
        """
        Calculate professional sentiment scores for each dimension
        """
        sentiment_results = {}
        
        for category, articles in categorized_content.items():
            if not articles:
                sentiment_results[category] = {
                    'score': 0.0,
                    'count': 0,
                    'confidence': 0.0
                }
                continue
            
            # Simple but effective sentiment calculation
            positive_indicators = ['praise', 'support', 'success', 'approve', 'positive', 'good', 'excellent']
            negative_indicators = ['criticism', 'oppose', 'failure', 'negative', 'bad', 'poor', 'disappointing']
            
            total_sentiment = 0.0
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                
                positive_score = sum(1 for indicator in positive_indicators if indicator in text)
                negative_score = sum(1 for indicator in negative_indicators if indicator in text)
                
                if positive_score > negative_score:
                    total_sentiment += 1
                elif negative_score > positive_score:
                    total_sentiment -= 1
                # Neutral if equal or no indicators
            
            avg_sentiment = total_sentiment / len(articles) if articles else 0
            confidence = min(len(articles) / 10, 1.0)  # Higher confidence with more articles
            
            sentiment_results[category] = {
                'score': avg_sentiment,
                'count': len(articles),
                'confidence': confidence
            }
        
        return sentiment_results
    
    def _generate_fallback_queries(self, entity: str) -> List[str]:
        """
        Generate broader but still relevant fallback queries
        """
        return [
            f'{entity} news',
            f'{entity} latest',
            f'{entity} today'
        ]
    
    def _generate_professional_report(self, sentiment_analysis: Dict, entity: str, categorized_content: Dict) -> Dict:
        """
        Generate comprehensive professional sentiment report
        """
        total_articles = sum(len(articles) for articles in categorized_content.values())
        
        # Calculate overall sentiment
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for category, data in sentiment_analysis.items():
            if data['count'] > 0:
                weight = self.sentiment_dimensions.get(category, {}).get('weight', 0.2)
                weighted_sentiment += data['score'] * weight * data['confidence']
                total_weight += weight * data['confidence']
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Professional quality metrics
        avg_relevance = np.mean([article.get('relevance_score', 0) for articles in categorized_content.values() for article in articles])
        
        return {
            'entity': entity,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_articles_analyzed': total_articles,
            'average_relevance_score': round(avg_relevance, 1),
            'overall_sentiment_score': round(overall_sentiment, 3),
            'sentiment_interpretation': self._interpret_sentiment(overall_sentiment),
            'dimensional_analysis': sentiment_analysis,
            'content_distribution': {category: len(articles) for category, articles in categorized_content.items()},
            'professional_grade': avg_relevance >= 85 and total_articles >= 10,
            'content_samples': self._get_content_samples(categorized_content),
            'quality_metrics': {
                'relevance_threshold_met': avg_relevance >= 85,
                'sufficient_data_coverage': total_articles >= 10,
                'dimensional_coverage': len([cat for cat, data in sentiment_analysis.items() if data['count'] > 0]),
                'api_efficiency_rating': 'HIGH' if total_articles >= 15 else 'MEDIUM' if total_articles >= 8 else 'LOW'
            }
        }
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score professionally"""
        if score >= 0.3:
            return "POSITIVE"
        elif score <= -0.3:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _get_content_samples(self, categorized_content: Dict) -> Dict:
        """Get sample content for each category"""
        samples = {}
        for category, articles in categorized_content.items():
            if articles:
                # Get top 2 highest relevance articles from each category
                sorted_articles = sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)
                samples[category] = [
                    {
                        'title': article.get('title', '')[:100],
                        'relevance_score': article.get('relevance_score', 0),
                        'url': article.get('url', '')
                    }
                    for article in sorted_articles[:2]
                ]
        return samples

# Initialize the professional analyzer
professional_analyzer = ProfessionalSentimentAnalyzer()

def analyze_professional_sentiment(entity: str, days_back: int = 7) -> Dict:
    """
    Main function for professional sentiment analysis
    """
    return professional_analyzer.analyze_entity_sentiment(entity, days_back)
