"""
Professional Phrase Analysis and Advanced Insights Module
Extract meaningful phrases and contextual sentiment patterns
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import logging
from typing import List, Dict, Tuple, Set
import re

# Try to import NLTK functions, with fallbacks
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Fallback tokenization functions
    def sent_tokenize(text):
        return re.split(r'[.!?]+', text)
    
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def ngrams(words, n):
        return [words[i:i+n] for i in range(len(words)-n+1)]

import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalPhraseAnalyzer:
    """Advanced phrase extraction and sentiment context analysis"""
    
    def __init__(self):
        """Initialize the phrase analyzer"""
        if NLTK_AVAILABLE:
            try:
                import nltk
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"NLTK setup issue: {e}")
                self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        else:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            self.nlp = None
            logger.warning("SpaCy model not available for advanced phrase analysis")
    
    def extract_key_phrases(self, df: pd.DataFrame, sentiment_filter: str = None) -> List[Tuple[str, int, float]]:
        """
        Extract meaningful key phrases with their frequency and average sentiment
        
        Returns:
            List of (phrase, frequency, avg_sentiment) tuples
        """
        if df.empty:
            return []
        
        # Filter by sentiment if specified
        if sentiment_filter and 'sentiment_label' in df.columns:
            df = df[df['sentiment_label'] == sentiment_filter]
        
        # Combine all text
        all_text = ' '.join(df['text'].astype(str).tolist())
        
        # Extract phrases using different methods
        phrases = []
        
        # Method 1: Named entities and noun phrases
        if self.nlp:
            phrases.extend(self._extract_spacy_phrases(all_text))
        
        # Method 2: N-grams with filtering
        phrases.extend(self._extract_ngram_phrases(all_text))
        
        # Method 3: Topic-specific patterns
        phrases.extend(self._extract_pattern_phrases(all_text))
        
        # Calculate phrase statistics
        phrase_stats = self._calculate_phrase_stats(phrases, df)
        
        # Sort by relevance score (combination of frequency and sentiment strength)
        phrase_stats.sort(key=lambda x: x[3], reverse=True)  # Sort by relevance score
        
        return [(phrase, freq, sentiment, relevance) for phrase, freq, sentiment, relevance in phrase_stats[:20]]
    
    def _extract_spacy_phrases(self, text: str) -> List[str]:
        """Extract noun phrases and named entities using SpaCy"""
        if not self.nlp:
            return []
        
        phrases = []
        doc = self.nlp(text)
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY']:
                phrases.append(ent.text.strip())
        
        # Noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2 and len(chunk.text) > 5:
                phrases.append(chunk.text.strip())
        
        return phrases
    
    def _extract_ngram_phrases(self, text: str) -> List[str]:
        """Extract meaningful n-grams"""
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        phrases = []
        
        # Bigrams and trigrams
        for n in [2, 3]:
            n_grams = list(ngrams(words, n))
            for gram in n_grams:
                phrase = ' '.join(gram)
                if len(phrase) > 6 and not phrase.isdigit():
                    phrases.append(phrase)
        
        return phrases
    
    def _extract_pattern_phrases(self, text: str) -> List[str]:
        """Extract phrases using patterns and keywords"""
        phrases = []
        
        # Common patterns for important phrases
        patterns = [
            r'\b(?:new|latest|recent|upcoming)\s+\w+(?:\s+\w+){0,2}',
            r'\b\w+(?:\s+\w+){0,2}\s+(?:announced|launched|released|unveiled)',
            r'\b(?:CEO|president|director|minister)\s+\w+(?:\s+\w+)?',
            r'\b\w+\s+(?:policy|strategy|plan|initiative|project)',
            r'\b(?:stock|shares|market|price|value)\s+\w+(?:\s+\w+)?',
            r'\b\w+(?:\s+\w+){0,2}\s+(?:meeting|conference|summit|deal)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend([match.strip() for match in matches])
        
        return phrases
    
    def _calculate_phrase_stats(self, phrases: List[str], df: pd.DataFrame) -> List[Tuple[str, int, float, float]]:
        """Calculate statistics for each phrase"""
        phrase_counter = Counter(phrases)
        phrase_stats = []
        
        for phrase, frequency in phrase_counter.items():
            if frequency >= 2 and len(phrase) > 5:  # Minimum frequency and length
                # Calculate average sentiment for this phrase
                phrase_texts = df[df['text'].str.contains(phrase, case=False, na=False)]
                
                if not phrase_texts.empty and 'compound' in phrase_texts.columns:
                    avg_sentiment = phrase_texts['compound'].mean()
                    sentiment_strength = abs(avg_sentiment)
                    
                    # Relevance score: frequency * sentiment_strength
                    relevance = frequency * (1 + sentiment_strength)
                    
                    phrase_stats.append((phrase, frequency, avg_sentiment, relevance))
        
        return phrase_stats
    
    def generate_insight_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate professional insights summary"""
        if df.empty:
            return {}
        
        insights = {}
        
        # Overall sentiment trends
        if 'compound' in df.columns:
            insights['overall_sentiment'] = {
                'mean': df['compound'].mean(),
                'std': df['compound'].std(),
                'trend': self._calculate_trend(df)
            }
        
        # Key phrases by sentiment
        insights['key_phrases'] = {
            'positive': self.extract_key_phrases(df, 'Positive')[:10],
            'negative': self.extract_key_phrases(df, 'Negative')[:10],
            'neutral': self.extract_key_phrases(df, 'Neutral')[:10]
        }
        
        # Source analysis
        if 'source' in df.columns:
            source_sentiment = df.groupby('source')['compound'].agg(['mean', 'count']).round(3)
            insights['source_analysis'] = source_sentiment.to_dict('index')
        
        # Time-based patterns
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_sentiment = df.groupby('hour')['compound'].mean()
            insights['temporal_patterns'] = {
                'peak_positive_hour': hourly_sentiment.idxmax(),
                'peak_negative_hour': hourly_sentiment.idxmin(),
                'hourly_trends': hourly_sentiment.to_dict()
            }
        
        return insights
    
    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """Calculate sentiment trend over time"""
        if len(df) < 5:
            return "insufficient_data"
        
        df_sorted = df.sort_values('timestamp')
        recent_sentiment = df_sorted.tail(len(df)//3)['compound'].mean()
        earlier_sentiment = df_sorted.head(len(df)//3)['compound'].mean()
        
        diff = recent_sentiment - earlier_sentiment
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def analyze_phrases(self, texts: List[str]) -> Dict[str, any]:
        """
        Analyze phrases from a list of texts - compatibility method for main app
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dict with phrase analysis results
        """
        # Create a temporary DataFrame for analysis
        df = pd.DataFrame({
            'text': texts,
            'timestamp': pd.date_range('2024-01-01', periods=len(texts)),
            'compound': [0.0] * len(texts),  # Placeholder values
            'sentiment_label': ['Neutral'] * len(texts)
        })
        
        # Extract phrases and generate insights
        key_phrases_tuples = self.extract_key_phrases(df)
        insights = self.generate_insight_summary(df)
        
        # Convert tuples to dictionaries for compatibility with visualizer
        key_phrases_dicts = []
        for phrase, freq, sentiment, relevance in key_phrases_tuples:
            key_phrases_dicts.append({
                'phrase': phrase,
                'frequency': freq,
                'avg_sentiment': sentiment,
                'relevance_score': relevance
            })
        
        return {
            'key_phrases': key_phrases_dicts,
            'insights': insights,
            'total_phrases': len(key_phrases_dicts),
            'unique_phrases': len(set([p['phrase'] for p in key_phrases_dicts]))
        }

def create_professional_phrase_analysis(df: pd.DataFrame) -> Dict[str, any]:
    """Main function to create professional phrase analysis"""
    analyzer = ProfessionalPhraseAnalyzer()
    return analyzer.generate_insight_summary(df)

def get_contextual_phrases(df: pd.DataFrame, sentiment: str) -> List[Dict[str, any]]:
    """Get phrases with context for a specific sentiment"""
    analyzer = ProfessionalPhraseAnalyzer()
    phrases = analyzer.extract_key_phrases(df, sentiment)
    
    contextual_phrases = []
    for phrase, freq, avg_sentiment, relevance in phrases:
        # Find example texts containing this phrase
        examples = df[df['text'].str.contains(phrase, case=False, na=False)]['text'].head(3).tolist()
        
        contextual_phrases.append({
            'phrase': phrase,
            'frequency': freq,
            'avg_sentiment': round(avg_sentiment, 3),
            'relevance_score': round(relevance, 2),
            'examples': examples
        })
    
    return contextual_phrases

if __name__ == "__main__":
    # Test the phrase analyzer
    sample_data = pd.DataFrame({
        'text': [
            'Tesla stock price surged after new Model Y announcement',
            'Elon Musk announced new Tesla factory in Berlin',
            'Tesla autopilot feature receives criticism from safety experts',
            'New Tesla charging stations announced across California',
            'Tesla quarterly earnings beat expectations significantly'
        ],
        'compound': [0.6, 0.4, -0.5, 0.3, 0.7],
        'sentiment_label': ['Positive', 'Positive', 'Negative', 'Positive', 'Positive'],
        'timestamp': pd.date_range('2024-01-01', periods=5),
        'source': ['News-Reuters'] * 5
    })
    
    analyzer = ProfessionalPhraseAnalyzer()
    insights = analyzer.generate_insight_summary(sample_data)
    print("Professional Insights:", insights)
