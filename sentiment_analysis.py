"""
Sentiment analysis module using VADER
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from typing import Dict, List
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analyzer using VADER"""
    
    def __init__(self):
        """Initialize the VADER sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER Sentiment Analyzer initialized")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary with sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score
        
        Args:
            compound_score (float): VADER compound score
            
        Returns:
            str: Sentiment label ('Positive', 'Negative', or 'Neutral')
        """
        if compound_score >= config.POSITIVE_THRESHOLD:
            return 'Positive'
        elif compound_score <= config.NEGATIVE_THRESHOLD:
            return 'Negative'
        else:
            return 'Neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for entire DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with 'text' column
            
        Returns:
            pd.DataFrame: DataFrame with sentiment scores and labels added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for sentiment analysis")
            return df
        
        logger.info(f"Starting sentiment analysis for {len(df)} texts")
        
        # Create a copy to avoid modifying original
        df_analyzed = df.copy()
        
        # Initialize sentiment columns
        df_analyzed['positive'] = 0.0
        df_analyzed['negative'] = 0.0
        df_analyzed['neutral'] = 0.0
        df_analyzed['compound'] = 0.0
        df_analyzed['sentiment_label'] = 'Neutral'
        
        # Analyze each text
        for idx, row in df_analyzed.iterrows():
            sentiment_scores = self.analyze_text(row['text'])
            
            df_analyzed.at[idx, 'positive'] = sentiment_scores['positive']
            df_analyzed.at[idx, 'negative'] = sentiment_scores['negative']
            df_analyzed.at[idx, 'neutral'] = sentiment_scores['neutral']
            df_analyzed.at[idx, 'compound'] = sentiment_scores['compound']
            df_analyzed.at[idx, 'sentiment_label'] = self.classify_sentiment(
                sentiment_scores['compound']
            )
        
        logger.info("Sentiment analysis completed")
        
        # Log sentiment distribution
        sentiment_counts = df_analyzed['sentiment_label'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
        return df_analyzed

def get_sentiment_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of sentiment analysis
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        Dict: Summary statistics
    """
    if df.empty or 'sentiment_label' not in df.columns:
        return {
            'total_texts': 0,
            'sentiment_distribution': {},
            'average_compound': 0.0,
            'most_positive': "",
            'most_negative': ""
        }
    
    sentiment_counts = df['sentiment_label'].value_counts().to_dict()
    
    # Get most positive and negative texts
    most_positive = ""
    most_negative = ""
    
    if 'compound' in df.columns:
        max_compound_idx = df['compound'].idxmax()
        min_compound_idx = df['compound'].idxmin()
        
        if pd.notna(max_compound_idx):
            most_positive = df.loc[max_compound_idx, 'text'][:100] + "..."
        if pd.notna(min_compound_idx):
            most_negative = df.loc[min_compound_idx, 'text'][:100] + "..."
    
    return {
        'total_texts': len(df),
        'sentiment_distribution': sentiment_counts,
        'average_compound': df['compound'].mean() if 'compound' in df.columns else 0.0,
        'most_positive': most_positive,
        'most_negative': most_negative
    }

def filter_by_sentiment(df: pd.DataFrame, sentiment: str) -> pd.DataFrame:
    """
    Filter DataFrame by sentiment label
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        sentiment (str): Sentiment to filter by ('Positive', 'Negative', 'Neutral')
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if df.empty or 'sentiment_label' not in df.columns:
        return pd.DataFrame()
    
    return df[df['sentiment_label'] == sentiment].copy()

if __name__ == "__main__":
    # Test the sentiment analyzer
    test_data = pd.DataFrame({
        'text': [
            "I love this product! It's amazing!",
            "This is terrible. I hate it.",
            "It's okay, nothing special.",
            "Best purchase ever! Highly recommend!",
            "Worst experience of my life."
        ]
    })
    
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_dataframe(test_data)
    
    print("Sentiment Analysis Results:")
    print(results[['text', 'compound', 'sentiment_label']])
    
    summary = get_sentiment_summary(results)
    print(f"\nSummary: {summary}")
