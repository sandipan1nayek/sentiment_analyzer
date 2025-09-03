"""
Visualization module for sentiment analysis dashboard
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import streamlit as st
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SentimentVisualizer:
    """Class for creating sentiment analysis visualizations"""
    
    def __init__(self):
        """Initialize the visualizer"""
        self.colors = {
            'Positive': '#2E8B57',
            'Negative': '#DC143C', 
            'Neutral': '#4682B4'
        }
        logger.info("Sentiment Visualizer initialized")
    
    def create_time_series_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create time series chart of sentiment over time
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            go.Figure: Plotly figure object
        """
        if df.empty or 'timestamp' not in df.columns or 'compound' not in df.columns:
            return go.Figure().add_annotation(text="No data available for time series")
        
        # Resample data by hour for better visualization
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_hourly = df.set_index('timestamp').resample('H')['compound'].mean().reset_index()
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_hourly['timestamp'],
            y=df_hourly['compound'],
            mode='lines+markers',
            name='Average Sentiment',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Add horizontal lines for sentiment thresholds
        fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                     annotation_text="Positive Threshold")
        fig.add_hline(y=-0.05, line_dash="dash", line_color="red", 
                     annotation_text="Negative Threshold")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     annotation_text="Neutral")
        
        fig.update_layout(
            title="Sentiment Score Over Time",
            xaxis_title="Time",
            yaxis_title="Average Compound Sentiment Score",
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_sentiment_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart of sentiment distribution
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            go.Figure: Plotly figure object
        """
        if df.empty or 'sentiment_label' not in df.columns:
            return go.Figure().add_annotation(text="No data available for distribution")
        
        sentiment_counts = df['sentiment_label'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=[self.colors.get(label, '#888888') for label in sentiment_counts.index],
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_source_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create bar chart comparing sentiment by source
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            go.Figure: Plotly figure object
        """
        if df.empty or 'source' not in df.columns or 'sentiment_label' not in df.columns:
            return go.Figure().add_annotation(text="No data available for source comparison")
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(df['source'], df['sentiment_label'])
        
        fig = go.Figure()
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            if sentiment in cross_tab.columns:
                fig.add_trace(go.Bar(
                    x=cross_tab.index,
                    y=cross_tab[sentiment],
                    name=sentiment,
                    marker_color=self.colors.get(sentiment, '#888888')
                ))
        
        fig.update_layout(
            title="Sentiment Distribution by Source",
            xaxis_title="Data Source",
            yaxis_title="Count",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_wordcloud(self, text_data: str, title: str = "Word Cloud", 
                        color_scheme: str = 'viridis') -> plt.Figure:
        """
        Create word cloud from text data
        
        Args:
            text_data (str): Text data for word cloud
            title (str): Title for the word cloud
            color_scheme (str): Color scheme for the word cloud
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if not text_data or not text_data.strip():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available for word cloud', 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.title(title)
            return fig
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap=color_scheme,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text_data)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_entity_chart(self, entities_data: List[tuple], title: str = "Top Entities") -> go.Figure:
        """
        Create bar chart for top entities
        
        Args:
            entities_data (List[tuple]): List of (entity, count) tuples
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        if not entities_data:
            return go.Figure().add_annotation(text="No entities found")
        
        entities, counts = zip(*entities_data)
        
        fig = go.Figure([go.Bar(
            x=list(counts),
            y=list(entities),
            orientation='h',
            marker_color='lightblue'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Frequency",
            yaxis_title="Entity",
            height=max(400, len(entities) * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig

def create_summary_metrics(df: pd.DataFrame) -> Dict:
    """
    Create summary metrics for the dashboard
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment analysis results
        
    Returns:
        Dict: Summary metrics
    """
    if df.empty:
        return {
            'total_items': 0,
            'avg_sentiment': 0.0,
            'positive_percentage': 0.0,
            'negative_percentage': 0.0,
            'neutral_percentage': 0.0
        }
    
    total_items = len(df)
    avg_sentiment = df['compound'].mean() if 'compound' in df.columns else 0.0
    
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts()
        positive_pct = (sentiment_counts.get('Positive', 0) / total_items) * 100
        negative_pct = (sentiment_counts.get('Negative', 0) / total_items) * 100
        neutral_pct = (sentiment_counts.get('Neutral', 0) / total_items) * 100
    else:
        positive_pct = negative_pct = neutral_pct = 0.0
    
    return {
        'total_items': total_items,
        'avg_sentiment': avg_sentiment,
        'positive_percentage': positive_pct,
        'negative_percentage': negative_pct,
        'neutral_percentage': neutral_pct
    }

def filter_data_by_timerange(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Filter data by time range
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp column
        days (int): Number of days to look back
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if df.empty or 'timestamp' not in df.columns:
        return df
    
    cutoff_date = datetime.now() - timedelta(days=days)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df[df['timestamp'] >= cutoff_date].copy()

if __name__ == "__main__":
    # Test the visualizer
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'text': [f"Sample text {i}" for i in range(10)],
        'compound': [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.4, -0.1, 0.1],
        'sentiment_label': ['Positive', 'Negative', 'Positive', 'Negative', 'Neutral', 
                           'Positive', 'Negative', 'Positive', 'Negative', 'Positive'],
        'source': ['NewsAPI', 'Twitter'] * 5
    })
    
    visualizer = SentimentVisualizer()
    
    # Test time series chart
    time_fig = visualizer.create_time_series_chart(test_data)
    print("Time series chart created")
    
    # Test sentiment distribution
    dist_fig = visualizer.create_sentiment_distribution_chart(test_data)
    print("Distribution chart created")
    
    # Test summary metrics
    metrics = create_summary_metrics(test_data)
    print(f"Summary metrics: {metrics}")
