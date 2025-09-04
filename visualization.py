"""
Professional Visualization Module for Production-Grade Sentiment Analysis
Stable, clear, and informative visualizations with advanced analytics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional color scheme
PROFESSIONAL_COLORS = {
    'Positive': '#2E8B57',    # Sea Green
    'Negative': '#DC143C',    # Crimson
    'Neutral': '#4682B4',     # Steel Blue
    'background': '#FAFAFA',
    'grid': '#E0E0E0',
    'text': '#2C3E50'
}

class ProfessionalVisualizer:
    """Professional-grade visualization class with stable, clear charts"""
    
    def __init__(self):
        """Initialize with professional styling"""
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette([PROFESSIONAL_COLORS['Positive'], 
                        PROFESSIONAL_COLORS['Negative'], 
                        PROFESSIONAL_COLORS['Neutral']])
        
        # Configure plotly theme
        self.plotly_theme = {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'colorway': [PROFESSIONAL_COLORS['Positive'], 
                           PROFESSIONAL_COLORS['Negative'], 
                           PROFESSIONAL_COLORS['Neutral']]
            }
        }
        
        logger.info("Professional Visualizer initialized")
    
    def create_professional_time_series(self, df: pd.DataFrame) -> go.Figure:
        """Create a professional time series chart with trend analysis"""
        if df.empty or 'timestamp' not in df.columns or 'compound' not in df.columns:
            return self._create_no_data_figure("No data available for time series analysis")
        
        # Prepare data with proper time binning
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Determine appropriate time binning
        time_span = (df['timestamp'].max() - df['timestamp'].min()).days
        if time_span <= 1:
            freq = 'h'  # Hourly
            title_freq = "Hourly"
        elif time_span <= 7:
            freq = '3h'  # 3-hourly
            title_freq = "3-Hour"
        else:
            freq = 'D'   # Daily
            title_freq = "Daily"
        
        # Aggregate data
        df_agg = df.set_index('timestamp').resample(freq).agg({
            'compound': ['mean', 'count', 'std'],
            'sentiment_label': lambda x: x.mode().iloc[0] if not x.empty else 'Neutral'
        }).round(3)
        
        df_agg.columns = ['sentiment_mean', 'data_points', 'sentiment_std', 'dominant_sentiment']
        df_agg = df_agg.reset_index().dropna()
        
        # Create the professional plot
        fig = go.Figure()
        
        # Main sentiment line with confidence band
        fig.add_trace(go.Scatter(
            x=df_agg['timestamp'],
            y=df_agg['sentiment_mean'],
            mode='lines+markers',
            name='Sentiment Trend',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6, color='#1f77b4'),
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<br>Data Points: %{customdata}<extra></extra>',
            customdata=df_agg['data_points']
        ))
        
        # Add confidence band if we have standard deviation
        if 'sentiment_std' in df_agg.columns and not df_agg['sentiment_std'].isna().all():
            upper_bound = df_agg['sentiment_mean'] + df_agg['sentiment_std']
            lower_bound = df_agg['sentiment_mean'] - df_agg['sentiment_std']
            
            fig.add_trace(go.Scatter(
                x=df_agg['timestamp'],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df_agg['timestamp'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Band',
                fillcolor='rgba(31,119,180,0.2)'
            ))
        
        # Add sentiment threshold lines
        fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                     annotation_text="Positive Threshold (0.05)", annotation_position="top right")
        fig.add_hline(y=-0.05, line_dash="dash", line_color="red", 
                     annotation_text="Negative Threshold (-0.05)", annotation_position="bottom right")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
        
        # Professional styling
        fig.update_layout(
            title=f'{title_freq} Sentiment Analysis Trend',
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            hovermode='x unified',
            showlegend=True,
            height=500,
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=16,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray', range=[-1, 1])
        )
        
        return fig
    
    def create_professional_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create professional sentiment distribution visualization"""
        if df.empty or 'sentiment_label' not in df.columns:
            return self._create_no_data_figure("No data available for distribution analysis")
        
        sentiment_counts = df['sentiment_label'].value_counts()
        total_count = sentiment_counts.sum()
        
        # Calculate percentages
        percentages = (sentiment_counts / total_count * 100).round(1)
        
        # Create subplot with pie chart and bar chart
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Distribution by Percentage", "Distribution by Count"),
            horizontal_spacing=0.15
        )
        
        # Pie chart
        colors = [PROFESSIONAL_COLORS.get(label, '#888888') for label in sentiment_counts.index]
        
        fig.add_trace(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='auto',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ), row=1, col=1)
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=colors,
            text=[f'{count}<br>({pct}%)' for count, pct in zip(sentiment_counts.values, percentages.values)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata}%<extra></extra>',
            customdata=percentages.values
        ), row=1, col=2)
        
        fig.update_layout(
            title_text="Professional Sentiment Distribution Analysis",
            showlegend=False,
            height=400,
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        fig.update_xaxes(title_text="Sentiment Category", row=1, col=2)
        fig.update_yaxes(title_text="Number of Items", row=1, col=2)
        
        return fig
    
    def create_source_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create professional source comparison analysis"""
        if df.empty or 'source' not in df.columns:
            return self._create_no_data_figure("No source data available")
        
        # Calculate metrics by source
        source_stats = df.groupby('source').agg({
            'compound': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).round(3)
        
        source_stats.columns = ['avg_sentiment', 'sentiment_std', 'total_count', 'sentiment_breakdown']
        source_stats = source_stats.reset_index()
        
        # Create professional source comparison
        fig = go.Figure()
        
        # Add bars for average sentiment
        fig.add_trace(go.Bar(
            x=source_stats['source'],
            y=source_stats['avg_sentiment'],
            error_y=dict(type='data', array=source_stats['sentiment_std']),
            marker_color='lightblue',
            text=[f'Avg: {avg:.3f}<br>Count: {count}' 
                 for avg, count in zip(source_stats['avg_sentiment'], source_stats['total_count'])],
            textposition='auto',
            name='Average Sentiment'
        ))
        
        # Add reference lines
        fig.add_hline(y=0.05, line_dash="dash", line_color="green", opacity=0.7)
        fig.add_hline(y=-0.05, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Professional Source Analysis",
            xaxis_title="Data Source",
            yaxis_title="Average Sentiment Score",
            height=400,
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=12),
            showlegend=False
        )
        
        return fig
    
    def create_professional_phrase_table(self, phrases_data: List[Dict]) -> go.Figure:
        """Create a professional table for phrase analysis"""
        if not phrases_data:
            return self._create_no_data_figure("No phrase data available")
        
        # Prepare data for table
        phrases = [item['phrase'] for item in phrases_data]
        frequencies = [item['frequency'] for item in phrases_data]
        sentiments = [f"{item['avg_sentiment']:.3f}" for item in phrases_data]
        relevance = [f"{item['relevance_score']:.2f}" for item in phrases_data]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Key Phrase</b>', '<b>Frequency</b>', '<b>Avg Sentiment</b>', '<b>Relevance Score</b>'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[phrases, frequencies, sentiments, relevance],
                fill_color=[['white', 'lightgray'] * (len(phrases) // 2 + 1)],
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Professional Key Phrase Analysis",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _create_no_data_figure(self, message: str) -> go.Figure:
        """Create a professional no-data figure"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"<b>{message}</b><br><br>Please ensure you have sufficient data<br>and try adjusting your search parameters.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white',
            height=400
        )
        return fig

def create_professional_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """Create professional summary metrics"""
    if df.empty:
        return {
            'total_items': 0,
            'avg_sentiment': 0.0,
            'sentiment_strength': 0.0,
            'data_quality_score': 0.0,
            'coverage_days': 0,
            'source_diversity': 0
        }
    
    metrics = {}
    
    # Basic metrics
    metrics['total_items'] = len(df)
    metrics['avg_sentiment'] = df['compound'].mean() if 'compound' in df.columns else 0.0
    metrics['sentiment_strength'] = abs(metrics['avg_sentiment'])
    
    # Data quality metrics
    if 'text_length' in df.columns:
        avg_length = df['text_length'].mean()
        metrics['data_quality_score'] = min(100, (avg_length / 100) * 100)  # Normalize to 0-100
    else:
        metrics['data_quality_score'] = 75  # Default
    
    # Coverage metrics
    if 'timestamp' in df.columns:
        time_span = (df['timestamp'].max() - df['timestamp'].min()).days
        metrics['coverage_days'] = max(1, time_span)
    else:
        metrics['coverage_days'] = 1
    
    # Source diversity
    if 'source' in df.columns:
        metrics['source_diversity'] = len(df['source'].unique())
    else:
        metrics['source_diversity'] = 1
    
    return metrics

if __name__ == "__main__":
    # Test the professional visualizer
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='H'),
        'compound': np.random.normal(0.1, 0.3, 20),
        'sentiment_label': np.random.choice(['Positive', 'Negative', 'Neutral'], 20),
        'source': np.random.choice(['News-Reuters', 'Reddit', 'HackerNews'], 20),
        'text_length': np.random.randint(50, 200, 20)
    })
    
    visualizer = ProfessionalVisualizer()
    
    # Test time series
    time_fig = visualizer.create_professional_time_series(test_data)
    print("Professional time series created")
    
    # Test distribution
    dist_fig = visualizer.create_professional_distribution(test_data)
    print("Professional distribution created")
    
    # Test metrics
    metrics = create_professional_metrics(test_data)
    print(f"Professional metrics: {metrics}")
