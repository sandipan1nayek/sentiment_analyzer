"""
Professional Sentiment Analysis Application
A production-grade sentiment analysis platform for news and social media content

Features:
- Real-time data fetching from Reddit, HackerNews, and NewsAPI
- Advanced phrase analysis with contextual insights
- Professional visualizations with stable, clear charts
- Comprehensive entity analysis and sentiment correlation
- Multi-source data integration with quality metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import professional modules
from data_fetching import ProfessionalDataFetcher
from sentiment_analysis import SentimentAnalyzer
from ner import EntityExtractor
from visualization import ProfessionalVisualizer, create_professional_metrics
from phrase_analysis import ProfessionalPhraseAnalyzer
from data_cleaning import DataCleaner
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Professional Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .professional-sidebar {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalSentimentApp:
    """Main application class for professional sentiment analysis"""
    
    def __init__(self):
        """Initialize the professional application"""
        self.initialize_components()
        self.initialize_session_state()
    
    def initialize_components(self):
        """Initialize all professional components"""
        try:
            self.data_fetcher = ProfessionalDataFetcher()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.entity_extractor = EntityExtractor()
            self.visualizer = ProfessionalVisualizer()
            self.phrase_analyzer = ProfessionalPhraseAnalyzer()
            self.data_cleaner = DataCleaner()
            logger.info("All professional components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Initialization error: {e}")
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = pd.DataFrame()
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = "Ready"
        if 'data_quality_score' not in st.session_state:
            st.session_state.data_quality_score = 0.0
        if 'phrase_insights' not in st.session_state:
            st.session_state.phrase_insights = []
    
    def render_header(self):
        """Render the professional header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Professional Sentiment Analysis Platform</h1>
            <p>Real-time sentiment analysis across News, Reddit, and HackerNews with advanced phrase insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_professional_sidebar(self):
        """Render the professional sidebar with controls"""
        with st.sidebar:
            st.markdown('<div class="professional-sidebar">', unsafe_allow_html=True)
            
            st.header("🔧 Analysis Configuration")
            
            # Data source selection
            st.subheader("Data Sources")
            sources = st.multiselect(
                "Select data sources:",
                ["NewsAPI", "Reddit", "HackerNews"],
                default=["NewsAPI", "Reddit", "HackerNews"],
                help="Choose which data sources to analyze"
            )
            
            # Time range selection
            st.subheader("Time Range")
            time_range = st.selectbox(
                "Analysis period:",
                [1, 3, 7, 14, 30],
                index=2,  # Default to 7 days
                help="Number of days to analyze"
            )
            
            # Search configuration
            st.subheader("Search Parameters")
            search_query = st.text_input(
                "Search keywords:",
                value="technology AI artificial intelligence",
                help="Keywords to search for in content"
            )
            
            # Show intelligent query expansion info
            if search_query.strip():
                st.info("🧠 **Smart Query Expansion Active**\n"
                       "The system automatically expands your query with related terms for better results!")
            
            # Advanced relevance filtering indicator
            st.success("🎯 **Advanced AI Filtering Enabled**\n"
                      "✅ Semantic similarity scoring\n"
                      "✅ Named entity recognition\n" 
                      "✅ Intelligent noise removal")
            
            # Advanced options
            with st.expander("Advanced Options"):
                min_text_length = st.slider("Minimum text length", 10, 500, 50)
                sentiment_threshold = st.slider("Sentiment threshold", 0.01, 0.2, 0.05, 0.01)
                max_items_per_source = st.slider("Max items per source", 20, 200, 50)
            
            # Action buttons
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Run Analysis", type="primary", width='stretch'):
                    self.run_professional_analysis(sources, time_range, search_query, 
                                                 min_text_length, sentiment_threshold, max_items_per_source)
            
            with col2:
                if st.button("🔄 Refresh Data", width='stretch'):
                    st.session_state.analysis_data = pd.DataFrame()
                    st.rerun()
            
            # Status indicator
            st.markdown("---")
            st.subheader("Status")
            status_color = {
                "Ready": "🟢",
                "Processing": "🟡", 
                "Complete": "🟢",
                "Error": "🔴"
            }
            st.write(f"{status_color.get(st.session_state.processing_status, '⚪')} {st.session_state.processing_status}")
            
            if st.session_state.last_update:
                st.write(f"📅 Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return sources, time_range, search_query, min_text_length, sentiment_threshold, max_items_per_source
    
    def run_professional_analysis(self, sources, time_range, search_query, min_text_length, sentiment_threshold, max_items_per_source):
        """Run the complete professional analysis pipeline"""
        try:
            st.session_state.processing_status = "Processing"
            
            with st.spinner("🔄 Fetching data from professional sources..."):
                # Fetch data from selected sources
                all_dataframes = []
                
                if "NewsAPI" in sources:
                    with st.spinner("📰 Fetching news articles..."):
                        news_data = self.data_fetcher.fetch_comprehensive_news(
                            topic=search_query,
                            days_back=time_range
                        )
                        if not news_data.empty:
                            all_dataframes.append(news_data)
                
                if "Reddit" in sources:
                    with st.spinner("🔄 Fetching Reddit discussions..."):
                        reddit_data = self.data_fetcher.fetch_reddit_data(
                            topic=search_query,
                            days_back=time_range
                        )
                        if not reddit_data.empty:
                            all_dataframes.append(reddit_data)
                
                if "HackerNews" in sources:
                    with st.spinner("📈 Fetching HackerNews stories..."):
                        hn_data = self.data_fetcher.fetch_hackernews_data(
                            topic=search_query,
                            days_back=time_range
                        )
                        if not hn_data.empty:
                            all_dataframes.append(hn_data)
            
            if not all_dataframes:
                st.warning("No data found for the specified parameters. Try adjusting your search terms or time range.")
                return
            
            # Create DataFrame and clean data
            with st.spinner("🧹 Cleaning and processing data..."):
                df = pd.concat(all_dataframes, ignore_index=True)
                df = self.data_cleaner.clean_data(df)
                
                # Filter by text length
                df = df[df['text'].str.len() >= min_text_length]
            
            if df.empty:
                st.warning("No data remaining after cleaning. Try reducing the minimum text length.")
                return
            
            # Perform sentiment analysis
            with st.spinner("🎯 Analyzing sentiment..."):
                df = self.sentiment_analyzer.analyze_dataframe(df)
                df['text_length'] = df['text'].str.len()
            
            # Extract entities
            with st.spinner("🏷️ Extracting entities..."):
                df = self.entity_extractor.extract_entities_dataframe(df)
            
            # Perform phrase analysis
            with st.spinner("🔍 Analyzing key phrases..."):
                phrase_insights = self.phrase_analyzer.analyze_phrases(df['text'].tolist())
                # Extract key phrases list from the returned dictionary
                if isinstance(phrase_insights, dict) and 'key_phrases' in phrase_insights:
                    st.session_state.phrase_insights = phrase_insights['key_phrases']
                else:
                    st.session_state.phrase_insights = []
            
            # Store processed data
            st.session_state.analysis_data = df
            st.session_state.last_update = datetime.now()
            st.session_state.processing_status = "Complete"
            st.session_state.data_quality_score = create_professional_metrics(df)['data_quality_score']
            
            st.success(f"✅ Analysis complete! Processed {len(df)} items from {len(sources)} sources.")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            st.session_state.processing_status = "Error"
            st.error(f"Analysis failed: {str(e)}")
    
    def render_professional_metrics(self):
        """Render professional metrics dashboard"""
        if st.session_state.analysis_data.empty:
            st.info("👆 Configure your analysis in the sidebar and click 'Run Analysis' to get started.")
            return
        
        df = st.session_state.analysis_data
        metrics = create_professional_metrics(df)
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📊 Total Items",
                value=f"{metrics['total_items']:,}",
                help="Total number of analyzed items"
            )
        
        with col2:
            sentiment_value = f"{metrics['avg_sentiment']:+.3f}"
            sentiment_delta = "Positive" if metrics['avg_sentiment'] > 0.05 else "Negative" if metrics['avg_sentiment'] < -0.05 else "Neutral"
            st.metric(
                label="💭 Avg Sentiment", 
                value=sentiment_value,
                delta=sentiment_delta,
                help="Average sentiment score (-1 to +1)"
            )
        
        with col3:
            st.metric(
                label="⚡ Sentiment Strength",
                value=f"{metrics['sentiment_strength']:.3f}",
                help="Absolute strength of sentiment"
            )
        
        with col4:
            quality_color = "🟢" if metrics['data_quality_score'] > 80 else "🟡" if metrics['data_quality_score'] > 60 else "🔴"
            st.metric(
                label=f"{quality_color} Data Quality",
                value=f"{metrics['data_quality_score']:.0f}%",
                help="Data quality assessment"
            )
        
        with col5:
            st.metric(
                label="🔄 Source Diversity",
                value=f"{metrics['source_diversity']} sources",
                help="Number of different data sources"
            )
    
    def render_analysis_tabs(self):
        """Render the main analysis tabs"""
        if st.session_state.analysis_data.empty:
            return
        
        df = st.session_state.analysis_data
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Time Series Analysis", 
            "📊 Distribution Analysis", 
            "🔍 Key Phrase Insights",
            "🏷️ Entity Analysis", 
            "📋 Source Comparison",
            "📄 Raw Data"
        ])
        
        with tab1:
            st.subheader("Professional Time Series Analysis")
            time_fig = self.visualizer.create_professional_time_series(df)
            st.plotly_chart(time_fig, width='stretch')
            
            # Additional insights
            if len(df) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📅 **Time Range:** {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    sentiment_trend = "📈 Improving" if df['compound'].iloc[-1] > df['compound'].iloc[0] else "📉 Declining"
                    st.info(f"**Trend:** {sentiment_trend}")
        
        with tab2:
            st.subheader("Professional Distribution Analysis")
            dist_fig = self.visualizer.create_professional_distribution(df)
            st.plotly_chart(dist_fig, width='stretch')
        
        with tab3:
            st.subheader("Advanced Key Phrase Insights")
            if st.session_state.phrase_insights:
                # Display top phrases in a professional table
                phrase_fig = self.visualizer.create_professional_phrase_table(st.session_state.phrase_insights[:20])
                st.plotly_chart(phrase_fig, width='stretch')
                
                # Additional phrase analytics
                st.subheader("Phrase Analytics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔥 Most Discussed Topics:**")
                    top_phrases = st.session_state.phrase_insights[:5]
                    for i, phrase in enumerate(top_phrases, 1):
                        sentiment_emoji = "😊" if phrase['avg_sentiment'] > 0.1 else "😞" if phrase['avg_sentiment'] < -0.1 else "😐"
                        st.write(f"{i}. {sentiment_emoji} **{phrase['phrase']}** ({phrase['frequency']} mentions)")
                
                with col2:
                    st.markdown("**📊 Sentiment Distribution:**")
                    positive_phrases = sum(1 for p in st.session_state.phrase_insights[:10] if p['avg_sentiment'] > 0.1)
                    negative_phrases = sum(1 for p in st.session_state.phrase_insights[:10] if p['avg_sentiment'] < -0.1)
                    neutral_phrases = 10 - positive_phrases - negative_phrases
                    
                    st.write(f"✅ Positive topics: {positive_phrases}")
                    st.write(f"❌ Negative topics: {negative_phrases}")
                    st.write(f"⚪ Neutral topics: {neutral_phrases}")
            else:
                st.info("No phrase insights available. Run the analysis to generate phrase data.")
        
        with tab4:
            st.subheader("Entity Analysis")
            if 'entities' in df.columns and not df['entities'].empty:
                # Extract all entities
                all_entities = []
                for entities_list in df['entities'].dropna():
                    if entities_list:  # Check if not empty
                        all_entities.extend(entities_list)
                
                if all_entities:
                    # Convert list of tuples to DataFrame with proper column names
                    entity_df = pd.DataFrame(all_entities, columns=['text', 'label'])
                    entity_counts = entity_df['text'].value_counts().head(15)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🏷️ Top Entities:**")
                        for entity, count in entity_counts.items():
                            st.write(f"• **{entity}**: {count} mentions")
                    
                    with col2:
                        st.markdown("**📊 Entity Types:**")
                        entity_types = entity_df['label'].value_counts()
                        for entity_type, count in entity_types.items():
                            st.write(f"• **{entity_type}**: {count} entities")
                else:
                    st.info("No entities extracted from the current dataset.")
            else:
                st.info("No entity data available.")
        
        with tab5:
            st.subheader("Source Comparison Analysis")
            if 'source' in df.columns:
                source_fig = self.visualizer.create_source_analysis(df)
                st.plotly_chart(source_fig, width='stretch')
                
                # Source statistics
                source_stats = df.groupby('source').agg({
                    'compound': ['mean', 'count'],
                    'sentiment_label': lambda x: x.mode().iloc[0] if not x.empty else 'Neutral'
                }).round(3)
                
                st.subheader("Source Statistics")
                for source in df['source'].unique():
                    source_data = df[df['source'] == source]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(f"📰 {source}", f"{len(source_data)} items")
                    with col2:
                        avg_sentiment = source_data['compound'].mean()
                        st.metric("Avg Sentiment", f"{avg_sentiment:+.3f}")
                    with col3:
                        dominant = source_data['sentiment_label'].mode().iloc[0] if not source_data.empty else 'N/A'
                        st.metric("Dominant", dominant)
            else:
                st.info("No source information available.")
        
        with tab6:
            st.subheader("Raw Data Analysis")
            
            # Data overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_usage:.1f} MB")
            
            # Sample data
            st.subheader("Sample Data (First 100 rows)")
            display_df = df[['timestamp', 'source', 'text', 'compound', 'sentiment_label']].head(100)
            st.dataframe(display_df, width='stretch')
            
            # Download option
            if st.button("📥 Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def run(self):
        """Run the main application"""
        self.render_header()
        
        # Get configuration from sidebar
        sources, time_range, search_query, min_text_length, sentiment_threshold, max_items_per_source = self.render_professional_sidebar()
        
        # Display metrics
        self.render_professional_metrics()
        
        # Display analysis tabs
        self.render_analysis_tabs()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px;'>
            🎯 Professional Sentiment Analysis Platform | Built with advanced NLP and real-time data integration
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Create and run the professional application
    app = ProfessionalSentimentApp()
    app.run()
