"""
Sentiment Analysis Pipeline - Main Streamlit Application

This application fetches news headlines and tweets for a given topic,
performs sentiment analysis, and provides interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    import config
    from data_fetching import fetch_all_data
    from data_cleaning import clean_data, preprocess_for_sentiment
    from sentiment_analysis import SentimentAnalyzer, get_sentiment_summary, filter_by_sentiment
    from ner import EntityExtractor, create_entity_summary, get_entities_for_wordcloud
    from visualization import SentimentVisualizer, create_summary_metrics, filter_data_by_timerange
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(**config.PAGE_CONFIG)

class SentimentAnalysisPipeline:
    """Main pipeline class for sentiment analysis"""
    
    def __init__(self):
        """Initialize the pipeline"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.visualizer = SentimentVisualizer()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame()
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = pd.DataFrame()
        if 'last_topic' not in st.session_state:
            st.session_state.last_topic = ""
        if 'last_days' not in st.session_state:
            st.session_state.last_days = 7
    
    def run_pipeline(self, topic: str, days_back: int) -> pd.DataFrame:
        """
        Run the complete sentiment analysis pipeline
        
        Args:
            topic (str): Topic to analyze
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Processed data with sentiment analysis
        """
        try:
            # Step 1: Fetch data
            with st.status("Fetching data...", expanded=True) as status:
                st.write("🔍 Searching for news articles...")
                st.write("🐦 Searching for tweets...")
                
                raw_data = fetch_all_data(topic, days_back)
                
                if raw_data.empty:
                    st.error("No data found for the given topic and time range.")
                    return pd.DataFrame()
                
                st.write(f"✅ Found {len(raw_data)} total items")
                status.update(label="Data fetched successfully!", state="complete")
            
            # Step 2: Clean data
            with st.status("Cleaning data...", expanded=False) as status:
                cleaned_data = clean_data(raw_data)
                st.write(f"✅ Cleaned data: {len(cleaned_data)} items remaining")
                status.update(label="Data cleaned successfully!", state="complete")
            
            # Step 3: Perform sentiment analysis
            with st.status("Analyzing sentiment...", expanded=False) as status:
                sentiment_data = self.sentiment_analyzer.analyze_dataframe(cleaned_data)
                st.write("✅ Sentiment analysis completed")
                status.update(label="Sentiment analysis completed!", state="complete")
            
            # Step 4: Extract entities
            with st.status("Extracting entities...", expanded=False) as status:
                if self.entity_extractor.nlp:
                    final_data = self.entity_extractor.extract_entities_from_dataframe(sentiment_data)
                    st.write("✅ Named entities extracted")
                else:
                    final_data = sentiment_data
                    st.warning("SpaCy model not available. Skipping entity extraction.")
                status.update(label="Entity extraction completed!", state="complete")
            
            return final_data
            
        except Exception as e:
            st.error(f"Error in pipeline: {str(e)}")
            logger.error(f"Pipeline error: {e}")
            return pd.DataFrame()
    
    def display_summary_metrics(self, df: pd.DataFrame):
        """Display summary metrics in the sidebar"""
        if df.empty:
            return
        
        metrics = create_summary_metrics(df)
        
        st.sidebar.markdown("### 📊 Summary Metrics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Items", metrics['total_items'])
            st.metric("Avg Sentiment", f"{metrics['avg_sentiment']:.3f}")
        
        with col2:
            st.metric("Positive %", f"{metrics['positive_percentage']:.1f}%")
            st.metric("Negative %", f"{metrics['negative_percentage']:.1f}%")
    
    def display_time_series_analysis(self, df: pd.DataFrame):
        """Display time series analysis"""
        st.header("📈 Time Series Analysis")
        
        if df.empty:
            st.info("No data available for time series analysis.")
            return
        
        # Create time series chart
        time_fig = self.visualizer.create_time_series_chart(df)
        st.plotly_chart(time_fig, use_container_width=True)
        
        # Additional time-based insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Daily Sentiment Trends")
            if 'timestamp' in df.columns and 'compound' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily_sentiment = df.groupby('date')['compound'].mean().reset_index()
                st.line_chart(daily_sentiment.set_index('date'))
        
        with col2:
            st.subheader("⏰ Hourly Activity")
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                hourly_counts = df['hour'].value_counts().sort_index()
                st.bar_chart(hourly_counts)
    
    def display_sentiment_distribution(self, df: pd.DataFrame):
        """Display sentiment distribution"""
        st.header("🎯 Sentiment Distribution")
        
        if df.empty:
            st.info("No data available for sentiment distribution.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            dist_fig = self.visualizer.create_sentiment_distribution_chart(df)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            # Source comparison
            source_fig = self.visualizer.create_source_comparison_chart(df)
            st.plotly_chart(source_fig, use_container_width=True)
    
    def display_word_clouds(self, df: pd.DataFrame):
        """Display word clouds for different sentiments"""
        st.header("☁️ Word Clouds")
        
        if df.empty:
            st.info("No data available for word clouds.")
            return
        
        col1, col2 = st.columns(2)
        
        # Positive sentiment word cloud
        with col1:
            st.subheader("Positive Sentiment")
            positive_data = filter_by_sentiment(df, 'Positive')
            if not positive_data.empty:
                positive_text = ' '.join(positive_data['text'].astype(str))
                if positive_text.strip():
                    pos_fig = self.visualizer.create_wordcloud(
                        positive_text, 
                        "Positive Sentiment Word Cloud", 
                        'Greens'
                    )
                    st.pyplot(pos_fig)
                else:
                    st.info("No positive sentiment text available.")
            else:
                st.info("No positive sentiment data found.")
        
        # Negative sentiment word cloud
        with col2:
            st.subheader("Negative Sentiment")
            negative_data = filter_by_sentiment(df, 'Negative')
            if not negative_data.empty:
                negative_text = ' '.join(negative_data['text'].astype(str))
                if negative_text.strip():
                    neg_fig = self.visualizer.create_wordcloud(
                        negative_text, 
                        "Negative Sentiment Word Cloud", 
                        'Reds'
                    )
                    st.pyplot(neg_fig)
                else:
                    st.info("No negative sentiment text available.")
            else:
                st.info("No negative sentiment data found.")
    
    def display_entity_analysis(self, df: pd.DataFrame):
        """Display named entity analysis"""
        st.header("🏷️ Named Entity Recognition")
        
        if df.empty or 'entities' not in df.columns:
            st.info("No entity data available.")
            return
        
        entity_summary = create_entity_summary(df)
        
        if entity_summary['total_entities'] == 0:
            st.info("No entities found in the data.")
            return
        
        # Display entity summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Entities", entity_summary['total_entities'])
        with col2:
            st.metric("Unique Entities", entity_summary['unique_entities'])
        with col3:
            most_common_type = max(entity_summary['entity_types'], 
                                 key=entity_summary['entity_types'].get) if entity_summary['entity_types'] else "None"
            st.metric("Most Common Type", most_common_type)
        
        # Top entities chart
        if entity_summary['top_entities']:
            entities_fig = self.visualizer.create_entity_chart(
                entity_summary['top_entities'], 
                "Top Named Entities"
            )
            st.plotly_chart(entities_fig, use_container_width=True)
        
        # Entity types breakdown
        if entity_summary['entity_types']:
            st.subheader("Entity Types Distribution")
            entity_types_df = pd.DataFrame(
                list(entity_summary['entity_types'].items()), 
                columns=['Entity Type', 'Count']
            )
            st.bar_chart(entity_types_df.set_index('Entity Type'))
    
    def display_data_table(self, df: pd.DataFrame):
        """Display the processed data table"""
        st.header("📋 Processed Data")
        
        if df.empty:
            st.info("No data to display.")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Negative", "Neutral"]
            )
        
        with col2:
            source_filter = st.selectbox(
                "Filter by Source",
                ["All"] + list(df['source'].unique()) if 'source' in df.columns else ["All"]
            )
        
        with col3:
            show_entities = st.checkbox("Show Entities", value=False)
        
        # Apply filters
        display_df = df.copy()
        
        if sentiment_filter != "All" and 'sentiment_label' in df.columns:
            display_df = display_df[display_df['sentiment_label'] == sentiment_filter]
        
        if source_filter != "All" and 'source' in df.columns:
            display_df = display_df[display_df['source'] == source_filter]
        
        # Select columns to display
        display_columns = ['timestamp', 'source', 'text', 'compound', 'sentiment_label']
        if show_entities and 'entities' in df.columns:
            display_columns.append('entities')
        
        # Display filtered data
        display_df_filtered = display_df[
            [col for col in display_columns if col in display_df.columns]
        ]
        
        st.dataframe(
            display_df_filtered,
            use_container_width=True,
            height=400
        )
        
        # Download button
        if not display_df_filtered.empty:
            csv = display_df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def run_app(self):
        """Run the main Streamlit application"""
        # Title and description
        st.title("🎭 Sentiment Analysis Pipeline")
        st.markdown("""
        This application analyzes sentiment in news headlines and tweets for any topic you specify.
        It provides comprehensive insights including time-series analysis, entity recognition, and interactive visualizations.
        """)
        
        # Sidebar inputs
        st.sidebar.header("🔧 Configuration")
        
        # Topic input
        topic = st.sidebar.text_input(
            "Enter Topic",
            value="Tesla",
            help="Enter a topic to analyze (e.g., 'Tesla', 'AI', 'iPhone 16')"
        )
        
        # Time range selection
        time_range_label = st.sidebar.selectbox(
            "Select Time Range",
            list(config.TIME_RANGES.keys()),
            index=1  # Default to 7 days
        )
        days_back = config.TIME_RANGES[time_range_label]
        
        # API key warning
        if config.NEWSAPI_KEY == 'your_newsapi_key_here':
            st.sidebar.warning(
                "⚠️ Please set your NewsAPI key in config.py or as an environment variable NEWSAPI_KEY"
            )
        
        # Run analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            if not topic.strip():
                st.error("Please enter a topic to analyze.")
                return
            
            # Check if we need to fetch new data
            need_new_data = (
                st.session_state.last_topic != topic or 
                st.session_state.last_days != days_back or
                st.session_state.processed_data.empty
            )
            
            if need_new_data:
                # Run the pipeline
                processed_data = self.run_pipeline(topic, days_back)
                
                if not processed_data.empty:
                    st.session_state.processed_data = processed_data
                    st.session_state.last_topic = topic
                    st.session_state.last_days = days_back
                    st.success("✅ Analysis completed successfully!")
                else:
                    st.error("❌ No data could be processed. Please try a different topic or time range.")
                    return
            else:
                st.info("Using cached results. Click 'Run Analysis' again to refresh data.")
        
        # Display results if data is available
        if not st.session_state.processed_data.empty:
            df = st.session_state.processed_data
            
            # Display summary metrics in sidebar
            self.display_summary_metrics(df)
            
            # Main content tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Time Series", 
                "🎯 Distribution", 
                "☁️ Word Clouds", 
                "🏷️ Entities", 
                "📋 Data"
            ])
            
            with tab1:
                self.display_time_series_analysis(df)
            
            with tab2:
                self.display_sentiment_distribution(df)
            
            with tab3:
                self.display_word_clouds(df)
            
            with tab4:
                self.display_entity_analysis(df)
            
            with tab5:
                self.display_data_table(df)
        
        else:
            st.info("👆 Enter a topic and click 'Run Analysis' to get started!")
            
            # Show example
            st.markdown("""
            ### 📚 How to Use:
            1. **Enter a Topic**: Type any topic you want to analyze (e.g., "Tesla", "AI", "Climate Change")
            2. **Select Time Range**: Choose how far back to search (1, 7, or 30 days)
            3. **Run Analysis**: Click the button to fetch and analyze data
            4. **Explore Results**: Navigate through the tabs to see different visualizations
            
            ### 🔧 Setup Required:
            - Get a free API key from [NewsAPI.org](https://newsapi.org/)
            - Add it to your environment variables or update `config.py`
            - Install required packages: `pip install -r requirements.txt`
            - Download SpaCy model: `python -m spacy download en_core_web_sm`
            """)

if __name__ == "__main__":
    # Initialize and run the application
    pipeline = SentimentAnalysisPipeline()
    pipeline.run_app()