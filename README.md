# Professional Sentiment Analyzer

A production-grade sentiment analysis platform for real-time analysis of news and social media content.

## Features

- **Real-time Data Sources**: NewsAPI, Reddit, HackerNews integration
- **Advanced Analytics**: VADER sentiment analysis with professional metrics
- **Entity Recognition**: Named entity extraction using SpaCy
- **Phrase Analysis**: Contextual phrase extraction and sentiment correlation
- **Interactive Visualizations**: Professional charts with Plotly
- **Multi-source Integration**: Comprehensive data quality metrics

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/sandipan1nayek/sentiment_analyzer.git
   cd sentiment_analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure API keys**
   - Copy `config.py` and add your NewsAPI key
   - Set `NEWS_API_KEY = "your_api_key_here"`

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Access the app**
   - Open your browser to `http://localhost:8501`

## Usage

1. Enter a search query (e.g., "Bitcoin", "Tesla", "Climate change")
2. Select data sources (News, Reddit, HackerNews)
3. Click "Run Analysis" to fetch and analyze data
4. Explore interactive charts and insights

## Requirements

- Python 3.8+
- NewsAPI key (free at newsapi.org)
- Internet connection for real-time data

## Project Structure

```
sentiment_analyzer/
├── main.py                          # Main Streamlit application
├── config.py                        # Configuration and API keys
├── smart_integrated_system.py       # Complete end-to-end system
├── wikidata_domain_detector.py      # Authoritative domain detection
├── universal_news_fetcher.py        # Single-call universal news API
├── sentiment_analysis.py            # VADER sentiment analysis
├── ner.py                          # Named entity recognition
├── phrase_analysis.py              # Advanced phrase extraction
├── visualization.py                # Professional charts
└── requirements.txt                # Dependencies
```

## License

MIT License - see LICENSE file for details.
