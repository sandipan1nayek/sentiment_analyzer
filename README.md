# Professional Sentiment Analyzer

A production-grade sentiment analysis platform for real-time analysis of news content with comprehensive data visualization and insights.

## 📊 Project Overview

This sentiment analysis application provides real-time analysis of news articles with advanced natural language processing capabilities. The platform combines multiple data sources with sophisticated sentiment analysis models to deliver actionable insights through an intuitive web interface.

### 🎯 Key Features

- **Real-time News Analysis**: Fetches and analyzes current news articles
- **Advanced Sentiment Detection**: Uses VADER sentiment analysis for accurate emotional tone detection
- **Named Entity Recognition**: Identifies and extracts key entities from news content
- **Phrase Analysis**: Contextual phrase extraction with sentiment correlation
- **Interactive Visualizations**: Professional charts and metrics with Plotly
- **Multi-metric Dashboard**: Comprehensive data quality and sentiment metrics

## 📸 Screenshots

### Main Dashboard
![Main Dashboard](screenshots/main_dashboard.png)
*The clean, intuitive interface with sidebar controls and main analysis area*

### Sentiment Distribution Results
![Distribution Results](screenshots/distribution_results.png)
*Interactive sentiment distribution charts with comprehensive metrics*

### Key Phrases Analysis
![Key Phrases](screenshots/key_phrases_analysis.png)
*Advanced phrase extraction with sentiment correlation table (now with improved black text)*

### Sidebar Configuration
![Sidebar Controls](screenshots/sidebar_configuration.png)
*Comprehensive analysis parameters and settings*

## 🗞️ Data Source Selection

### Chosen Data Source: NewsAPI

**Primary Data Source**: [NewsAPI](https://newsapi.org/)

**Rationale for Selection**:

1. **Comprehensive Coverage**: NewsAPI aggregates from 80,000+ news sources worldwide
2. **Real-time Updates**: Provides fresh content updated every few minutes
3. **Reliable API**: Well-documented, stable API with consistent data structure
4. **Rich Metadata**: Includes publication date, source, author, and full content
5. **Global Reach**: Covers international news in multiple languages
6. **Free Tier Available**: Allows up to 1,000 requests per day for development

**Data Quality Benefits**:
- High-quality, editorial content from reputable sources
- Consistent JSON format for reliable parsing
- Built-in source credibility through vetted publishers
- Real-time availability ensures current event analysis

## 🤖 Sentiment Analysis Model

### Chosen Model: VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Model Selection**: VADER Sentiment Intensity Analyzer

**Why VADER was Selected**:

1. **Social Media Optimized**: Specifically designed for social media text and news content
2. **No Training Required**: Pre-built lexicon approach, no need for training data
3. **Punctuation Sensitive**: Understands emphasis through punctuation (!!!, ???)
4. **Capitalization Aware**: Recognizes sentiment intensity through capitalization
5. **Emoticon Recognition**: Interprets emoticons and emojis appropriately
6. **Speed**: Extremely fast analysis suitable for real-time applications
7. **Intensity Scoring**: Provides compound scores from -1 (most negative) to +1 (most positive)

**Technical Advantages**:
- Handles negations effectively ("not good" vs "good")
- Understands degree modifiers ("very good" vs "good")
- Works well with informal language and news headlines
- Provides granular sentiment scores (positive, negative, neutral, compound)

**Alternative Models Considered**:
- **TextBlob**: Less accurate for news content and social media
- **BERT-based models**: Too resource-intensive for real-time analysis
- **Custom models**: Would require extensive training data and maintenance

### 🔒 **Important Security Note**

**Why is `config.py` not in the repository?**

The `config.py` file contains sensitive API keys and is intentionally excluded from version control for security reasons. This prevents accidental exposure of API keys on public repositories.

**For new users**:
1. Use the provided `config.py.template` as a starting point
2. Copy it to `config.py` and add your actual API keys
3. Never commit `config.py` to version control
4. The `.gitignore` file already protects against accidental commits

## 🚀 Local Setup and Execution

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or higher)
- **Internet connection** for real-time data fetching
- **NewsAPI key** (free registration at [newsapi.org](https://newsapi.org/))

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sandipan1nayek/sentiment_analyzer.git
   cd sentiment_analyzer
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy Language Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Configure API Keys**
   
   **Create Environment File**:
   
   a) **Copy the environment template**:
   ```bash
   copy .env.template .env
   # On macOS/Linux: cp .env.template .env
   ```
   
   b) **Get your NewsAPI key**:
   - Visit [NewsAPI.org](https://newsapi.org/register)
   - Create a free account (completely free)
   - Copy your API key from the dashboard
   
   c) **Edit .env file**:
   ```bash
   # Open .env in any text editor and replace:
   NEWSAPI_KEY=your_actual_api_key_here
   ```
   
   d) **Example .env file**:
   ```bash
   # Your .env should look like this:
   NEWSAPI_KEY=1234567890abcdef1234567890abcdef
   ENVIRONMENT=development
   ```

6. **Run the Application**
   ```bash
   streamlit run main.py
   ```

7. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The sentiment analyzer interface will load

### 🎮 Usage Instructions

1. **Enter Search Query**: Type your search term (e.g., "artificial intelligence", "climate change")
2. **Configure Settings**: Adjust time range and analysis parameters in the sidebar
3. **Run Analysis**: Click "🚀 Run Analysis" to fetch and analyze articles
4. **Explore Results**: Navigate through different tabs:
   - **Distribution**: View sentiment distribution charts
   - **Key Phrases**: Analyze important phrases and topics
   - **Sources**: Examine source diversity and quality
   - **Data**: Access raw data and download options

## 📁 Project Structure

```
sentiment_analyzer/
├── main.py                     # Main Streamlit application
├── config.py                   # Configuration settings (now in repository)
├── .env.template              # Environment variables template
├── .env                       # Your API keys (create from template)
├── final_working_system.py     # Core news fetching and processing
├── sentiment_analysis.py       # VADER sentiment analysis implementation
├── ner.py                     # Named entity recognition with SpaCy
├── phrase_analysis.py         # Advanced phrase extraction and analysis
├── visualization.py           # Professional charts and visualizations
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore patterns (includes .env)
└── screenshots/             # Application screenshots
    ├── main_dashboard.png
    ├── distribution_results.png
    ├── key_phrases_analysis.png
    └── sidebar_configuration.png
```

**Note**: `config.py` is created by the user from `config.py.template` and contains sensitive API keys.

## � Security & Configuration

This project uses environment variables for secure API key management:

- **`.env` file**: Contains your private API keys (never committed to git)
- **`.env.template`**: Template file showing required environment variables
- **`config.py`**: Safe to include in repository, loads from environment variables
- **`.gitignore`**: Ensures `.env` file is never accidentally committed

This approach ensures:
- ✅ API keys remain private and secure
- ✅ Easy setup for new developers
- ✅ No sensitive data in repository history
- ✅ Production deployment compatibility

## �🛠️ Technical Architecture

### Core Components

- **Frontend**: Streamlit web interface with responsive design
- **Backend**: Python-based analysis pipeline
- **Data Processing**: Pandas for data manipulation and analysis
- **Visualization**: Plotly for interactive charts and graphs
- **NLP Pipeline**: SpaCy + VADER for comprehensive text analysis

### Data Flow

1. **Input**: User query through web interface
2. **Fetch**: NewsAPI retrieval with relevance filtering
3. **Process**: Sentiment analysis, entity extraction, phrase analysis
4. **Visualize**: Interactive charts and metrics dashboard
5. **Export**: CSV download capability for further analysis

## 📊 Performance Metrics

- **Analysis Speed**: ~2-3 seconds for 50+ articles
- **Accuracy**: VADER provides ~80% accuracy on news content
- **Data Quality**: Built-in quality scoring and source diversity metrics
- **Reliability**: Guaranteed 50+ articles per query through robust fetching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your NewsAPI key is correctly set in `config.py`
2. **SpaCy Model Missing**: Run `python -m spacy download en_core_web_sm`
3. **Port Already in Use**: Change the port with `streamlit run main.py --server.port 8502`
4. **Slow Loading**: Check your internet connection and API rate limits

### Support

For issues and questions:
- Check the [Issues](https://github.com/sandipan1nayek/sentiment_analyzer/issues) page
- Create a new issue with detailed description
- Include error messages and system information

## 🏆 Acknowledgments

- **NewsAPI** for providing comprehensive news data
- **VADER Sentiment** for robust sentiment analysis
- **SpaCy** for advanced natural language processing
- **Streamlit** for the intuitive web framework
- **Plotly** for interactive visualizations
