# 🎭 Sentiment Analysis Pipeline

A comprehensive Python application that performs sentiment analysis on news headlines and sample social media data for any given topic. Features real-time data fetching, sentiment classification, named entity recognition, and interactive visualizations.

## 🌟 Features

- **Multi-Source Data Fetching**: Aggregates data from NewsAPI and generates sample social media data
- **Real-Time Sentiment Analysis**: Uses VADER sentiment analysis with customizable thresholds
- **Named Entity Recognition**: Extracts people, organizations, and locations using SpaCy
- **Interactive Visualizations**: Time-series analysis, sentiment distribution, word clouds
- **Streamlit Dashboard**: User-friendly web interface
- **Data Export**: Download results as CSV
- **Configurable Time Ranges**: 1 day, 7 days, or 30 days analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection for data fetching
- NewsAPI key (free from [newsapi.org](https://newsapi.org/))

### Quick Setup with Virtual Environment

**For Windows:**
```bash
# Clone or download this repository
git clone <repository-url>
cd sentiment_analyzer

# Run the setup script
setup_venv.bat
```

**For macOS/Linux:**
```bash
# Clone or download this repository
git clone <repository-url>
cd sentiment_analyzer

# Make the script executable and run
chmod +x setup_venv.sh
./setup_venv.sh
```

### Manual Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Test the setup
python test_setup.py
```

## 🔑 API Configuration

1. **Get NewsAPI Key**
   - Visit [newsapi.org](https://newsapi.org/)
   - Sign up for a free account
   - Copy your API key

2. **Configure the Key**
   
   **Option A: Environment Variable (Recommended)**
   ```bash
   # Windows
   set NEWSAPI_KEY=your_api_key_here
   
   # macOS/Linux
   export NEWSAPI_KEY=your_api_key_here
   ```
   
   **Option B: Update config.py**
   ```python
   NEWSAPI_KEY = 'your_api_key_here'
   ```

## 🚀 Running the Application

### Using the Batch Script (Windows)
```bash
run_app.bat
```

### Manual Start
```bash
# Activate virtual environment first
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Start the application
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Dashboard

1. **Enter a Topic**: Type any topic you want to analyze (e.g., "Tesla", "AI", "Climate Change")
2. **Select Time Range**: Choose how far back to search (1, 7, or 30 days)
3. **Run Analysis**: Click the "Run Analysis" button
4. **Explore Results**: Navigate through the tabs to see different visualizations

### Available Visualizations

- **📈 Time Series**: Sentiment trends over time
- **🎯 Distribution**: Pie charts and source comparisons
- **☁️ Word Clouds**: Visual representation of frequent words
- **🏷️ Entities**: Named entity recognition results
- **📋 Data**: Raw data table with filtering and export options

## 📊 Data Sources

- **NewsAPI**: Real news headlines from various sources
- **Sample Social Media Data**: Generated examples for demonstration (replaces Twitter due to API limitations)

Note: For production use with real Twitter data, you would need to implement Twitter API v2 authentication.

## 📊 Output Format

The pipeline generates a DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Date and time of the post/article |
| `source` | Data source (NewsAPI or Twitter) |
| `text` | Cleaned text content |
| `positive` | VADER positive sentiment score |
| `negative` | VADER negative sentiment score |
| `neutral` | VADER neutral sentiment score |
| `compound` | VADER compound sentiment score |
| `sentiment_label` | Classified sentiment (Positive/Negative/Neutral) |
| `entities` | Extracted named entities (if SpaCy available) |

## 🔧 Configuration

Key settings in `config.py`:

```python
# Sentiment thresholds
POSITIVE_THRESHOLD = 0.05    # Compound score >= 0.05 = Positive
NEGATIVE_THRESHOLD = -0.05   # Compound score <= -0.05 = Negative

# Data limits
MAX_RESULTS_PER_SOURCE = 100
TWITTER_SEARCH_LIMIT = 100

# Time ranges
TIME_RANGES = {
    "1 Day": 1,
    "7 Days": 7,
    "30 Days": 30
}
```

## 📁 Project Structure

```
sentiment_analyzer/
├── main.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── data_fetching.py        # NewsAPI and sample data generation
├── data_cleaning.py        # Text preprocessing and cleaning
├── sentiment_analysis.py   # VADER sentiment analysis
├── ner.py                 # Named entity recognition
├── visualization.py       # Plotly and matplotlib visualizations
├── setup.py               # Setup and installation script
├── test_setup.py          # Setup verification script
├── setup_venv.bat         # Windows virtual environment setup
├── setup_venv.sh          # Linux/macOS virtual environment setup
├── run_app.bat            # Windows application launcher
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🔍 How It Works

1. **Data Fetching**: Retrieves recent news articles and generates sample social media data
2. **Data Cleaning**: Removes URLs, emojis, special characters, and duplicates
3. **Sentiment Analysis**: Applies VADER sentiment analyzer to classify text
4. **Entity Extraction**: Uses SpaCy to identify people, organizations, and locations
5. **Visualization**: Creates interactive charts and word clouds
6. **Dashboard**: Presents results in an intuitive Streamlit interface

## 🧪 Testing Individual Components

Each module can be tested independently:

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Test data fetching
python data_fetching.py

# Test sentiment analysis
python sentiment_analysis.py

# Test entity recognition
python ner.py

# Test visualizations
python visualization.py

# Test entire setup
python test_setup.py
```

## 📋 Dependencies

Core packages:
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `requests` - HTTP requests for NewsAPI
- `vaderSentiment` - Sentiment analysis
- `spacy` - Natural language processing
- `plotly` - Interactive visualizations
- `wordcloud` - Word cloud generation
- `matplotlib` - Static visualizations

## 🚨 Troubleshooting

### Common Issues

1. **"SpaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"No data found"**
   - Check your internet connection
   - Verify NewsAPI key is valid
   - Try a more popular topic or longer time range

3. **"Import errors"**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is being used

4. **"API rate limit exceeded"**
   - NewsAPI free tier has daily limits
   - Wait or upgrade your NewsAPI plan

5. **"Virtual environment issues"**
   ```bash
   # Delete and recreate venv
   rmdir /s venv  # Windows
   # or
   rm -rf venv    # macOS/Linux
   
   # Then run setup again
   setup_venv.bat  # Windows
   # or
   ./setup_venv.sh # macOS/Linux
   ```

### Performance Tips

- Start with shorter time ranges (1-7 days) for faster results
- Popular topics yield more data than niche subjects
- Entity extraction can be disabled if SpaCy model is unavailable

## 🌐 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your NewsAPI key to secrets management
5. Deploy!

### Local Network Access

```bash
streamlit run main.py --server.address 0.0.0.0
```

## 📈 Future Enhancements

- [ ] Real Twitter API integration with proper authentication
- [ ] Support for additional data sources (Reddit, LinkedIn)
- [ ] Machine learning-based sentiment models
- [ ] Real-time streaming data
- [ ] Advanced filtering and search
- [ ] Sentiment tracking over longer periods
- [ ] Export to different formats (PDF, Excel)
- [ ] Email notifications for sentiment alerts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [SpaCy](https://spacy.io/) for NLP capabilities
- [Streamlit](https://streamlit.io/) for the web framework
- [NewsAPI](https://newsapi.org/) for news data

---

**Happy Analyzing! 🎉**

## 🎯 Getting Started Checklist

- [ ] Clone the repository
- [ ] Run `setup_venv.bat` (Windows) or `./setup_venv.sh` (macOS/Linux)
- [ ] Get NewsAPI key from [newsapi.org](https://newsapi.org/)
- [ ] Set environment variable: `set NEWSAPI_KEY=your_key_here`
- [ ] Run `run_app.bat` or `streamlit run main.py`
- [ ] Open browser to `http://localhost:8501`
- [ ] Enter a topic and start analyzing!