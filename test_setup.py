"""
Test script to verify the sentiment analysis pipeline setup
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    required_modules = [
        'pandas',
        'numpy',
        'requests',
        'streamlit',
        'vaderSentiment',
        'spacy',
        'matplotlib',
        'seaborn',
        'plotly',
        'wordcloud'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return failed_imports

def test_spacy_model():
    """Test if SpaCy English model is available"""
    print("\n🧪 Testing SpaCy model...")
    
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp("Apple Inc. is a technology company.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✅ SpaCy model working. Found entities: {entities}")
        return True
    except OSError:
        print("❌ SpaCy model 'en_core_web_sm' not found")
        print("   Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ SpaCy error: {e}")
        return False

def test_sentiment_analysis():
    """Test VADER sentiment analysis"""
    print("\n🧪 Testing sentiment analysis...")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible. I hate it.",
            "It's okay, nothing special."
        ]
        
        for text in test_texts:
            scores = analyzer.polarity_scores(text)
            print(f"✅ '{text[:30]}...' -> Compound: {scores['compound']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        return False

def test_data_fetching():
    """Test data fetching functions (without API calls)"""
    print("\n🧪 Testing data fetching modules...")
    
    try:
        # Test individual module imports
        modules_to_test = [
            'config',
            'data_fetching', 
            'data_cleaning',
            'sentiment_analysis',
            'ner',
            'visualization'
        ]
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                print(f"✅ {module_name}")
            except Exception as e:
                print(f"❌ {module_name}: {e}")
                return False
        
        print("✅ All custom modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Custom module import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Sentiment Analysis Pipeline Setup")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test SpaCy model
    spacy_ok = test_spacy_model()
    
    # Test sentiment analysis
    sentiment_ok = test_sentiment_analysis()
    
    # Test custom modules
    modules_ok = test_data_fetching()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    if not failed_imports:
        print("✅ All required packages installed")
    else:
        print(f"❌ Missing packages: {', '.join(failed_imports)}")
    
    if spacy_ok:
        print("✅ SpaCy model available")
    else:
        print("⚠️ SpaCy model missing (entity recognition will be disabled)")
    
    if sentiment_ok:
        print("✅ Sentiment analysis working")
    else:
        print("❌ Sentiment analysis not working")
    
    if modules_ok:
        print("✅ Custom modules imported successfully")
    else:
        print("❌ Custom module import failed")
    
    # Overall status
    critical_failures = failed_imports or not sentiment_ok or not modules_ok
    
    if not critical_failures:
        print("\n🎉 Setup appears to be working! You can run: streamlit run main.py")
    else:
        print("\n⚠️ Some issues found. Please fix them before running the application.")
        
        if failed_imports:
            print(f"   Run: pip install {' '.join(failed_imports)}")
        if not spacy_ok:
            print("   Run: python -m spacy download en_core_web_sm")
    
    return not critical_failures

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
