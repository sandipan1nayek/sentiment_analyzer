"""
DEMO SCRIPT FOR SMART INTEGRATED SYSTEM
Test the complete Wikidata + Universal NewsAPI workflow
"""

from smart_integrated_system import analyze_smart_sentiment

def run_demo():
    """Run a complete demo of the smart integrated system"""
    print("🚀 SMART INTEGRATED SENTIMENT SYSTEM DEMO")
    print("=" * 60)
    
    test_query = "Tesla"
    print(f"📝 Demo Query: {test_query}")
    print("-" * 60)
    
    try:
        # Run the complete analysis
        result = analyze_smart_sentiment(test_query, days_back=7)
        
        print("\n📊 DEMO RESULTS SUMMARY:")
        print(f"✅ Domain: {result['domain_detection']['primary_domain']}")
        print(f"✅ Entity Type: {result['domain_detection']['entity_type']}")
        print(f"✅ Confidence: {result['domain_detection']['confidence']}%")
        print(f"✅ Articles Found: {result['content_analysis']['total_articles']}")
        print(f"✅ Overall Sentiment: {result['sentiment_analysis']['overall_sentiment']}")
        print(f"✅ Quality Grade: {result['quality_metrics']['overall_grade']}")
        print(f"✅ API Calls Used: {result['api_usage']['total_api_calls']}")
        
        print("\n🎯 SYSTEM EFFICIENCY:")
        print(f"• Wikidata Calls: {result['api_usage']['wikidata_calls']} (FREE)")
        print(f"• NewsAPI Calls: {result['api_usage']['newsapi_calls']} (PAID)")
        print(f"• Efficiency Rating: {result['api_usage']['efficiency_rating']}")
        
        print("\n📋 SOURCE COVERAGE:")
        source_coverage = result['source_coverage']
        print(f"• Total Sources: {source_coverage['total_sources']}")
        print(f"• Source Diversity: {source_coverage['source_diversity']}")
        print(f"• Universal Coverage: {source_coverage['universal_coverage']}")
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
