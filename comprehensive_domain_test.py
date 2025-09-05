"""
🧠 COMPREHENSIVE DOMAIN DETECTION TEST
Tests the enhanced intelligent domain detector with various query types
"""

from intelligent_domain_detector import IntelligentDomainDetector

def main():
    detector = IntelligentDomainDetector()
    
    # Test queries with expected domains
    test_cases = [
        # Finance domain
        ("India Economic Growth", "finance"),
        ("US stock market crash", "finance"),
        ("Bitcoin price surge", "finance"),
        ("Tesla stock performance", "finance"),
        ("Apple revenue report", "finance"),
        ("Federal Reserve interest rates", "finance"),
        ("Economic inflation concerns", "finance"),
        
        # Technology domain
        ("Apple iPhone Latest", "technology"),
        ("Microsoft AI Technology", "technology"),
        ("ChatGPT AI development", "technology"),
        ("Docker containerization", "technology"),
        ("Python programming tutorial", "technology"),
        ("React Native app development", "technology"),
        ("Google Pixel camera features", "technology"),
        
        # Entertainment domain
        ("Netflix streaming", "entertainment"),
        ("Bollywood movie review", "entertainment"),
        ("Taylor Swift concert", "entertainment"),
        ("Stranger Things season", "entertainment"),
        
        # Healthcare domain
        ("COVID vaccine effectiveness", "healthcare"),
        ("Heart surgery procedure", "healthcare"),
        ("Diabetes treatment options", "healthcare"),
        
        # Sports domain
        ("Cricket IPL match", "sports"),
        ("Virat Kohli performance", "sports"),
        ("Tennis Wimbledon championship", "sports"),
        
        # Person domain
        ("Narendra Modi policies", "person"),
        ("Elon Musk announcement", "person"),
        ("Bill Gates interview", "person"),
        
        # Location domain
        ("Mumbai traffic issues", "location"),
        ("Delhi pollution levels", "location"),
        ("Singapore development", "location"),
        
        # Organization domain
        ("Google company culture", "organization"),
        ("Amazon workplace policies", "organization"),
        ("Microsoft layoffs", "organization"),
        
        # Automotive domain
        ("Tesla Model 3 review", "automotive"),
        ("BMW electric vehicle", "automotive"),
        ("Ford truck specifications", "automotive")
    ]
    
    print("🧠 COMPREHENSIVE DOMAIN DETECTION TEST")
    print("=" * 50)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        result = detector.detect_domain_intelligently(query)
        detected = result['primary_domain']
        confidence = result['confidence'] * 100
        
        status = "✅ CORRECT" if detected == expected else "❌ WRONG"
        if detected == expected:
            correct += 1
        
        print(f"Query: '{query}'")
        print(f"  → Detected: {detected} ({confidence:.1f}%)")
        print(f"  → Expected: {expected}")
        print(f"  → Status: {status}")
        print()
    
    accuracy = (correct / total) * 100
    print("=" * 50)
    print(f"📊 OVERALL RESULTS:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 50)
    
    if accuracy >= 90:
        print("🎉 EXCELLENT! Domain detection is production-ready!")
    elif accuracy >= 80:
        print("✅ GOOD! Minor improvements may be needed.")
    else:
        print("⚠️ NEEDS IMPROVEMENT! Accuracy below target.")

if __name__ == "__main__":
    main()
