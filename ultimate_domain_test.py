"""
🎯 ULTIMATE DOMAIN DETECTION TEST
Comprehensive test with 200+ entities to verify 98% accuracy threshold
"""

from intelligent_domain_detector import IntelligentDomainDetector

def main():
    detector = IntelligentDomainDetector()
    
    # Comprehensive test cases with expected domains
    test_cases = [
        # People - should detect as 'person'
        ("Narendra Modi", "person"),
        ("Elon Musk", "person"),
        ("Virat Kohli", "person"),
        ("Taylor Swift", "person"),
        ("Barack Obama", "person"),
        ("Greta Thunberg", "person"),
        ("Cristiano Ronaldo", "person"),
        ("Lionel Messi", "person"),
        ("Jeff Bezos", "person"),
        ("Amitabh Bachchan", "person"),
        ("Sundar Pichai", "person"),
        ("Bill Gates", "person"),
        ("Mark Zuckerberg", "person"),
        ("Priyanka Chopra", "person"),
        ("Ratan Tata", "person"),
        ("Sachin Tendulkar", "person"),
        ("Usain Bolt", "person"),
        ("Serena Williams", "person"),
        ("Malala Yousafzai", "person"),
        ("Shah Rukh Khan", "person"),
        ("Joe Biden", "person"),
        ("Vladimir Putin", "person"),
        ("Xi Jinping", "person"),
        ("Angela Merkel", "person"),
        ("Oprah Winfrey", "person"),
        
        # Organizations/Companies - should detect as 'organization'
        ("Tesla", "organization"),
        ("Tata Motors", "organization"),
        ("Google", "organization"),
        ("Microsoft", "organization"),
        ("Amazon", "organization"),
        ("Infosys", "organization"),
        ("Reliance", "organization"),
        ("Flipkart", "organization"),
        ("Mahindra", "organization"),
        ("SpaceX", "organization"),
        ("Apple", "organization"),
        ("IBM", "organization"),
        ("Meta", "organization"),
        ("Netflix", "organization"),
        ("Adobe", "organization"),
        ("Intel", "organization"),
        ("Nvidia", "organization"),
        ("Samsung", "organization"),
        ("BYD", "organization"),
        ("Ford", "organization"),
        ("Toyota", "organization"),
        ("Honda", "organization"),
        ("Ola", "organization"),
        ("Paytm", "organization"),
        ("Zoom", "organization"),
        ("Xiaomi", "organization"),
        
        # Sports - should detect as 'sports'
        ("Cricket", "sports"),
        ("Football", "sports"),
        ("Tennis", "sports"),
        ("Olympics", "sports"),
        ("NBA", "sports"),
        ("Formula 1", "sports"),
        ("Wimbledon", "sports"),
        ("Hockey", "sports"),
        ("Rugby", "sports"),
        ("Chess", "sports"),
        ("IPL", "sports"),
        ("FIFA World Cup", "sports"),
        ("Super Bowl", "sports"),
        ("La Liga", "sports"),
        ("UEFA Champions League", "sports"),
        ("Tour de France", "sports"),
        ("Copa America", "sports"),
        ("Asian Games", "sports"),
        ("Commonwealth Games", "sports"),
        ("Pro Kabaddi League", "sports"),
        
        # Technology - should detect as 'technology'
        ("Artificial Intelligence", "technology"),
        ("Machine Learning", "technology"),
        ("Blockchain", "technology"),
        ("Quantum Computing", "technology"),
        ("5G", "technology"),
        ("Metaverse", "technology"),
        ("Cybersecurity", "technology"),
        ("Cloud Computing", "technology"),
        ("Internet of Things", "technology"),
        ("Big Data", "technology"),
        ("Virtual Reality", "technology"),
        ("Augmented Reality", "technology"),
        ("Generative AI", "technology"),
        ("Robotics", "technology"),
        ("Natural Language Processing", "technology"),
        ("Computer Vision", "technology"),
        ("Data Science", "technology"),
        ("Edge Computing", "technology"),
        ("Autonomous Cars", "technology"),
        ("Neural Networks", "technology"),
        
        # Locations - should detect as 'location'
        ("New Delhi", "location"),
        ("Mumbai", "location"),
        ("New York", "location"),
        ("London", "location"),
        ("Paris", "location"),
        ("Tokyo", "location"),
        ("Bengaluru", "location"),
        ("Beijing", "location"),
        ("Sydney", "location"),
        ("Dubai", "location"),
        ("Los Angeles", "location"),
        ("Berlin", "location"),
        ("Rome", "location"),
        ("Cape Town", "location"),
        ("Cairo", "location"),
        ("Singapore", "location"),
        ("Toronto", "location"),
        ("Hong Kong", "location"),
        ("Moscow", "location"),
        ("Riyadh", "location"),
        ("San Francisco", "location"),
        ("Jakarta", "location"),
        ("Mexico City", "location"),
        ("Seoul", "location"),
        ("Dhaka", "location"),
        ("Kathmandu", "location"),
        
        # Events - context-dependent
        ("World Cup 2022", "sports"),
        ("Oscars 2023", "entertainment"),
        ("G20 Summit", "organization"),
        ("COP28", "organization"),
        ("Indian Elections", "organization"),
        ("Super Bowl 2025", "sports"),
        ("FIFA World Cup", "sports"),
        ("Olympics 2024", "sports"),
        ("CES 2025", "technology"),
        ("UN General Assembly", "organization"),
        ("Davos WEF", "organization"),
        ("Cannes Film Festival", "entertainment"),
        ("Grammy Awards", "entertainment"),
        ("Nobel Prize Ceremony", "organization"),
        ("Met Gala", "entertainment"),
        ("Asian Games 2023", "sports"),
        ("ICC World Cup", "sports"),
        ("Commonwealth Summit", "organization"),
        ("BRICS Summit", "organization"),
        ("Auto Expo India", "automotive"),
        
        # Products - context-dependent
        ("iPhone 15", "technology"),
        ("Samsung Galaxy S24", "technology"),
        ("Tesla Model Y", "automotive"),
        ("PlayStation 5", "technology"),
        ("Xbox Series X", "technology"),
        ("MacBook Pro", "technology"),
        ("Kindle", "technology"),
        ("Ola S1 Pro", "automotive"),
        ("BYD Atto 3", "automotive"),
        ("DJI Mavic 3", "technology"),
        ("Google Pixel 8", "technology"),
        ("Apple Vision Pro", "technology"),
        ("Surface Laptop", "technology"),
        ("OnePlus 12", "technology"),
        ("Realme GT 6", "technology"),
        ("Lenovo ThinkPad", "technology"),
        ("HP Spectre", "technology"),
        ("Sony WH-1000XM5", "technology"),
        ("Bose 700", "technology"),
        ("Canon EOS R5", "technology"),
        
        # Political/Organizations
        ("BJP", "organization"),
        ("Congress", "organization"),
        ("Democrats", "organization"),
        ("Republicans", "organization"),
        ("European Union", "organization"),
        ("NATO", "organization"),
        ("United Nations", "organization"),
        ("Labour Party", "organization"),
        ("RSS", "organization"),
        ("Aam Aadmi Party", "organization"),
        ("Greenpeace", "organization"),
        ("Amnesty International", "organization"),
        ("WHO", "organization"),
        ("IMF", "organization"),
        ("World Bank", "organization"),
        ("OPEC", "organization"),
        ("ASEAN", "organization"),
        ("African Union", "organization"),
        ("UNICEF", "organization"),
        ("Red Cross", "organization"),
        
        # Additional companies/products
        ("Ola Electric", "automotive"),
        ("Rivian", "automotive"),
        ("Nio", "automotive"),
        ("Baidu Apollo", "technology"),
        ("Palantir", "technology"),
        ("Zoomcar", "organization"),
        ("Grab Holdings", "organization"),
        ("Sea Limited", "organization"),
        ("Payoneer", "organization"),
        ("Signal App", "technology"),
        ("ProtonMail", "technology"),
        ("DuckDuckGo", "technology"),
        ("Brave Browser", "technology"),
        ("Mastodon", "technology"),
        ("Bluesky", "technology"),
        ("Threads", "technology"),
        ("Quora Poe", "technology"),
        ("Koo App", "technology"),
        ("Moj App", "technology"),
        
        # Education
        ("Vedantu", "education"),
        ("Byju's", "education"),
        ("Unacademy", "education"),
        ("PhysicsWallah", "education"),
        ("Toppr", "education"),
        ("Eruditus", "education"),
        ("upGrad", "education"),
        ("Coursera", "education"),
        ("EdX", "education"),
        ("Khan Academy", "education"),
        
        # E-commerce/Lifestyle
        ("Nykaa", "organization"),
        ("Mamaearth", "organization"),
        ("Boat Lifestyle", "organization"),
        ("Noise Smartwatch", "technology"),
        ("Fire-Boltt", "technology"),
        ("Pebble Watch", "technology"),
        ("Fastrack Reflex", "technology"),
        ("Titan Smart", "technology"),
        ("Cult.fit", "healthcare"),
        ("HealthifyMe", "healthcare"),
        
        # Healthcare/Pharma
        ("Patanjali Ayurved", "healthcare"),
        ("Dabur India", "healthcare"),
        ("Emami", "healthcare"),
        ("Marico", "healthcare"),
        ("Himalaya Wellness", "healthcare"),
        ("Zydus Wellness", "healthcare"),
        ("Mankind Pharma", "healthcare"),
        ("Glenmark Life", "healthcare"),
        ("Cipla", "healthcare"),
        ("Sun Pharma", "healthcare"),
        
        # Space/Science
        ("Hyperloop One", "technology"),
        ("Virgin Galactic", "organization"),
        ("Blue Origin", "organization"),
        ("Rocket Lab", "organization"),
        ("ISRO Gaganyaan", "technology"),
        ("Chandrayaan 3", "technology"),
        ("Artemis Program", "technology"),
        ("Starship", "technology"),
        ("PSLV-C58", "technology"),
        ("Vega Rocket", "technology"),
        
        # E-commerce
        ("Tata Neu", "organization"),
        ("JioMart", "organization"),
        ("Udaan", "organization"),
        ("Meesho", "organization"),
        ("Snapdeal", "organization"),
        ("ShopClues", "organization"),
        ("Club Factory", "organization"),
        ("Shein", "organization"),
        ("Temu", "organization"),
        ("Pinduoduo", "organization"),
    ]
    
    print("🎯 ULTIMATE DOMAIN DETECTION TEST")
    print("=" * 60)
    print(f"Testing {len(test_cases)} entities for 98% accuracy threshold")
    print("=" * 60)
    
    correct = 0
    total = len(test_cases)
    failed_cases = []
    
    for query, expected in test_cases:
        result = detector.detect_domain_intelligently(query)
        detected = result['primary_domain']
        confidence = result['confidence'] * 100
        
        is_correct = detected == expected
        if is_correct:
            correct += 1
        else:
            failed_cases.append((query, expected, detected, confidence))
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {query}")
        print(f"   → Detected: {detected} ({confidence:.1f}%)")
        print(f"   → Expected: {expected}")
        if not is_correct:
            print(f"   → ⚠️ MISMATCH")
        print()
    
    accuracy = (correct / total) * 100
    
    print("=" * 60)
    print(f"📊 FINAL RESULTS:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Failed: {len(failed_cases)} cases")
    print("=" * 60)
    
    if accuracy >= 98.0:
        print("🎉 SUCCESS! Model achieves 98%+ accuracy!")
        print("✅ Domain detection model is READY FOR PRODUCTION!")
    else:
        print("❌ FAILED! Model below 98% threshold")
        print("🔧 Needs improvement before production deployment")
        print("\n📋 Failed Cases:")
        for query, expected, detected, confidence in failed_cases:
            print(f"   • {query}: {expected} → {detected} ({confidence:.1f}%)")
    
    return accuracy >= 98.0

if __name__ == "__main__":
    main()
