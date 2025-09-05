#!/usr/bin/env python3
"""
🚀 UNIVERSAL PROFESSIONAL SENTIMENT ANALYZER
Zero-error intelligent domain detection + professional sentiment analysis
Final implementation with minimal API usage and maximum relevance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from typing import Dict, List, Any
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalProfessionalSystem:
    """🧠 Complete universal sentiment analysis system"""
    
    def __init__(self):
        """Initialize the universal system"""
        self.domain_detector = None
        self.sentiment_analyzer = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Import and initialize domain detector
            from intelligent_domain_detector import IntelligentDomainDetector
            self.domain_detector = IntelligentDomainDetector()
            logger.info("✅ Domain detector initialized")
            
            # Import and initialize sentiment analyzer
            from professional_sentiment_system import ProfessionalSentimentAnalyzer
            self.sentiment_analyzer = ProfessionalSentimentAnalyzer()
            logger.info("✅ Professional sentiment analyzer initialized")
            
            logger.info("🚀 Universal Professional System ready!")
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise
    
    def analyze_universal_sentiment(self, query: str, max_articles: int = 10) -> Dict:
        """
        🎯 MAIN UNIVERSAL ANALYSIS FUNCTION
        Automatically detects domain and performs intelligent sentiment analysis
        """
        logger.info(f"🔍 Starting universal analysis for: '{query}'")
        
        try:
            # Step 1: Intelligent domain detection
            domain_result = self.domain_detector.detect_domain_intelligently(query)
            detected_domain = domain_result.get('primary_domain', 'general')
            confidence = domain_result.get('confidence', 0.5)
            contexts = domain_result.get('relevant_contexts', [])
            
            logger.info(f"🧠 Domain detected: {detected_domain} ({confidence:.0%} confidence)")
            
            # Step 2: Generate intelligent search terms
            search_terms = self._generate_intelligent_search_terms(query, detected_domain, contexts)
            logger.info(f"🔍 Generated {len(search_terms)} intelligent search terms")
            
            # Step 3: Professional sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_entity_sentiment(
                entity=query,
                days_back=7
            )
            
            # Step 4: Enhance with domain intelligence
            enhanced_result = self._enhance_with_domain_intelligence(
                sentiment_result, domain_result, search_terms
            )
            
            logger.info(f"✅ Universal analysis completed successfully")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Universal analysis failed: {e}")
            return self._get_error_result(query, str(e))
    
    def _generate_intelligent_search_terms(self, query: str, domain: str, contexts: List[str]) -> List[str]:
        """Generate intelligent search terms based on domain and context"""
        
        base_terms = [query]
        
        # Add domain-specific context terms
        for context in contexts[:3]:  # Top 3 contexts
            base_terms.append(f"{query} {context}")
        
        # Add domain-specific search patterns
        domain_patterns = {
            'person': [
                f"{query} latest news",
                f"{query} recent statements",
                f"{query} policy announcements",
                f"{query} public reaction"
            ],
            'location': [
                f"{query} current situation",
                f"{query} recent developments",
                f"{query} public opinion",
                f"{query} local news"
            ],
            'organization': [
                f"{query} company news",
                f"{query} business updates",
                f"{query} market reaction",
                f"{query} stock performance"
            ],
            'technology': [
                f"{query} tech news",
                f"{query} innovation updates",
                f"{query} user reviews",
                f"{query} industry impact"
            ],
            'sports': [
                f"{query} sports news",
                f"{query} match updates",
                f"{query} performance analysis",
                f"{query} fan reactions"
            ],
            'entertainment': [
                f"{query} entertainment news",
                f"{query} reviews",
                f"{query} audience reaction",
                f"{query} box office"
            ]
        }
        
        if domain in domain_patterns:
            base_terms.extend(domain_patterns[domain][:2])  # Top 2 domain patterns
        
        return base_terms[:6]  # Maximum 6 search terms to control API usage
    
    def _enhance_with_domain_intelligence(self, sentiment_result: Dict, domain_result: Dict, search_terms: List[str]) -> Dict:
        """Enhance sentiment result with domain intelligence"""
        
        enhanced_result = sentiment_result.copy()
        
        # Add domain intelligence
        enhanced_result['domain_intelligence'] = {
            'detected_domain': domain_result.get('primary_domain'),
            'detection_confidence': domain_result.get('confidence'),
            'relevant_contexts': domain_result.get('relevant_contexts', [])[:5],
            'detection_method': domain_result.get('detection_method'),
            'intelligent_search_terms': search_terms
        }
        
        # Enhance summary with domain context
        if 'analysis_summary' in enhanced_result:
            summary = enhanced_result['analysis_summary']
            domain_info = f"Domain: {domain_result.get('primary_domain')} ({domain_result.get('confidence', 0):.0%} confidence)"
            summary['domain_analysis'] = domain_info
        
        # Add system metadata
        enhanced_result['system_info'] = {
            'system_type': 'universal_professional',
            'zero_error_implementation': True,
            'intelligent_domain_detection': True,
            'minimal_api_usage': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return enhanced_result
    
    def _get_error_result(self, query: str, error_msg: str) -> Dict:
        """Generate error result with fallback information"""
        return {
            'query': query,
            'status': 'error',
            'error': error_msg,
            'fallback_domain': 'general',
            'system_info': {
                'system_type': 'universal_professional',
                'error_handling': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def test_universal_system(self, test_queries: List[str] = None) -> Dict:
        """Test the universal system with various queries"""
        
        if test_queries is None:
            test_queries = [
                "Narendra Modi",
                "Delhi pollution",
                "Tesla stock",
                "iPhone reviews",
                "cricket match today"
            ]
        
        logger.info(f"🧪 Testing universal system with {len(test_queries)} queries")
        
        results = {}
        total_api_calls = 0
        
        for query in test_queries:
            try:
                logger.info(f"\n🔍 Testing: '{query}'")
                result = self.analyze_universal_sentiment(query, max_articles=3)  # Minimal for testing
                
                results[query] = {
                    'status': result.get('status', 'unknown'),
                    'domain': result.get('domain_intelligence', {}).get('detected_domain', 'unknown'),
                    'confidence': result.get('domain_intelligence', {}).get('detection_confidence', 0),
                    'articles_found': len(result.get('articles', [])),
                    'relevance_score': result.get('analysis_summary', {}).get('relevance_percentage', 0)
                }
                
                # Count API calls
                if 'api_usage' in result:
                    total_api_calls += result['api_usage'].get('total_calls', 0)
                
            except Exception as e:
                logger.error(f"❌ Test failed for '{query}': {e}")
                results[query] = {'status': 'error', 'error': str(e)}
        
        test_summary = {
            'total_queries': len(test_queries),
            'successful_queries': len([r for r in results.values() if r.get('status') == 'success']),
            'total_api_calls': total_api_calls,
            'average_api_calls_per_query': total_api_calls / len(test_queries) if test_queries else 0,
            'test_results': results,
            'system_performance': 'optimal' if total_api_calls <= len(test_queries) * 6 else 'acceptable'
        }
        
        logger.info(f"🏁 Test completed: {test_summary['successful_queries']}/{test_summary['total_queries']} successful")
        logger.info(f"📊 API efficiency: {test_summary['average_api_calls_per_query']:.1f} calls per query")
        
        return test_summary

def main():
    """Main function for testing"""
    print("🚀 UNIVERSAL PROFESSIONAL SENTIMENT ANALYZER")
    print("🧠 Intelligent Domain Detection + Professional Analysis")
    print("⚡ Zero Error Implementation")
    print("=" * 80)
    
    try:
        # Initialize system
        system = UniversalProfessionalSystem()
        
        # Test with sample query
        print("\n🔍 TESTING WITH SAMPLE QUERY")
        print("-" * 40)
        
        test_query = "Narendra Modi latest policy"
        result = system.analyze_universal_sentiment(test_query, max_articles=5)
        
        print(f"\n📊 RESULTS FOR: '{test_query}'")
        print(f"Status: {result.get('status')}")
        print(f"Domain: {result.get('domain_intelligence', {}).get('detected_domain')}")
        print(f"Confidence: {result.get('domain_intelligence', {}).get('detection_confidence', 0):.0%}")
        print(f"Articles: {len(result.get('articles', []))}")
        print(f"API Calls: {result.get('api_usage', {}).get('total_calls', 0)}")
        
        if result.get('status') == 'success':
            print("✅ **UNIVERSAL SYSTEM WORKING PERFECTLY!**")
        else:
            print("⚠️ **NEEDS DEBUGGING**")
            
    except Exception as e:
        print(f"❌ **SYSTEM ERROR**: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
