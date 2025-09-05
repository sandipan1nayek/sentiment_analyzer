"""
Advanced Relevance Filtering System
Implements semantic search, entity matching, and intelligent query expansion
for highly accurate sentiment analysis
"""

import spacy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRelevanceFilter:
    """
    Professional relevance filtering using multiple approaches:
    1. Intelligent query expansion
    2. Named Entity Recognition (NER)
    3. Semantic similarity scoring
    4. Context-aware filtering
    """
    
    def __init__(self):
        try:
            # Load SpaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.error("SpaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        try:
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
            
        # Initialize query expansion patterns
        self.query_patterns = self._initialize_query_patterns()
        
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize intelligent query expansion patterns"""
        return {
            # Political figures
            'narendra modi': [
                '"Narendra Modi"', '"PM Modi"', '"Prime Minister Modi"',
                '"Modi government"', '"Indian PM"', '"Modi administration"'
            ],
            'modi': [
                '"Narendra Modi"', '"PM Modi"', '"Prime Minister Modi"'
            ],
            'joe biden': [
                '"Joe Biden"', '"President Biden"', '"Biden administration"',
                '"US President"'
            ],
            'biden': [
                '"Joe Biden"', '"President Biden"', '"Biden administration"'
            ],
            'donald trump': [
                '"Donald Trump"', '"President Trump"', '"Trump campaign"',
                '"former president Trump"'
            ],
            'trump': [
                '"Donald Trump"', '"President Trump"', '"Trump campaign"'
            ],
            
            # Technology terms
            'ai': [
                '"artificial intelligence"', '"AI technology"', '"machine learning"',
                '"AI development"', '"AI innovation"', '"AI systems"'
            ],
            'artificial intelligence': [
                '"artificial intelligence"', '"AI technology"', '"machine learning"',
                '"deep learning"', '"neural networks"'
            ],
            'chatgpt': [
                '"ChatGPT"', '"OpenAI"', '"AI chatbot"', '"GPT model"'
            ],
            'openai': [
                '"OpenAI"', '"ChatGPT"', '"GPT"', '"Sam Altman"'
            ],
            
            # Companies
            'tesla': [
                '"Tesla Inc"', '"Tesla Motors"', '"Elon Musk Tesla"',
                '"Tesla stock"', '"Tesla vehicle"'
            ],
            'apple': [
                '"Apple Inc"', '"Apple company"', '"iPhone"', '"Tim Cook"',
                '"Apple stock"', '"iOS"'
            ],
            'microsoft': [
                '"Microsoft Corporation"', '"Microsoft company"', '"Satya Nadella"',
                '"Microsoft stock"', '"Windows"'
            ],
            'google': [
                '"Google"', '"Alphabet Inc"', '"Sundar Pichai"', '"Google search"'
            ],
            'meta': [
                '"Meta"', '"Facebook"', '"Mark Zuckerberg"', '"Instagram"'
            ],
            
            # Cryptocurrencies
            'bitcoin': [
                '"Bitcoin"', '"BTC"', '"cryptocurrency Bitcoin"', '"Bitcoin price"'
            ],
            'ethereum': [
                '"Ethereum"', '"ETH"', '"Ethereum blockchain"', '"Ethereum price"'
            ],
            'crypto': [
                '"cryptocurrency"', '"crypto market"', '"digital currency"', '"blockchain"'
            ],
            
            # General topics
            'climate change': [
                '"climate change"', '"global warming"', '"climate crisis"',
                '"carbon emissions"', '"climate policy"'
            ],
            'covid': [
                '"COVID-19"', '"coronavirus"', '"pandemic"', '"COVID vaccine"'
            ],
            'ukraine': [
                '"Ukraine"', '"Russia Ukraine"', '"Ukrainian war"', '"Zelensky"'
            ]
        }
    
    def expand_search_queries(self, topic: str) -> List[str]:
        """Intelligently expand search queries for better relevance"""
        topic_lower = topic.lower().strip()
        
        # Check for exact matches in patterns
        if topic_lower in self.query_patterns:
            queries = self.query_patterns[topic_lower].copy()
        else:
            # Check for partial matches
            queries = []
            for key, patterns in self.query_patterns.items():
                if key in topic_lower or topic_lower in key:
                    queries.extend(patterns[:3])  # Limit to top 3 patterns
            
            # If no patterns found, create basic variations
            if not queries:
                queries = [
                    f'"{topic}"',  # Exact phrase
                    topic,
                    f'{topic} news',
                    f'{topic} update'
                ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        logger.info(f"Expanded '{topic}' to {len(unique_queries)} targeted queries")
        return unique_queries[:5]  # Limit to top 5 queries
    
    def filter_by_relevance(self, df: pd.DataFrame, topic: str, threshold: float = 0.6) -> pd.DataFrame:
        """Apply multi-layer relevance filtering"""
        if df.empty:
            return df
            
        logger.info(f"Applying relevance filtering for topic: '{topic}' (threshold: {threshold})")
        
        # Step 1: Basic keyword relevance
        df['keyword_score'] = df['text'].apply(lambda x: self._calculate_keyword_score(x, topic))
        
        # Step 2: Entity relevance (if SpaCy is available)
        if self.nlp:
            df['entity_score'] = df['text'].apply(lambda x: self._calculate_entity_score(x, topic))
        else:
            df['entity_score'] = 0.5  # Default score
        
        # Step 3: Semantic similarity (if model is available)
        if self.sentence_model:
            df['semantic_score'] = self._calculate_semantic_scores(df['text'].tolist(), topic)
        else:
            df['semantic_score'] = 0.5  # Default score
        
        # Step 4: Combined relevance score
        df['relevance_score'] = (
            0.4 * df['keyword_score'] + 
            0.3 * df['entity_score'] + 
            0.3 * df['semantic_score']
        )
        
        # Step 5: Filter by threshold
        relevant_df = df[df['relevance_score'] >= threshold].copy()
        
        # Step 6: Sort by relevance and limit results
        relevant_df = relevant_df.sort_values('relevance_score', ascending=False)
        relevant_df = relevant_df.head(100)  # Keep top 100 most relevant
        
        logger.info(f"Filtered from {len(df)} to {len(relevant_df)} highly relevant items")
        
        return relevant_df
    
    def _calculate_keyword_score(self, text: str, topic: str) -> float:
        """Calculate keyword-based relevance score"""
        if not text or not topic:
            return 0.0
            
        text_lower = text.lower()
        topic_lower = topic.lower()
        
        score = 0.0
        
        # Exact phrase match (highest score)
        if topic_lower in text_lower:
            score += 1.0
        
        # Individual word matches
        topic_words = topic_lower.split()
        text_words = text_lower.split()
        
        for word in topic_words:
            if len(word) > 2:  # Skip very short words
                if word in text_words:
                    score += 0.3
        
        # Normalize by topic complexity
        max_score = 1.0 + (len(topic_words) * 0.3)
        return min(score / max_score, 1.0)
    
    def _calculate_entity_score(self, text: str, topic: str) -> float:
        """Calculate entity-based relevance using NER"""
        if not self.nlp or not text:
            return 0.5
            
        try:
            # Process text with SpaCy
            doc = self.nlp(text)
            
            # Extract entities
            entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
            entity_texts = [ent[0] for ent in entities]
            
            topic_lower = topic.lower()
            score = 0.0
            
            # Direct entity match
            if any(topic_lower in entity_text for entity_text in entity_texts):
                score += 1.0
            
            # Entity type matching
            if self._is_person_topic(topic_lower):
                person_entities = [ent for ent in entities if ent[1] == 'PERSON']
                if person_entities:
                    score += 0.5
            
            if self._is_organization_topic(topic_lower):
                org_entities = [ent for ent in entities if ent[1] in ['ORG', 'PRODUCT']]
                if org_entities:
                    score += 0.5
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in entity scoring: {e}")
            return 0.5
    
    def _calculate_semantic_scores(self, texts: List[str], topic: str) -> np.ndarray:
        """Calculate semantic similarity scores using sentence transformers"""
        if not self.sentence_model or not texts:
            return np.array([0.5] * len(texts))
            
        try:
            # Create query embedding
            query_embedding = self.sentence_model.encode([f"News about {topic}"])
            
            # Create text embeddings
            text_embeddings = self.sentence_model.encode(texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, text_embeddings)[0]
            
            # Normalize to 0-1 range
            normalized_scores = (similarities + 1) / 2
            
            return normalized_scores
            
        except Exception as e:
            logger.warning(f"Error in semantic scoring: {e}")
            return np.array([0.5] * len(texts))
    
    def _is_person_topic(self, topic: str) -> bool:
        """Check if topic is likely a person's name"""
        person_indicators = ['modi', 'biden', 'trump', 'musk', 'gates', 'bezos', 'cook']
        return any(indicator in topic for indicator in person_indicators)
    
    def _is_organization_topic(self, topic: str) -> bool:
        """Check if topic is likely an organization"""
        org_indicators = ['tesla', 'apple', 'microsoft', 'google', 'meta', 'amazon', 'netflix']
        return any(indicator in topic for indicator in org_indicators)
    
    def remove_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove noisy, irrelevant, or low-quality content"""
        if df.empty:
            return df
            
        original_count = len(df)
        
        # Remove very short texts
        df = df[df['text'].str.len() >= 20]
        
        # Remove promotional/spam content
        spam_patterns = [
            r'click here', r'buy now', r'subscribe', r'advertisement',
            r'sponsored', r'promotion', r'discount', r'sale'
        ]
        
        spam_regex = '|'.join(spam_patterns)
        df = df[~df['text'].str.contains(spam_regex, case=False, na=False)]
        
        # Remove duplicates and near-duplicates
        df = df.drop_duplicates(subset=['text'])
        
        # Remove texts that are too generic
        generic_patterns = [
            r'^breaking:?\s*$', r'^update:?\s*$', r'^news:?\s*$'
        ]
        
        for pattern in generic_patterns:
            df = df[~df['text'].str.match(pattern, case=False)]
        
        logger.info(f"Noise removal: {original_count} → {len(df)} items")
        
        return df.reset_index(drop=True)

# Initialize global filter instance
relevance_filter = AdvancedRelevanceFilter()

def apply_relevance_filtering(df: pd.DataFrame, topic: str, threshold: float = 0.6) -> pd.DataFrame:
    """Apply advanced relevance filtering to dataframe"""
    return relevance_filter.filter_by_relevance(df, topic, threshold)

def expand_query(topic: str) -> List[str]:
    """Expand search query intelligently"""
    return relevance_filter.expand_search_queries(topic)

def clean_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Remove noise from data"""
    return relevance_filter.remove_noise(df)

if __name__ == "__main__":
    # Test the relevance filter
    test_data = pd.DataFrame({
        'text': [
            'Narendra Modi announces new policy',
            'Indian stock market rises',
            'Tesla stock price increases',
            'Elon Musk tweets about AI',
            'Random news about weather'
        ]
    })
    
    print("Testing relevance filter for 'Narendra Modi':")
    filtered = apply_relevance_filtering(test_data, 'Narendra Modi', 0.5)
    print(filtered[['text', 'relevance_score']])
