"""
Named Entity Recognition module using SpaCy
"""

import pandas as pd
import spacy
from collections import Counter
import logging
from typing import List, Dict, Tuple
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """Named Entity Recognition using SpaCy"""
    
    def __init__(self):
        """Initialize SpaCy model"""
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
            logger.info(f"SpaCy model '{config.SPACY_MODEL}' loaded successfully")
        except OSError:
            logger.error(f"SpaCy model '{config.SPACY_MODEL}' not found. Please install it using: python -m spacy download {config.SPACY_MODEL}")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            List[Tuple[str, str]]: List of (entity_text, entity_label) tuples
        """
        if not self.nlp or not isinstance(text, str) or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Filter for relevant entity types
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    entities.append((ent.text.strip(), ent.label_))
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_entities_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entities from all texts in DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with 'text' column
            
        Returns:
            pd.DataFrame: DataFrame with 'entities' column added
        """
        if df.empty or not self.nlp:
            logger.warning("Empty DataFrame or SpaCy model not loaded")
            return df
        
        logger.info(f"Extracting entities from {len(df)} texts")
        
        df_with_entities = df.copy()
        df_with_entities['entities'] = df_with_entities['text'].apply(self.extract_entities)
        
        logger.info("Entity extraction completed")
        return df_with_entities
    
    def get_top_entities(self, df: pd.DataFrame, entity_type: str = None, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top entities by frequency
        
        Args:
            df (pd.DataFrame): DataFrame with 'entities' column
            entity_type (str): Specific entity type to filter by (optional)
            top_n (int): Number of top entities to return
            
        Returns:
            List[Tuple[str, int]]: List of (entity, count) tuples
        """
        if df.empty or 'entities' not in df.columns:
            return []
        
        all_entities = []
        
        for entities_list in df['entities']:
            if isinstance(entities_list, list):
                for entity_text, entity_label in entities_list:
                    if entity_type is None or entity_label == entity_type:
                        # Clean entity text
                        clean_entity = entity_text.strip().title()
                        if len(clean_entity) > 1:  # Filter out single characters
                            all_entities.append(clean_entity)
        
        # Count entities and return top N
        entity_counter = Counter(all_entities)
        return entity_counter.most_common(top_n)
    
    def get_entities_by_sentiment(self, df: pd.DataFrame, sentiment: str) -> List[Tuple[str, int]]:
        """
        Get top entities for a specific sentiment
        
        Args:
            df (pd.DataFrame): DataFrame with 'entities' and 'sentiment_label' columns
            sentiment (str): Sentiment to filter by ('Positive', 'Negative', 'Neutral')
            
        Returns:
            List[Tuple[str, int]]: List of (entity, count) tuples
        """
        if df.empty or 'entities' not in df.columns or 'sentiment_label' not in df.columns:
            return []
        
        # Filter by sentiment
        sentiment_df = df[df['sentiment_label'] == sentiment]
        
        return self.get_top_entities(sentiment_df, top_n=10)

def create_entity_summary(df: pd.DataFrame) -> Dict:
    """
    Create a comprehensive entity summary
    
    Args:
        df (pd.DataFrame): DataFrame with entity extraction results
        
    Returns:
        Dict: Entity summary statistics
    """
    if df.empty or 'entities' not in df.columns:
        return {
            'total_entities': 0,
            'unique_entities': 0,
            'entity_types': {},
            'top_entities': []
        }
    
    extractor = EntityExtractor()
    
    # Count total and unique entities
    all_entities = []
    entity_types = Counter()
    
    for entities_list in df['entities']:
        if isinstance(entities_list, list):
            for entity_text, entity_label in entities_list:
                all_entities.append(entity_text.strip().title())
                entity_types[entity_label] += 1
    
    total_entities = len(all_entities)
    unique_entities = len(set(all_entities))
    top_entities = extractor.get_top_entities(df, top_n=15)
    
    return {
        'total_entities': total_entities,
        'unique_entities': unique_entities,
        'entity_types': dict(entity_types),
        'top_entities': top_entities
    }

def get_entities_for_wordcloud(df: pd.DataFrame, sentiment: str = None) -> str:
    """
    Get entities as text for word cloud generation
    
    Args:
        df (pd.DataFrame): DataFrame with entity extraction results
        sentiment (str): Optional sentiment filter
        
    Returns:
        str: Space-separated entity text for word cloud
    """
    if df.empty or 'entities' not in df.columns:
        return ""
    
    # Filter by sentiment if specified
    filtered_df = df
    if sentiment and 'sentiment_label' in df.columns:
        filtered_df = df[df['sentiment_label'] == sentiment]
    
    entities_text = []
    
    for entities_list in filtered_df['entities']:
        if isinstance(entities_list, list):
            for entity_text, entity_label in entities_list:
                # Include relevant entity types
                if entity_label in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                    clean_entity = entity_text.strip().title()
                    if len(clean_entity) > 1:
                        entities_text.append(clean_entity)
    
    return ' '.join(entities_text)

if __name__ == "__main__":
    # Test the entity extractor
    test_data = pd.DataFrame({
        'text': [
            "Apple Inc. CEO Tim Cook announced new iPhone features in California.",
            "Microsoft and Google are competing in the AI market.",
            "Tesla's Elon Musk discussed electric vehicles in Austin, Texas."
        ],
        'sentiment_label': ['Positive', 'Neutral', 'Positive']
    })
    
    extractor = EntityExtractor()
    if extractor.nlp:
        results = extractor.extract_entities_from_dataframe(test_data)
        
        print("Entity Extraction Results:")
        for idx, row in results.iterrows():
            print(f"Text: {row['text'][:50]}...")
            print(f"Entities: {row['entities']}")
            print()
        
        summary = create_entity_summary(results)
        print(f"Entity Summary: {summary}")
    else:
        print("SpaCy model not available for testing")
