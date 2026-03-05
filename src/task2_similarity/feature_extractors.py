"""
Feature extraction for similarity computation.
Multiple approaches for product and user representation.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductFeatureExtractor:
    """
    Extract features from product metadata for similarity computation.
    Three approaches: TF-IDF, BERT embeddings, metadata features.
    """
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.bert_model = None
        self.svd = None
        
    def fit_tfidf(self, products_df: pd.DataFrame, max_features: int = 5000) -> np.ndarray:
        """
        Approach 1: TF-IDF on title + description.
        Good for: Text-based similarity, keyword matching.
        """
        logger.info("Fitting TF-IDF vectorizer...")
        
        # Combine title and description
        texts = (products_df['title'].fillna('') + ' ' + 
                products_df['description'].fillna(''))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore terms that appear in < 2 documents
            max_df=0.8  # Ignore terms that appear in > 80% of documents
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Reduce dimensionality for faster similarity computation
        self.svd = TruncatedSVD(n_components=256, random_state=42)
        reduced = self.svd.fit_transform(tfidf_matrix)
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = reduced / norms
        
        logger.info(f"Reduced TF-IDF shape: {normalized.shape}")
        return normalized
    
    def fit_bert_embeddings(self, products_df: pd.DataFrame, 
                           model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """
        Approach 2: BERT sentence embeddings.
        Good for: Semantic similarity, understanding meaning beyond keywords.
        """
        logger.info(f"Loading BERT model: {model_name}")
        
        self.bert_model = SentenceTransformer(model_name)
        
        # Combine title and description
        texts = (products_df['title'].fillna('') + '. ' + 
                products_df['description'].fillna(''))
        
        logger.info("Computing BERT embeddings...")
        embeddings = self.bert_model.encode(
            texts.tolist(),
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        
        logger.info(f"BERT embeddings shape: {normalized.shape}")
        return normalized
    
    def extract_metadata_features(self, products_df: pd.DataFrame) -> np.ndarray:
        """
        Approach 3: Structured metadata features.
        Good for: Price sensitivity, category preferences, brand loyalty.
        """
        logger.info("Extracting metadata features...")
        
        features_list = []
        
        for _, row in products_df.iterrows():
            feat = []
            
            # Price (normalized)
            price = row.get('price')
            if pd.notna(price) and price > 0:
                feat.append(float(price))
            else:
                feat.append(0.0)
            
            # Category one-hot (simplified - use category encoding)
            category = row.get('main_category', 'Unknown')
            # Create a simple hash-based encoding
            cat_hash = hash(category) % 100 / 100.0
            feat.append(cat_hash)
            
            # Rating features
            avg_rating = row.get('average_rating')
            if pd.notna(avg_rating):
                feat.append(float(avg_rating))
            else:
                feat.append(0.0)
            
            rating_count = row.get('rating_number')
            if pd.notna(rating_count):
                feat.append(min(int(rating_count), 1000) / 1000.0)  # Cap at 1000
            else:
                feat.append(0.0)
            
            # Has description
            has_desc = 1.0 if pd.notna(row.get('description')) and len(str(row['description'])) > 10 else 0.0
            feat.append(has_desc)
            
            # Has features list (FIXED - handle NumPy arrays properly)
            features = row.get('features', [])
            if isinstance(features, np.ndarray):
                has_features = features.size > 0 if features.size == 1 else len(features) > 0
            elif isinstance(features, list):
                has_features = len(features) > 0
            elif isinstance(features, (str, bytes)):
                has_features = len(str(features)) > 0
            else:
                has_features = features is not None
            
            # Additional check for empty arrays
            if has_features and isinstance(features, np.ndarray):
                # Check if array contains only empty/None values
                try:
                    if features.dtype == object:
                        has_features = any(f is not None and str(f) not in ['', 'nan', 'None'] for f in features.flatten())
                    else:
                        has_features = features.size > 0
                except:
                    has_features = False
            
            feat.append(1.0 if has_features else 0.0)
            
            features_list.append(feat)
        
        features_array = np.array(features_list)
        
        # Normalize each column
        for i in range(features_array.shape[1]):
            col = features_array[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                features_array[:, i] = (col - col_min) / (col_max - col_min)
        
        logger.info(f"Metadata features shape: {features_array.shape}")
        return features_array
    
    def fit_hybrid(self, products_df: pd.DataFrame, 
                   weights: List[float] = [0.5, 0.4, 0.1]) -> np.ndarray:
        """
        Approach 4: Hybrid combination of all features.
        Weights: [TF-IDF, BERT, Metadata]
        """
        logger.info("Creating hybrid feature vectors...")
        
        tfidf = self.fit_tfidf(products_df)
        bert = self.fit_bert_embeddings(products_df)
        meta = self.extract_metadata_features(products_df)
        
        # Ensure same number of samples
        n_samples = min(len(tfidf), len(bert), len(meta))
        tfidf = tfidf[:n_samples]
        bert = bert[:n_samples]
        meta = meta[:n_samples]
        
        # Weighted concatenation
        # Scale dimensions to balance contributions
        tfidf_scaled = tfidf * weights[0]
        bert_scaled = bert * weights[1]
        meta_scaled = meta * weights[2]
        
        # Pad metadata to match embedding dimensions if needed
        if meta_scaled.shape[1] < tfidf.shape[1]:
            padding = np.zeros((meta_scaled.shape[0], tfidf.shape[1] - meta_scaled.shape[1]))
            meta_scaled = np.concatenate([meta_scaled, padding], axis=1)
        
        # Combine (truncate to smallest dimension or pad)
        min_dim = min(tfidf_scaled.shape[1], bert_scaled.shape[1], meta_scaled.shape[1])
        hybrid = np.concatenate([
            tfidf_scaled[:, :min_dim],
            bert_scaled[:, :min_dim],
            meta_scaled[:, :min_dim]
        ], axis=1)
        
        # Renormalize
        norms = np.linalg.norm(hybrid, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = hybrid / norms
        
        logger.info(f"Hybrid features shape: {normalized.shape}")
        return normalized


class UserFeatureExtractor:
    """
    Extract features from user behavior for similarity computation.
    """
    
    def __init__(self):
        self.bert_model = None
        
    def extract_rating_patterns(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Approach 1: User rating behavior statistics.
        """
        logger.info("Extracting rating patterns...")
        
        user_stats = reviews_df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'helpful_vote': 'sum',
            'verified_purchase': 'mean',
            'text_length': 'mean'
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = [
            'user_id', 'avg_rating', 'rating_std', 'review_count',
            'min_rating', 'max_rating', 'total_helpful', 'verified_ratio', 'avg_text_length'
        ]
        
        # Fill NaN
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        user_stats['avg_text_length'] = user_stats['avg_text_length'].fillna(0)
        
        return user_stats
    
    def extract_category_preferences(self, reviews_df: pd.DataFrame,
                                    products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Approach 2: Category affinity vectors.
        FIXED: Properly handle the value_counts result.
        """
        logger.info("Extracting category preferences...")
        
        # Merge to get categories
        merged = reviews_df.merge(
            products_df[['parent_asin', 'main_category']], 
            on='parent_asin', 
            how='left'
        )
        
        # Get all categories first
        all_categories = products_df['main_category'].dropna().unique().tolist()
        logger.info(f"Found {len(all_categories)} unique categories")
        
        # Create category counts for each user
        user_category_counts = merged.groupby(['user_id', 'main_category']).size().reset_index(name='count')
        
        # Pivot to wide format
        category_matrix = user_category_counts.pivot(
            index='user_id', 
            columns='main_category', 
            values='count'
        ).fillna(0)
        
        # Ensure all categories are present (add missing columns with 0)
        for cat in all_categories:
            if cat not in category_matrix.columns:
                category_matrix[cat] = 0.0
        
        # Normalize rows to probabilities
        row_sums = category_matrix.sum(axis=1)
        category_matrix = category_matrix.div(row_sums, axis=0).fillna(0)
        
        # Reset index to make user_id a column
        category_matrix = category_matrix.reset_index()
        
        logger.info(f"Category preference matrix shape: {category_matrix.shape}")
        return category_matrix
    
    def extract_review_text_embeddings(self, reviews_df: pd.DataFrame,
                                      model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
        """
        Approach 3: Average of review text embeddings.
        """
        logger.info("Computing review text embeddings...")
        
        self.bert_model = SentenceTransformer(model_name)
        
        # Aggregate reviews by user
        user_texts = reviews_df.groupby('user_id').apply(
            lambda x: ' '.join(x['title'].fillna('') + '. ' + x['text'].fillna(''))
        ).reset_index()
        user_texts.columns = ['user_id', 'combined_text']
        
        # Truncate long texts
        user_texts['combined_text'] = user_texts['combined_text'].str[:1000]
        
        logger.info(f"Encoding {len(user_texts)} user texts...")
        embeddings = self.bert_model.encode(
            user_texts['combined_text'].tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        
        result = pd.DataFrame({
            'user_id': user_texts['user_id'],
            'embedding': list(normalized)
        })
        
        return result