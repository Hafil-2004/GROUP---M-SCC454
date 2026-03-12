"""
Similarity computation service for products and users.
Integrates feature extraction with vector search.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from feature_extractors import ProductFeatureExtractor, UserFeatureExtractor
from vector_stores import FaissIndex, AnnoyIndexWrapper, BruteForceIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductSimilarityService:
    """
    Service for computing product-product similarity.
    Multiple approaches with different feature representations.
    """
    
    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df.reset_index(drop=True)
        self.product_ids = products_df['parent_asin'].tolist()
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.product_ids)}
        
        self.extractor = ProductFeatureExtractor()
        
        # Store indices for different approaches
        self.indices = {}
        self.features = {}
        
    def build_indices(self):
        """Build all similarity indices."""
        logger.info("Building product similarity indices...")
        
        # Approach 1: TF-IDF + FAISS
        logger.info("\n--- Approach 1: TF-IDF ---")
        tfidf_features = self.extractor.fit_tfidf(self.products_df)
        self.features['tfidf'] = tfidf_features
        
        faiss_tfidf = FaissIndex(tfidf_features.shape[1], 'FlatIP')
        faiss_tfidf.build_index(tfidf_features, self.product_ids)
        self.indices['tfidf_faiss'] = faiss_tfidf
        
        # Approach 2: BERT Embeddings + FAISS
        logger.info("\n--- Approach 2: BERT Embeddings ---")
        bert_features = self.extractor.fit_bert_embeddings(self.products_df)
        self.features['bert'] = bert_features
        
        faiss_bert = FaissIndex(bert_features.shape[1], 'FlatIP')
        faiss_bert.build_index(bert_features, self.product_ids)
        self.indices['bert_faiss'] = faiss_bert
        
        # Approach 3: Metadata Features + Annoy
        logger.info("\n--- Approach 3: Metadata Features ---")
        meta_features = self.extractor.extract_metadata_features(self.products_df)
        self.features['metadata'] = meta_features
        
        annoy_meta = AnnoyIndexWrapper(meta_features.shape[1], 'angular')
        annoy_meta.build_index(meta_features, self.product_ids, n_trees=10)
        self.indices['metadata_annoy'] = annoy_meta
        
        # Approach 4: Hybrid (combination)
        logger.info("\n--- Approach 4: Hybrid ---")
        # Use pre-fitted extractors
        hybrid_features = np.concatenate([
            self.features['tfidf'][:, :128],  # Truncated TF-IDF
            self.features['bert'][:, :128],   # Truncated BERT
            self.features['metadata']         # Full metadata
        ], axis=1)
        
        # Normalize
        norms = np.linalg.norm(hybrid_features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        hybrid_features = hybrid_features / norms
        
        self.features['hybrid'] = hybrid_features
        
        faiss_hybrid = FaissIndex(hybrid_features.shape[1], 'FlatIP')
        faiss_hybrid.build_index(hybrid_features, self.product_ids)
        self.indices['hybrid_faiss'] = faiss_hybrid
        
        logger.info("\n✓ All product similarity indices built")
        
    def find_similar_products(self, product_id: str, n: int = 10,
                             method: str = 'bert_faiss') -> List[Dict]:
        """
        Find N most similar products using specified method.
        
        Methods: 'tfidf_faiss', 'bert_faiss', 'metadata_annoy', 'hybrid_faiss'
        """
        if method not in self.indices:
            raise ValueError(f"Unknown method: {method}")
        
        # Get query vector
        idx = self.id_to_idx.get(product_id)
        if idx is None:
            return []
        
        feature_key = method.split('_')[0]
        query_vector = self.features[feature_key][idx]
        
        # Search
        results = self.indices[method].search(query_vector, n + 1)  # +1 to exclude self
        
        # Format results (exclude the query product itself)
        similar_products = []
        for sim_id, score in results:
            if sim_id != product_id:
                product_info = self.products_df[
                    self.products_df['parent_asin'] == sim_id
                ].iloc[0]
                similar_products.append({
                    'parent_asin': sim_id,
                    'title': product_info['title'],
                    'similarity_score': score,
                    'price': product_info.get('price'),
                    'main_category': product_info.get('main_category')
                })
            if len(similar_products) >= n:
                break
        
        return similar_products
    
    def compare_methods(self, sample_product_ids: List[str], n: int = 10) -> Dict:
        """Compare similarity results across different methods."""
        comparison = {}
        
        for product_id in sample_product_ids:
            product_results = {}
            for method in ['tfidf_faiss', 'bert_faiss', 'metadata_annoy', 'hybrid_faiss']:
                results = self.find_similar_products(product_id, n=n, method=method)
                product_results[method] = results
            comparison[product_id] = product_results
        
        return comparison


class UserSimilarityService:
    """
    Service for computing user-user similarity.
    """
    
    def __init__(self, reviews_df: pd.DataFrame, products_df: pd.DataFrame):
        self.reviews_df = reviews_df
        self.products_df = products_df
        self.extractor = UserFeatureExtractor()
        
        self.user_ids = []
        self.indices = {}
        self.features = {}
        
    def build_indices(self):
        """Build user similarity indices."""
        logger.info("Building user similarity indices...")
        
        # Approach 1: Rating Patterns + FAISS
        logger.info("\n--- Approach 1: Rating Behavior Patterns ---")
        rating_patterns = self.extractor.extract_rating_patterns(self.reviews_df)
        self.user_ids = rating_patterns['user_id'].tolist()
        
        # Create feature vectors (exclude user_id)
        feature_cols = [c for c in rating_patterns.columns if c != 'user_id']
        rating_features = rating_patterns[feature_cols].fillna(0).values
        
        # Normalize
        norms = np.linalg.norm(rating_features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        rating_features = rating_features / norms
        
        self.features['rating'] = rating_features
        
        faiss_rating = FaissIndex(rating_features.shape[1], 'FlatIP')
        faiss_rating.build_index(rating_features, self.user_ids)
        self.indices['rating_faiss'] = faiss_rating
        
        # Approach 2: Category Preferences
        logger.info("\n--- Approach 2: Category Preferences ---")
        cat_prefs = self.extractor.extract_category_preferences(
            self.reviews_df, self.products_df
        )
        
        # Convert to matrix (exclude user_id column)
        cat_matrix = cat_prefs.drop('user_id', axis=1).fillna(0).values
        
        self.features['category'] = cat_matrix
        
        faiss_cat = FaissIndex(cat_matrix.shape[1], 'FlatIP')
        faiss_cat.build_index(cat_matrix, cat_prefs['user_id'].tolist())
        self.indices['category_faiss'] = faiss_cat
        
        # Approach 3: Review Text Embeddings
        logger.info("\n--- Approach 3: Review Text Embeddings ---")
        text_embeddings = self.extractor.extract_review_text_embeddings(self.reviews_df)
        
        # Stack embeddings
        embeddings = np.vstack(text_embeddings['embedding'].values)
        self.features['text'] = embeddings
        
        faiss_text = FaissIndex(embeddings.shape[1], 'FlatIP')
        faiss_text.build_index(embeddings, text_embeddings['user_id'].tolist())
        self.indices['text_faiss'] = faiss_text
        
        logger.info("\n✓ All user similarity indices built")
        
    def find_similar_users(self, user_id: str, n: int = 10,
                          method: str = 'rating_faiss') -> List[Dict]:
        """Find N most similar users."""
        if method not in self.indices:
            raise ValueError(f"Unknown method: {method}")
        
        # Get query vector
        idx = self.user_ids.index(user_id) if user_id in self.user_ids else None
        if idx is None:
            return []
        
        feature_key = method.split('_')[0]
        query_vector = self.features[feature_key][idx]
        
        # Search
        results = self.indices[method].search(query_vector, n + 1)
        
        # Format results
        similar_users = []
        for sim_id, score in results:
            if sim_id != user_id:
                # Get user stats
                user_reviews = self.reviews_df[self.reviews_df['user_id'] == sim_id]
                similar_users.append({
                    'user_id': sim_id,
                    'similarity_score': score,
                    'review_count': len(user_reviews),
                    'avg_rating': user_reviews['rating'].mean()
                })
            if len(similar_users) >= n:
                break
        
        return similar_users


def main():
    """Demo similarity computation."""
    from pathlib import Path
    
    # Load data
    logger.info("Loading processed data...")
    products_df = pd.read_parquet("../data/processed/All_Beauty_metadata_cleaned.parquet")
    reviews_df = pd.read_parquet("../data/processed/All_Beauty_reviews_cleaned.parquet")
    
    # Sample for faster demo
    products_sample = products_df.head(5000)
    reviews_sample = reviews_df[reviews_df['parent_asin'].isin(products_sample['parent_asin'])]
    
    # Product similarity
    logger.info("\n" + "="*60)
    logger.info("PRODUCT SIMILARITY")
    logger.info("="*60)
    
    product_sim = ProductSimilarityService(products_sample)
    product_sim.build_indices()
    
    # Test with sample product
    sample_product = products_sample.iloc[0]['parent_asin']
    sample_title = products_sample.iloc[0]['title']
    
    print(f"\nQuery Product: {sample_title[:60]}...")
    print(f"ASIN: {sample_product}\n")
    
    for method in ['tfidf_faiss', 'bert_faiss', 'metadata_annoy']:
        print(f"\n--- {method.upper()} ---")
        similar = product_sim.find_similar_products(sample_product, n=5, method=method)
        for i, prod in enumerate(similar, 1):
            print(f"{i}. {prod['title'][:50]}... (score: {prod['similarity_score']:.3f})")
    
    # User similarity
    logger.info("\n" + "="*60)
    logger.info("USER SIMILARITY")
    logger.info("="*60)
    
    user_sim = UserSimilarityService(reviews_sample, products_sample)
    user_sim.build_indices()
    
    sample_user = user_sim.user_ids[0]
    print(f"\nQuery User: {sample_user[:30]}...")
    
    for method in ['rating_faiss', 'category_faiss', 'text_faiss']:
        print(f"\n--- {method.upper()} ---")
        similar = user_sim.find_similar_users(sample_user, n=5, method=method)
        for i, user in enumerate(similar, 1):
            print(f"{i}. {user['user_id'][:30]}... (score: {user['similarity_score']:.3f}, "
                  f"reviews: {user['review_count']})")
    
    # Save comparison for report
    logger.info("\n" + "="*60)
    logger.info("SAVING COMPARISON RESULTS")
    logger.info("="*60)
    
    sample_products = products_sample.head(5)['parent_asin'].tolist()
    comparison = product_sim.compare_methods(sample_products, n=10)
    
    # Save to file for report
    import json
    with open('../docs/similarity_comparison.json', 'w') as f:
        # Convert to serializable format
        serializable = {}
        for pid, methods in comparison.items():
            serializable[pid] = {
                m: [{'asin': p['parent_asin'], 'title': p['title'][:100], 
                     'score': p['similarity_score']} for p in results]
                for m, results in methods.items()
            }
        json.dump(serializable, f, indent=2)
    
    logger.info("✓ Results saved to docs/similarity_comparison.json")


if __name__ == "__main__":
    main()