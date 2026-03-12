"""
Recommendation system implementing multiple approaches.
Collaborative filtering, content-based, and hybrid methods.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import sys

# Add parent directories to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / 'task2_similarity'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Multi-approach recommendation system for Amazon products.
    """
    
    def __init__(self, reviews_df: pd.DataFrame, products_df: pd.DataFrame):
        self.reviews_df = reviews_df
        self.products_df = products_df
        
        # These will be set after building matrix
        self.product_ids = None  # Products in interaction matrix only
        self.id_to_idx = None
        self.all_product_ids = products_df['parent_asin'].tolist()  # Full catalog
        
        # User-item interaction matrix
        self.user_item_matrix = None
        self.user_ids = None
        self.item_user_matrix = None
        
        # Similarity matrices
        self.user_similarity = None
        self.item_similarity = None
        
        # Content features
        self.product_features = None
        self.feature_extractor = None
        
        # Model storage
        self.models = {}
        
    def build_user_item_matrix(self):
        """Build user-item interaction matrix from reviews."""
        logger.info("Building user-item interaction matrix...")
        
        # Create user-item matrix (ratings as values)
        matrix_df = self.reviews_df.pivot_table(
            index='user_id',
            columns='parent_asin',
            values='rating',
            aggfunc='mean',
            fill_value=0
        )
        
        self.user_ids = matrix_df.index.tolist()
        self.product_ids = matrix_df.columns.tolist()  # Only products with reviews
        self.id_to_idx = {id_: i for i, id_ in enumerate(self.product_ids)}
        
        self.user_item_matrix = matrix_df.values
        
        # Item-user matrix (transpose)
        self.item_user_matrix = self.user_item_matrix.T
        
        logger.info(f"✓ User-item matrix: {self.user_item_matrix.shape}")
        logger.info(f"  Products in matrix: {len(self.product_ids)}")
        logger.info(f"  Sparsity: {(self.user_item_matrix == 0).sum() / self.user_item_matrix.size:.2%}")
        
        return matrix_df
    
    # =========================================================================
    # APPROACH 1: Collaborative Filtering (User-Based)
    # =========================================================================
    
    def fit_user_based_cf(self, n_neighbors: int = 50):
        """
        User-based collaborative filtering using cosine similarity.
        """
        logger.info(f"\n--- Approach 1: User-Based Collaborative Filtering ---")
        
        if self.user_item_matrix is None:
            self.build_user_item_matrix()
        
        # Compute user-user similarity (cosine)
        logger.info("Computing user-user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        self.models['user_cf'] = {
            'similarity': self.user_similarity,
            'n_neighbors': n_neighbors
        }
        
        logger.info(f"✓ User-based CF ready: {self.user_similarity.shape}")
        
    def recommend_user_based(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations using user-based CF.
        """
        if 'user_cf' not in self.models:
            raise ValueError("Model not fitted. Call fit_user_based_cf() first.")
        
        # Get user index
        if user_id not in self.user_ids:
            return []  # Cold start
        
        user_idx = self.user_ids.index(user_id)
        
        # Find similar users
        similarities = self.user_similarity[user_idx]
        similar_users = np.argsort(similarities)[::-1][1:self.models['user_cf']['n_neighbors']+1]
        
        # Aggregate ratings from similar users
        scores = np.zeros(len(self.product_ids))  # Only for products in matrix
        sim_sum = 0
        
        for sim_user_idx in similar_users:
            sim_score = similarities[sim_user_idx]
            if sim_score <= 0:
                continue
            
            # Weighted ratings
            user_ratings = self.user_item_matrix[sim_user_idx]
            scores += sim_score * user_ratings
            sim_sum += sim_score
        
        if sim_sum > 0:
            scores /= sim_sum
        
        # Exclude already rated items
        user_rated = self.user_item_matrix[user_idx] > 0
        scores[user_rated] = -np.inf
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:n]
        
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:
                recommendations.append((self.product_ids[idx], float(scores[idx])))
        
        return recommendations
    
    # =========================================================================
    # APPROACH 2: Collaborative Filtering (Item-Based)
    # =========================================================================
    
    def fit_item_based_cf(self):
        """
        Item-based collaborative filtering.
        """
        logger.info(f"\n--- Approach 2: Item-Based Collaborative Filtering ---")
        
        if self.user_item_matrix is None:
            self.build_user_item_matrix()
        
        # Compute item-item similarity
        logger.info("Computing item-item similarity...")
        self.item_similarity = cosine_similarity(self.item_user_matrix)
        
        self.models['item_cf'] = {
            'similarity': self.item_similarity
        }
        
        logger.info(f"✓ Item-based CF ready: {self.item_similarity.shape}")
    
    def recommend_item_based(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations using item-based CF.
        """
        if 'item_cf' not in self.models:
            raise ValueError("Model not fitted. Call fit_item_based_cf() first.")
        
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix[user_idx]
        
        # Get items the user has rated positively (>3)
        rated_items = np.where(user_ratings > 3)[0]
        
        if len(rated_items) == 0:
            return []
        
        # Compute recommendation scores
        scores = np.zeros(len(self.product_ids))
        
        for item_idx in rated_items:
            rating = user_ratings[item_idx]
            # Weighted sum of similar items
            similar_items = self.item_similarity[item_idx]
            scores += rating * similar_items
        
        # Normalize by number of rated items
        scores /= len(rated_items)
        
        # Exclude already rated
        scores[user_ratings > 0] = -np.inf
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:n]
        
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:
                recommendations.append((self.product_ids[idx], float(scores[idx])))
        
        return recommendations
    
    # =========================================================================
    # APPROACH 3: Content-Based (TF-IDF Similarity)
    # =========================================================================
    
    def fit_content_based(self, max_products: int = 5000):
        """
        Content-based filtering using product features.
        """
        logger.info(f"\n--- Approach 3: Content-Based Filtering ---")
        
        from task2_similarity.feature_extractors import ProductFeatureExtractor
        
        # Use products that are in our interaction matrix + sample from full catalog
        matrix_products = set(self.product_ids)
        
        # Get product info for matrix products
        matrix_product_df = self.products_df[self.products_df['parent_asin'].isin(matrix_products)]
        
        # If we need more, sample from remaining
        remaining = self.products_df[~self.products_df['parent_asin'].isin(matrix_products)]
        n_sample = min(max_products - len(matrix_product_df), len(remaining))
        
        if n_sample > 0:
            sample_df = remaining.sample(n=n_sample, random_state=42)
            products_sample = pd.concat([matrix_product_df, sample_df])
        else:
            products_sample = matrix_product_df
        
        products_sample = products_sample.head(max_products)
        
        self.feature_extractor = ProductFeatureExtractor()
        
        # Use TF-IDF features
        logger.info(f"Computing TF-IDF features for {len(products_sample)} products...")
        tfidf_matrix = self.feature_extractor.fit_tfidf(products_sample)
        
        # Store the sample products and features
        self.models['content_based'] = {
            'tfidf_matrix': tfidf_matrix,
            'extractor': self.feature_extractor,
            'products_sample': products_sample,
            'product_ids': products_sample['parent_asin'].tolist()
        }
        
        logger.info(f"✓ Content-based ready: {tfidf_matrix.shape}")
    
    def recommend_content_based(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend based on user's past purchases.
        """
        if 'content_based' not in self.models:
            raise ValueError("Model not fitted. Call fit_content_based() first.")
        
        model = self.models['content_based']
        sample_ids = model['product_ids']
        tfidf_matrix = model['tfidf_matrix']
        
        # Get user's rated items that are in our sample
        user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
        
        if len(user_reviews) == 0:
            return []
        
        # Get positively rated items that are in our sample
        liked_items = user_reviews[user_reviews['rating'] >= 4]['parent_asin'].tolist()
        liked_items = [asin for asin in liked_items if asin in sample_ids]
        
        if len(liked_items) == 0:
            # Use top rated items if no 4+ ratings
            liked_items = user_reviews.nlargest(3, 'rating')['parent_asin'].tolist()
            liked_items = [asin for asin in liked_items if asin in sample_ids]
        
        if len(liked_items) == 0:
            return []
        
        # Compute similarities on-demand for liked items only
        scores = np.zeros(len(sample_ids))
        
        for asin in liked_items:
            idx = sample_ids.index(asin)
            # Compute cosine similarity for this item vs all others
            item_vector = tfidf_matrix[idx:idx+1]
            similarities = cosine_similarity(item_vector, tfidf_matrix)[0]
            scores += similarities
        
        scores /= len(liked_items)
        
        # Exclude already rated
        for asin in user_reviews['parent_asin']:
            if asin in sample_ids:
                scores[sample_ids.index(asin)] = -np.inf
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:n]
        
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:
                recommendations.append((sample_ids[idx], float(scores[idx])))
        
        return recommendations
    
    # =========================================================================
    # APPROACH 4: Hybrid (User CF + Content-Based)
    # =========================================================================
    
    def fit_hybrid(self, alpha: float = 0.5):
        """
        Hybrid approach combining collaborative and content filtering.
        """
        logger.info(f"\n--- Approach 4: Hybrid (α={alpha}) ---")
        
        # Ensure both models are fitted
        if 'user_cf' not in self.models:
            self.fit_user_based_cf()
        if 'content_based' not in self.models:
            self.fit_content_based()
        
        self.models['hybrid'] = {
            'alpha': alpha
        }
        
        logger.info(f"✓ Hybrid model ready")
    
    def recommend_hybrid(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations.
        """
        if 'hybrid' not in self.models:
            raise ValueError("Model not fitted. Call fit_hybrid() first.")
        
        alpha = self.models['hybrid']['alpha']
        
        # Get scores from both approaches
        try:
            user_recs = self.recommend_user_based(user_id, n=50)
        except:
            user_recs = []
        
        try:
            content_recs = self.recommend_content_based(user_id, n=50)
        except:
            content_recs = []
        
        # Create score dictionaries
        user_scores = {asin: score for asin, score in user_recs}
        content_scores = {asin: score for asin, score in content_recs}
        
        # Combine scores
        all_items = set(user_scores.keys()) | set(content_scores.keys())
        hybrid_scores = {}
        
        for item in all_items:
            u_score = user_scores.get(item, 0)
            c_score = content_scores.get(item, 0)
            hybrid_scores[item] = alpha * u_score + (1 - alpha) * c_score
        
        # Sort and return top N
        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n]
    
    # =========================================================================
    # COLD START HANDLING
    # =========================================================================
    
    def recommend_popularity_based(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Fallback for cold-start users: recommend popular items.
        """
        logger.info("Using popularity-based fallback...")
        
        # Compute popularity score (rating count * average rating)
        product_stats = self.reviews_df.groupby('parent_asin').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        product_stats.columns = ['parent_asin', 'rating_count', 'avg_rating']
        
        # Popularity score
        product_stats['popularity'] = product_stats['rating_count'] * product_stats['avg_rating']
        
        # Get top N
        top_products = product_stats.nlargest(n, 'popularity')
        
        return [(row['parent_asin'], row['popularity']) 
                for _, row in top_products.iterrows()]
    
    def recommend_for_new_user(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Handle cold-start: new user with no history.
        """
        # Check if user has history
        user_history = self.reviews_df[self.reviews_df['user_id'] == user_id]
        
        if len(user_history) == 0:
            logger.info(f"Cold start for user {user_id}: using popularity")
            return self.recommend_popularity_based(n)
        
        # User has some history, use standard recommendation
        return self.recommend_hybrid(user_id, n)
    
    def recommend_for_new_product(self, product_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar products for a new product.
        """
        if 'content_based' not in self.models:
            self.fit_content_based()
        
        model = self.models['content_based']
        sample_ids = model['product_ids']
        tfidf_matrix = model['tfidf_matrix']
        
        if product_id not in sample_ids:
            return []
        
        idx = sample_ids.index(product_id)
        
        # Compute similarities on-demand
        item_vector = tfidf_matrix[idx:idx+1]
        similarities = cosine_similarity(item_vector, tfidf_matrix)[0]
        
        # Get top similar items
        top_indices = np.argsort(similarities)[::-1][1:n+1]  # Exclude self
        
        return [(sample_ids[i], float(similarities[i])) for i in top_indices]
    
    # =========================================================================
    # EVALUATION FRAMEWORK
    # =========================================================================
    
    def evaluate_recommendations(self, test_users: List[str], 
                                  k: int = 10,
                                  method: str = 'hybrid') -> Dict:
        """
        Evaluate recommendation quality using Precision@K and Recall@K.
        """
        logger.info(f"\nEvaluating {method} recommendations for {len(test_users)} users...")
        
        precisions = []
        recalls = []
        
        for user_id in test_users:
            # Get user's reviews
            user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
            
            if len(user_reviews) < 5:
                continue
            
            # Split: use half for ground truth
            split_idx = len(user_reviews) // 2
            ground_truth = set(user_reviews.iloc[split_idx:]['parent_asin'])
            
            if len(ground_truth) == 0:
                continue
            
            # Get recommendations
            try:
                if method == 'user_cf':
                    recs = self.recommend_user_based(user_id, n=k)
                elif method == 'item_cf':
                    recs = self.recommend_item_based(user_id, n=k)
                elif method == 'content_based':
                    recs = self.recommend_content_based(user_id, n=k)
                elif method == 'hybrid':
                    recs = self.recommend_hybrid(user_id, n=k)
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as e:
                logger.warning(f"Error recommending for {user_id}: {e}")
                continue
            
            if len(recs) == 0:
                continue
            
            recommended_items = set([asin for asin, _ in recs])
            
            # Compute metrics
            hits = len(recommended_items & ground_truth)
            precision = hits / len(recommended_items)
            recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        results = {
            'method': method,
            'k': k,
            'n_users_evaluated': len(precisions),
            'precision_at_k': np.mean(precisions) if precisions else 0,
            'recall_at_k': np.mean(recalls) if recalls else 0,
            'f1_at_k': 2 * (np.mean(precisions) * np.mean(recalls)) / (np.mean(precisions) + np.mean(recalls)) if (precisions and recalls and np.mean(precisions) + np.mean(recalls) > 0) else 0
        }
        
        logger.info(f"  Precision@{k}: {results['precision_at_k']:.3f}")
        logger.info(f"  Recall@{k}: {results['recall_at_k']:.3f}")
        logger.info(f"  F1@{k}: {results['f1_at_k']:.3f}")
        
        return results
    
    def compare_methods(self, test_users: List[str], k: int = 10) -> pd.DataFrame:
        """
        Compare all recommendation methods.
        """
        logger.info("\n" + "="*60)
        logger.info("COMPARING RECOMMENDATION METHODS")
        logger.info("="*60)
        
        results = []
        
        for method in ['user_cf', 'item_cf', 'content_based', 'hybrid']:
            try:
                result = self.evaluate_recommendations(test_users, k=k, method=method)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {method}: {e}")
        
        return pd.DataFrame(results)


def main():
    """Execute Task 4: Recommendation System."""
    logger.info("="*70)
    logger.info("TASK 4: RECOMMENDATION SYSTEM")
    logger.info("="*70)
    
    # Setup paths
    data_dir = Path("../data/processed")
    docs_dir = Path("../docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("\nLoading data...")
    products_df = pd.read_parquet(data_dir / "All_Beauty_metadata_cleaned.parquet")
    reviews_df = pd.read_parquet(data_dir / "All_Beauty_reviews_cleaned.parquet")
    
    # Sample for faster processing
    sample_users = reviews_df['user_id'].unique()[:5000]
    reviews_sample = reviews_df[reviews_df['user_id'].isin(sample_users)]
    
    logger.info(f"Loaded {len(products_df)} products, {len(reviews_sample)} reviews")
    logger.info(f"Unique users: {reviews_sample['user_id'].nunique()}")
    
    # Initialize service
    rec_service = RecommendationService(reviews_sample, products_df)
    
    # Build interaction matrix FIRST (sets product_ids)
    rec_service.build_user_item_matrix()
    
    # =========================================================================
    # FIT ALL MODELS
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("FITTING MODELS")
    logger.info("="*70)
    
    # Approach 1: User-based CF
    rec_service.fit_user_based_cf(n_neighbors=30)
    
    # Approach 2: Item-based CF
    rec_service.fit_item_based_cf()
    
    # Approach 3: Content-based
    rec_service.fit_content_based(max_products=5000)
    
    # Approach 4: Hybrid
    rec_service.fit_hybrid(alpha=0.6)
    
    # =========================================================================
    # GENERATE RECOMMENDATIONS FOR 5 SAMPLE USERS
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("GENERATING RECOMMENDATIONS (5 Sample Users)")
    logger.info("="*70)
    
    # Select 5 users with sufficient history
    user_review_counts = reviews_sample['user_id'].value_counts()
    sample_user_ids = user_review_counts.head(5).index.tolist()
    
    all_recommendations = {}
    
    for user_id in sample_user_ids:
        logger.info(f"\n--- User: {user_id[:20]}... ---")
        user_history = reviews_sample[reviews_sample['user_id'] == user_id]
        logger.info(f"  History: {len(user_history)} reviews, avg rating: {user_history['rating'].mean():.2f}")
        
        user_recs = {}
        
        for method_name, method_func in [
            ('User-CF', rec_service.recommend_user_based),
            ('Item-CF', rec_service.recommend_item_based),
            ('Content-Based', rec_service.recommend_content_based),
            ('Hybrid', rec_service.recommend_hybrid)
        ]:
            try:
                recs = method_func(user_id, n=10)
                user_recs[method_name] = [
                    {
                        'asin': asin,
                        'title': products_df[products_df['parent_asin'] == asin]['title'].values[0] if asin in products_df['parent_asin'].values else 'Unknown',
                        'score': round(score, 3)
                    }
                    for asin, score in recs[:5]
                ]
                logger.info(f"  {method_name}: {len(recs)} recommendations")
            except Exception as e:
                logger.error(f"  {method_name}: Error - {str(e)[:100]}")
                user_recs[method_name] = []
        
        all_recommendations[user_id] = user_recs
    
    # Save recommendations
    import json
    with open(docs_dir / "recommendations_sample.json", "w") as f:
        json.dump(all_recommendations, f, indent=2)
    
    logger.info(f"\n✓ Saved sample recommendations to docs/recommendations_sample.json")
    
    # =========================================================================
    # COLD-START DEMONSTRATION
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("COLD-START HANDLING")
    logger.info("="*70)
    
    # Simulate cold-start user
    cold_start_recs = rec_service.recommend_popularity_based(n=10)
    logger.info("\nPopularity-based recommendations (for new users):")
    for i, (asin, score) in enumerate(cold_start_recs[:5], 1):
        title = products_df[products_df['parent_asin'] == asin]['title'].values
        title_str = title[0][:50] if len(title) > 0 else 'Unknown'
        logger.info(f"  {i}. {title_str}... (score: {score:.1f})")
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION")
    logger.info("="*70)
    
    # Select test users (different from training)
    test_users = user_review_counts.iloc[5:25].index.tolist()
    
    comparison_df = rec_service.compare_methods(test_users, k=10)
    
    # Save comparison
    comparison_df.to_csv(docs_dir / "recommendation_evaluation.csv", index=False)
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("TASK 4 COMPLETE")
    logger.info("="*70)
    logger.info("Deliverables:")
    logger.info("  - docs/recommendations_sample.json (Top-10 for 5 users)")
    logger.info("  - docs/recommendation_evaluation.csv (Performance metrics)")


if __name__ == "__main__":
    main()