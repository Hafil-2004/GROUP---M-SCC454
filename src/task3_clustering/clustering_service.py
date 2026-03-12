"""
Clustering service for user segmentation and product grouping.
Implements K-means, DBSCAN, and hierarchical clustering with visualization.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import sys
import os
import gc
import time
import warnings

# Suppress UMAP warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / 'task2_similarity'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusteringService:
    """
    Clustering for users and products with multiple algorithms and evaluation.
    """
    
    def __init__(self):
        self.models = {}
        self.labels = {}
        self.features = {}
        self.scalers = {}
        
    def _sample_silhouette_score(self, X: np.ndarray, labels: np.ndarray, 
                                  sample_size: int = 1000, 
                                  random_state: int = 42) -> float:
        """
        Calculate silhouette score on a random sample to save memory.
        """
        n_samples = len(X)
        if n_samples <= sample_size:
            return silhouette_score(X, labels)
        
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_samples, sample_size, replace=False)
        return silhouette_score(X[indices], labels[indices])
        
    def find_optimal_k(self, features: np.ndarray, max_k: int = 10) -> Tuple[int, List[float], List[float]]:
        """
        Find optimal K using elbow method and silhouette score.
        """
        start_time = time.time()
        logger.info(f"Finding optimal K (2-{max_k})...")
        logger.info(f"Feature matrix shape: {features.shape}")
        
        inertias = []
        silhouettes = []
        
        n_samples = features.shape[0]
        use_sampling = n_samples > 5000
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm='lloyd')
            labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            
            if use_sampling:
                sil_score = self._sample_silhouette_score(features, labels, sample_size=1000)
            else:
                sil_score = silhouette_score(features, labels)
            
            silhouettes.append(sil_score)
            logger.info(f"  K={k}: Silhouette={sil_score:.3f}, Inertia={kmeans.inertia_:.0f}")
        
        optimal_k = range(2, max_k + 1)[np.argmax(silhouettes)]
        elapsed = time.time() - start_time
        logger.info(f"✓ Optimal K selected: {optimal_k} in {elapsed:.1f}s")
        
        return optimal_k, silhouettes, inertias
    
    def cluster_kmeans(self, features: np.ndarray, ids: List[str], 
                      n_clusters: Optional[int] = None,
                      name: str = 'kmeans') -> Tuple[pd.DataFrame, Dict]:
        """
        K-means clustering with automatic K selection.
        """
        start_time = time.time()
        logger.info(f"Clustering with K-means (name={name})...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        self.scalers[name] = scaler
        
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_k(X_scaled)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm='lloyd')
        labels = kmeans.fit_predict(X_scaled)
        
        n_samples = X_scaled.shape[0]
        if n_samples > 5000:
            sil_score = self._sample_silhouette_score(X_scaled, labels, sample_size=1000)
        else:
            sil_score = silhouette_score(X_scaled, labels)
            
        ch_score = calinski_harabasz_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ K-means complete in {elapsed:.1f}s: {n_clusters} clusters")
        
        self.models[name] = kmeans
        self.labels[name] = labels
        self.features[name] = features
        
        result = pd.DataFrame({
            'id': ids,
            'cluster': labels,
            'feature_1': features[:, 0] if features.shape[1] > 0 else 0,
            'feature_2': features[:, 1] if features.shape[1] > 1 else 0
        })
        
        return result, {
            'algorithm': 'K-means',
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        }
    
    def cluster_dbscan(self, features: np.ndarray, ids: List[str],
                      eps: float = 0.5, min_samples: int = 5,
                      name: str = 'dbscan') -> Tuple[pd.DataFrame, Dict]:
        """
        DBSCAN clustering for noise-resistant segmentation.
        """
        start_time = time.time()
        logger.info(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        self.scalers[name] = scaler
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_pct = (n_noise / len(labels)) * 100
        
        if n_clusters > 1:
            mask = labels != -1
            X_mask = X_scaled[mask]
            labels_mask = labels[mask]
            
            if len(X_mask) > 5000:
                sil_score = self._sample_silhouette_score(X_mask, labels_mask, sample_size=1000)
            else:
                sil_score = silhouette_score(X_mask, labels_mask)
        else:
            sil_score = -1
        
        elapsed = time.time() - start_time
        logger.info(f"✓ DBSCAN complete in {elapsed:.1f}s: {n_clusters} clusters, {noise_pct:.1f}% noise")
        
        self.models[name] = dbscan
        self.labels[name] = labels
        self.features[name] = features
        
        result = pd.DataFrame({
            'id': ids,
            'cluster': labels,
            'feature_1': features[:, 0] if features.shape[1] > 0 else 0,
            'feature_2': features[:, 1] if features.shape[1] > 1 else 0
        })
        
        return result, {
            'algorithm': 'DBSCAN',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': noise_pct,
            'silhouette': sil_score if n_clusters > 1 else None
        }
    
    def cluster_hierarchical(self, features: np.ndarray, ids: List[str],
                            n_clusters: int = 5,
                            linkage: str = 'ward',
                            name: str = 'hierarchical') -> Tuple[pd.DataFrame, Dict]:
        """
        Hierarchical clustering with Ward linkage.
        """
        start_time = time.time()
        logger.info(f"Clustering with Hierarchical ({linkage} linkage, k={n_clusters})...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        self.scalers[name] = scaler
        
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = agg.fit_predict(X_scaled)
        
        n_samples = X_scaled.shape[0]
        if n_samples > 5000:
            sil_score = self._sample_silhouette_score(X_scaled, labels, sample_size=1000)
        else:
            sil_score = silhouette_score(X_scaled, labels)
            
        ch_score = calinski_harabasz_score(X_scaled, labels)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Hierarchical complete in {elapsed:.1f}s")
        
        self.models[name] = agg
        self.labels[name] = labels
        self.features[name] = features
        
        result = pd.DataFrame({
            'id': ids,
            'cluster': labels,
            'feature_1': features[:, 0] if features.shape[1] > 0 else 0,
            'feature_2': features[:, 1] if features.shape[1] > 1 else 0
        })
        
        return result, {
            'algorithm': 'Hierarchical',
            'linkage': linkage,
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score
        }
    
    def reduce_dimensions(self, features: np.ndarray, 
                         method: str = 'umap',
                         n_components: int = 2) -> np.ndarray:
        """
        Dimensionality reduction for visualization.
        """
        start_time = time.time()
        logger.info(f"Reducing dimensions with {method}...")
        
        n_samples = len(features)
        if n_samples > 10000:
            logger.info(f"Sampling {n_samples} to 10000 for viz...")
            rng = np.random.RandomState(42)
            indices = rng.choice(n_samples, 10000, replace=False)
            features_sample = features[indices]
        else:
            features_sample = features
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(features_sample)
        elif method == 'umap':
            # More stable UMAP settings
            reducer = umap.UMAP(
                n_components=n_components, 
                random_state=42, 
                min_dist=0.1,
                n_neighbors=15,
                metric='euclidean',
                init='random',  # Avoid spectral init issues
                transform_seed=42
            )
            reduced = reducer.fit_transform(features_sample)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Dim reduction complete in {elapsed:.1f}s")
        return reduced
    
    def visualize_clusters(self, features_2d: np.ndarray,
                          labels: np.ndarray,
                          title: str = "Cluster Visualization",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)):
        """
        Create scatter plot of clusters.
        """
        plt.figure(figsize=figsize)
        
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = [0, 0, 0, 1]
                label_name = 'Noise'
            else:
                label_name = f'Cluster {label}'
            
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[color], label=label_name, alpha=0.6, s=20, edgecolors='none')  # Smaller points
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Lower DPI for speed
            logger.info(f"✓ Saved to {save_path}")
        
        plt.close()  # Don't show, just save
    
    def plot_elbow_curve(self, inertias: List[float], silhouettes: List[float],
                        save_path: Optional[str] = None):
        """
        Plot elbow curve and silhouette scores.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        K_range = range(2, len(inertias) + 2)
        
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('K')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('K')
        ax2.set_ylabel('Silhouette')
        ax2.set_title('Silhouette Score')
        ax2.grid(True, alpha=0.3)
        optimal_k = list(K_range)[np.argmax(silhouettes)]
        ax2.axvline(optimal_k, color='green', linestyle='--', label=f'Optimal K={optimal_k}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✓ Saved elbow curve to {save_path}")
        
        plt.close()
    
    def analyze_user_clusters(self, clustered_df: pd.DataFrame,
                             reviews_df: pd.DataFrame) -> Dict:
        """Analyze user cluster characteristics."""
        logger.info("Analyzing user clusters...")
        
        analysis = {}
        
        for cluster_id in sorted(clustered_df['cluster'].unique()):
            if cluster_id == -1:
                continue
            
            cluster_users = clustered_df[clustered_df['cluster'] == cluster_id]['id']
            cluster_reviews = reviews_df[reviews_df['user_id'].isin(cluster_users)]
            
            analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_users),
                'avg_rating': float(cluster_reviews['rating'].mean()) if len(cluster_reviews) > 0 else 0,
                'rating_std': float(cluster_reviews['rating'].std()) if len(cluster_reviews) > 0 else 0,
                'total_reviews': len(cluster_reviews),
                'verified_purchase_pct': float(cluster_reviews['verified_purchase'].mean() * 100) if 'verified_purchase' in cluster_reviews else 0,
                'top_categories': cluster_reviews['main_category'].value_counts().head(3).to_dict() if 'main_category' in cluster_reviews else {}
            }
        
        return analysis
    
    def analyze_product_clusters(self, clustered_df: pd.DataFrame,
                                  products_df: pd.DataFrame) -> Dict:
        """Analyze product cluster characteristics."""
        logger.info("Analyzing product clusters...")
        
        analysis = {}
        
        for cluster_id in sorted(clustered_df['cluster'].unique()):
            if cluster_id == -1:
                continue
            
            cluster_products = clustered_df[clustered_df['cluster'] == cluster_id]['id']
            cluster_items = products_df[products_df['parent_asin'].isin(cluster_products)]
            
            analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_products),
                'avg_price': float(cluster_items['price'].mean()) if 'price' in cluster_items else 0,
                'price_range': {
                    'min': float(cluster_items['price'].min()) if 'price' in cluster_items else 0,
                    'max': float(cluster_items['price'].max()) if 'price' in cluster_items else 0
                },
                'avg_rating': float(cluster_items['average_rating'].mean()) if 'average_rating' in cluster_items else 0,
                'top_categories': cluster_items['main_category'].value_counts().head(3).to_dict() if 'main_category' in cluster_items else {},
                'sample_products': cluster_items['title'].head(3).tolist() if 'title' in cluster_items else []
            }
        
        return analysis


def main():
    """Execute Task 3: Clustering Analysis."""
    total_start = time.time()
    
    try:
        from task2_similarity.feature_extractors import UserFeatureExtractor, ProductFeatureExtractor
    except ImportError:
        logger.error("Could not import feature_extractors.")
        raise
    
    logger.info("="*70)
    logger.info("TASK 3: CLUSTERING ANALYSIS")
    logger.info("="*70)
    
    data_dir = Path("../data/processed")
    docs_dir = Path("../docs")
    docs_dir.mkdir(exist_ok=True)
    
    logger.info("\nLoading data...")
    products_df = pd.read_parquet(data_dir / "All_Beauty_metadata_cleaned.parquet")
    reviews_df = pd.read_parquet(data_dir / "All_Beauty_reviews_cleaned.parquet")
    
    logger.info(f"Loaded {len(products_df)} products, {len(reviews_df)} reviews")
    
    cluster_service = ClusteringService()
    
    # ============================================================
    # PART A: USER CLUSTERING
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("PART A: USER CLUSTERING")
    logger.info("="*70)
    
    user_extractor = UserFeatureExtractor()
    
    # Approach 1: User Rating Behavior (K-means)
    logger.info("\n--- Approach 1: User Rating Behavior (K-means) ---")
    rating_patterns = user_extractor.extract_rating_patterns(reviews_df)
    
    feature_cols = ['avg_rating', 'rating_std', 'review_count', 'verified_ratio', 'avg_text_length']
    user_features = rating_patterns[feature_cols].fillna(0).values
    
    logger.info(f"User features shape: {user_features.shape}")
    
    optimal_k, silhouettes, inertias = cluster_service.find_optimal_k(user_features, max_k=8)
    cluster_service.plot_elbow_curve(inertias, silhouettes, save_path=str(docs_dir / "user_cluster_elbow_curve.png"))
    
    user_clusters_km, km_metrics = cluster_service.cluster_kmeans(
        user_features, 
        rating_patterns['user_id'].tolist(),
        n_clusters=optimal_k,
        name='user_kmeans'
    )
    
    # Visualization with sampling
    if len(user_features) > 10000:
        rng = np.random.RandomState(42)
        viz_indices = rng.choice(len(user_features), 10000, replace=False)
        user_features_viz = user_features[viz_indices]
        labels_viz = user_clusters_km['cluster'].values[viz_indices]
    else:
        user_features_viz = user_features
        labels_viz = user_clusters_km['cluster'].values
    
    user_features_2d = cluster_service.reduce_dimensions(user_features_viz, method='umap')
    cluster_service.visualize_clusters(
        user_features_2d, labels_viz,
        title=f"User Clusters (K-means, K={optimal_k})",
        save_path=str(docs_dir / "user_clusters_kmeans.png")
    )
    
    user_analysis = cluster_service.analyze_user_clusters(user_clusters_km, reviews_df)
    
    logger.info("\nUser Cluster Profiles (K-means):")
    for cluster_name, stats in user_analysis.items():
        logger.info(f"  {cluster_name}: {stats['size']} users, Avg Rating: {stats['avg_rating']:.2f}")
    
    # Approach 2: User Category Preferences (DBSCAN) - WITH SAMPLING
    logger.info("\n--- Approach 2: User Category Preferences (DBSCAN) ---")
    cat_prefs = user_extractor.extract_category_preferences(reviews_df, products_df)
    cat_features = cat_prefs.drop('user_id', axis=1).fillna(0).values
    
    logger.info(f"Category features shape: {cat_features.shape}")
    
    # CRITICAL: Sample to 50K max for DBSCAN
    if len(cat_features) > 50000:
        logger.info(f"SAMPLING: Using 50,000 of {len(cat_features)} users for DBSCAN")
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(cat_features), 50000, replace=False)
        cat_features_db = cat_features[sample_idx]
        user_ids_db = cat_prefs['user_id'].iloc[sample_idx].tolist()
    else:
        cat_features_db = cat_features
        user_ids_db = cat_prefs['user_id'].tolist()
    
    user_clusters_db, db_metrics = cluster_service.cluster_dbscan(
        cat_features_db, user_ids_db, eps=0.3, min_samples=10, name='user_dbscan'
    )
    
    # Visualization with sampling
    if len(cat_features_db) > 10000:
        rng = np.random.RandomState(42)
        viz_idx = rng.choice(len(cat_features_db), 10000, replace=False)
        cat_features_viz = cat_features_db[viz_idx]
        labels_viz = user_clusters_db['cluster'].values[viz_idx]
    else:
        cat_features_viz = cat_features_db
        labels_viz = user_clusters_db['cluster'].values
    
    cat_features_2d = cluster_service.reduce_dimensions(cat_features_viz, method='umap')
    cluster_service.visualize_clusters(
        cat_features_2d, labels_viz,
        title="User Clusters (DBSCAN)",
        save_path=str(docs_dir / "user_clusters_dbscan.png")
    )
    
    gc.collect()
    
    # ============================================================
    # PART B: PRODUCT CLUSTERING
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("PART B: PRODUCT CLUSTERING")
    logger.info("="*70)
    
    product_extractor = ProductFeatureExtractor()
    products_sample = products_df.head(2000)
    
    # Approach 1: BERT Embeddings (K-means)
    logger.info("\n--- Approach 1: Product BERT Embeddings (K-means) ---")
    bert_embeddings = product_extractor.fit_bert_embeddings(products_sample)
    
    logger.info(f"BERT embeddings shape: {bert_embeddings.shape}")
    
    optimal_k_prod, sil_prod, inert_prod = cluster_service.find_optimal_k(bert_embeddings, max_k=8)
    cluster_service.plot_elbow_curve(inert_prod, sil_prod, save_path=str(docs_dir / "product_cluster_elbow_curve.png"))
    
    product_clusters_km, prod_km_metrics = cluster_service.cluster_kmeans(
        bert_embeddings,
        products_sample['parent_asin'].tolist(),
        n_clusters=optimal_k_prod,
        name='product_kmeans'
    )
    
    prod_features_2d = cluster_service.reduce_dimensions(bert_embeddings, method='umap')
    cluster_service.visualize_clusters(
        prod_features_2d,
        product_clusters_km['cluster'].values,
        title=f"Product Clusters (K-means, K={optimal_k_prod})",
        save_path=str(docs_dir / "product_clusters_kmeans.png")
    )
    
    product_analysis = cluster_service.analyze_product_clusters(product_clusters_km, products_sample)
    
    logger.info("\nProduct Cluster Profiles (K-means):")
    for cluster_name, stats in product_analysis.items():
        logger.info(f"  {cluster_name}: {stats['size']} products, Avg Price: ${stats['avg_price']:.2f}")
    
    # Approach 2: Metadata Features (Hierarchical)
    logger.info("\n--- Approach 2: Product Metadata (Hierarchical) ---")
    meta_features = product_extractor.extract_metadata_features(products_sample)
    
    logger.info(f"Metadata features shape: {meta_features.shape}")
    
    product_clusters_hc, prod_hc_metrics = cluster_service.cluster_hierarchical(
        meta_features,
        products_sample['parent_asin'].tolist(),
        n_clusters=5,
        linkage='ward',
        name='product_hierarchical'
    )
    
    meta_features_2d = cluster_service.reduce_dimensions(meta_features, method='umap')
    cluster_service.visualize_clusters(
        meta_features_2d,
        product_clusters_hc['cluster'].values,
        title="Product Clusters (Hierarchical)",
        save_path=str(docs_dir / "product_clusters_hierarchical.png")
    )
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)
    
    user_clusters_km.to_parquet(data_dir / "user_clusters_kmeans.parquet")
    user_clusters_db.to_parquet(data_dir / "user_clusters_dbscan.parquet")
    product_clusters_km.to_parquet(data_dir / "product_clusters_kmeans.parquet")
    product_clusters_hc.to_parquet(data_dir / "product_clusters_hierarchical.parquet")
    
    all_metrics = {
        'user_kmeans': km_metrics,
        'user_dbscan': db_metrics,
        'product_kmeans': prod_km_metrics,
        'product_hierarchical': prod_hc_metrics
    }
    
    import json
    with open(docs_dir / "clustering_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    with open(docs_dir / "user_cluster_analysis.json", "w") as f:
        json.dump(user_analysis, f, indent=2)
    
    with open(docs_dir / "product_cluster_analysis.json", "w") as f:
        json.dump(product_analysis, f, indent=2)
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n✓ Task 3 Complete in {total_elapsed/60:.1f} minutes!")
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"User K-means: {km_metrics['n_clusters']} clusters, Silhouette: {km_metrics['silhouette']:.3f}")
    logger.info(f"User DBSCAN: {db_metrics['n_clusters']} clusters, {db_metrics['noise_percentage']:.1f}% noise")
    logger.info(f"Product K-means: {prod_km_metrics['n_clusters']} clusters, Silhouette: {prod_km_metrics['silhouette']:.3f}")
    logger.info(f"Product Hierarchical: {prod_hc_metrics['n_clusters']} clusters, Silhouette: {prod_hc_metrics['silhouette']:.3f}")


if __name__ == "__main__":
    main()