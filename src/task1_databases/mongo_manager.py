"""
Task 1: MongoDB Database Implementation
Document store for flexible product metadata and reviews.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.collection import Collection
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_python_types(obj):
    """Convert NumPy types to Python native types for MongoDB"""
    if obj is None:
        return None
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [to_python_types(x) for x in obj]
    if np.issubdtype(type(obj), np.integer):
        return int(obj)
    if np.issubdtype(type(obj), np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, (str, bool)):
        return obj
    return str(obj)


class MongoManager:
    """
    MongoDB implementation for Amazon Reviews 2023.
    Optimized for flexible schema and fast document retrieval.
    """
    
    def __init__(self, host="localhost", port=27017, db_name="amazon_reviews"):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        
        # Collections
        self.products: Collection = self.db.products
        self.reviews: Collection = self.db.reviews
        self.users: Collection = self.db.users
        
    def create_indexes(self):
        """Create indexes for the 5 required queries."""
        logger.info("Creating MongoDB indexes...")
        
        # Q1: Product info by parent_asin
        self.products.create_index("parent_asin", unique=True)
        
        # Q2: Recent reviews by product
        self.reviews.create_index([("parent_asin", ASCENDING), 
                                   ("timestamp", DESCENDING)])
        
        # Q3: Text search on title and description
        self.products.create_index([("title", TEXT), 
                                   ("description", TEXT)])
        
        # Q4: User review history
        self.reviews.create_index([("user_id", ASCENDING), 
                                   ("timestamp", DESCENDING)])
        
        # Q5: Product statistics (covered by parent_asin index)
        self.reviews.create_index("parent_asin")
        
        # Additional indexes
        self.products.create_index("main_category")
        self.products.create_index("store")
        self.users.create_index("user_id", unique=True)
        
        logger.info("✓ Indexes created")
    
    def load_data(self, processed_dir: str = "data/processed",
                  sample_size: Optional[int] = None):
        """Load data into MongoDB."""
        processed_path = Path(processed_dir)
        
        # Clear existing data
        self.products.delete_many({})
        self.reviews.delete_many({})
        self.users.delete_many({})
        
        # Load products
        logger.info("Loading products into MongoDB...")
        products_df = pd.read_parquet(processed_path / "All_Beauty_metadata_cleaned.parquet")
        
        if sample_size:
            products_df = products_df.head(sample_size)
        
        # Convert to documents - with NumPy conversion
        product_docs = []
        for _, row in products_df.iterrows():
            doc = {
                "parent_asin": str(row['parent_asin']),
                "title": str(row['title']) if pd.notna(row['title']) else '',
                "description": str(row['description']) if pd.notna(row.get('description')) else '',
                "main_category": str(row['main_category']) if pd.notna(row.get('main_category')) else '',
                "average_rating": float(row['average_rating']) if pd.notna(row.get('average_rating')) else None,
                "rating_number": int(row['rating_number']) if pd.notna(row.get('rating_number')) else 0,
                "price": float(row['price']) if pd.notna(row.get('price')) else None,
                "store": str(row['store']) if pd.notna(row.get('store')) else '',
                "features": to_python_types(row.get('features', [])),
                "categories": to_python_types(row.get('categories', [])),
                "details": to_python_types(row.get('details', {})),
                "bought_together": to_python_types(row.get('bought_together', []))
            }
            product_docs.append(doc)
        
        if product_docs:
            self.products.insert_many(product_docs)
            logger.info(f"✓ Loaded {len(product_docs):,} products")
        
        loaded_asins = set(products_df['parent_asin'].tolist())
        
        # Load reviews
        logger.info("Loading reviews into MongoDB...")
        reviews_df = pd.read_parquet(processed_path / "All_Beauty_reviews_cleaned.parquet")
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(loaded_asins)]
        
        review_docs = []
        for _, row in reviews_df.iterrows():
            doc = {
                "review_id": f"{row['user_id']}_{row['parent_asin']}_{row['timestamp']}",
                "user_id": str(row['user_id']),
                "parent_asin": str(row['parent_asin']),
                "rating": int(row['rating']),
                "title": str(row['title']) if pd.notna(row.get('title')) else '',
                "text": str(row['text']) if pd.notna(row.get('text')) else '',
                "helpful_vote": int(row['helpful_vote']) if pd.notna(row.get('helpful_vote')) else 0,
                "verified_purchase": bool(row['verified_purchase']) if pd.notna(row.get('verified_purchase')) else False,
                "timestamp": row['timestamp'],
                "has_image": bool(row['has_image']) if pd.notna(row.get('has_image')) else False,
                "text_length": int(row['text_length']) if pd.notna(row.get('text_length')) else 0,
                "year": int(row['year']),
                "month": int(row['month'])
            }
            review_docs.append(doc)
        
        if review_docs:
            # Insert in batches
            batch_size = 10000
            for i in range(0, len(review_docs), batch_size):
                batch = review_docs[i:i + batch_size]
                self.reviews.insert_many(batch)
                if (i + batch_size) % 50000 == 0 or (i + batch_size) >= len(review_docs):
                    logger.info(f"  ... {min(i + len(batch), len(review_docs)):,} reviews inserted")
            
            logger.info(f"✓ Loaded {len(review_docs):,} reviews")
        
        # Pre-compute user statistics
        logger.info("Computing user statistics...")
        pipeline = [
            {"$group": {
                "_id": "$user_id",
                "review_count": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"},
                "first_review": {"$min": "$timestamp"},
                "last_review": {"$max": "$timestamp"}
            }}
        ]
        user_stats = list(self.reviews.aggregate(pipeline))
        
        if user_stats:
            for stat in user_stats:
                stat['user_id'] = stat.pop('_id')
            self.users.insert_many(user_stats)
            logger.info(f"✓ Loaded {len(user_stats):,} users")
        
        self.create_indexes()
    
    # QUERY 1: Product Information
    def query_product_info(self, parent_asin: str) -> Optional[Dict]:
        """Get product by parent ASIN."""
        return self.products.find_one({"parent_asin": parent_asin}, {"_id": 0})
    
    # QUERY 2: Recent N Reviews
    def query_recent_reviews(self, parent_asin: str, n: int = 10) -> List[Dict]:
        """Get recent N reviews for a product."""
        cursor = self.reviews.find(
            {"parent_asin": parent_asin},
            {"_id": 0}
        ).sort("timestamp", DESCENDING).limit(n)
        return list(cursor)
    
    # QUERY 3: Keyword Search
    def query_keyword_search(self, keyword: str, limit: int = 20) -> List[Dict]:
        """Full-text search in products."""
        cursor = self.products.find(
            {"$text": {"$search": keyword}},
            {"_id": 0, "score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        return list(cursor)
    
    # QUERY 4: User Review History
    def query_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's review history with product details."""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"timestamp": DESCENDING}},
            {"$limit": limit},
            {"$lookup": {
                "from": "products",
                "localField": "parent_asin",
                "foreignField": "parent_asin",
                "as": "product"
            }},
            {"$unwind": "$product"},
            {"$project": {
                "_id": 0,
                "parent_asin": 1,
                "title": "$product.title",
                "rating": 1,
                "timestamp": 1,
                "review_title": "$title",
                "verified_purchase": 1
            }}
        ]
        return list(self.reviews.aggregate(pipeline))
    
    # QUERY 5: Product Statistics
    def query_product_statistics(self, parent_asin: str) -> Optional[Dict]:
        """Get product statistics using aggregation."""
        pipeline = [
            {"$match": {"parent_asin": parent_asin}},
            {"$group": {
                "_id": "$parent_asin",
                "total_reviews": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"},
                "rating_std": {"$stdDevPop": "$rating"},
                "star_1": {"$sum": {"$cond": [{"$eq": ["$rating", 1]}, 1, 0]}},
                "star_2": {"$sum": {"$cond": [{"$eq": ["$rating", 2]}, 1, 0]}},
                "star_3": {"$sum": {"$cond": [{"$eq": ["$rating", 3]}, 1, 0]}},
                "star_4": {"$sum": {"$cond": [{"$eq": ["$rating", 4]}, 1, 0]}},
                "star_5": {"$sum": {"$cond": [{"$eq": ["$rating", 5]}, 1, 0]}},
                "verified_count": {"$sum": {"$cond": ["$verified_purchase", 1, 0]}},
                "verified_avg_rating": {"$avg": {"$cond": ["$verified_purchase", "$rating", None]}}
            }}
        ]
        result = list(self.reviews.aggregate(pipeline))
        if result:
            stats = result[0]
            stats['parent_asin'] = stats.pop('_id')
            total = stats['total_reviews']
            for i in range(1, 6):
                key = f'star_{i}'
                stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
            return stats
        return None
    
    def benchmark_queries(self, iterations: int = 10) -> pd.DataFrame:
        """Benchmark all 5 queries."""
        import random
        
        # Get samples
        sample_products = [p['parent_asin'] for p in self.products.find({}, {"parent_asin": 1}).limit(5)]
        sample_users = [u['user_id'] for u in self.users.find({}, {"user_id": 1}).limit(5)]
        
        results = []
        
        # Q1
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_product_info(random.choice(sample_products))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q1: Product Info',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Document lookup by _id'
        })
        
        # Q2
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_recent_reviews(random.choice(sample_products), n=10)
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q2: Recent Reviews',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Compound index scan'
        })
        
        # Q3
        times = []
        keywords = ['conditioner', 'shampoo', 'cream', 'oil', 'serum']
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_keyword_search(random.choice(keywords))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q3: Keyword Search',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Full-text search (text index)'
        })
        
        # Q4
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_user_history(random.choice(sample_users))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q4: User History',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Aggregation with $lookup'
        })
        
        # Q5
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_product_statistics(random.choice(sample_products))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q5: Product Stats',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Aggregation pipeline'
        })
        
        df = pd.DataFrame(results)
        df.to_csv('docs/mongo_benchmark.csv', index=False)
        logger.info("✓ Benchmark results saved to docs/mongo_benchmark.csv")
        
        return df


def main():
    """Demo of MongoDB implementation"""
    
    mongo = MongoManager()
    
    # Load data
    mongo.load_data(sample_size=10000)
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING MONGODB QUERIES")
    print("="*60)
    
    sample_product = mongo.products.find_one()["parent_asin"]
    
    print(f"\nQ1: Product Info for {sample_product}")
    result = mongo.query_product_info(sample_product)
    print(f"  Title: {result['title'][:60]}...")
    
    print(f"\nQ2: Recent 5 reviews")
    reviews = mongo.query_recent_reviews(sample_product, n=5)
    print(f"  Found {len(reviews)} reviews")
    for r in reviews[:2]:
        print(f"    {r['rating']}★: {r['title'][:40]}...")
    
    print(f"\nQ3: Keyword search 'conditioner'")
    products = mongo.query_keyword_search("conditioner", limit=3)
    print(f"  Found {len(products)} products")
    for p in products:
        print(f"    {p['title'][:50]}... (score: {p.get('score', 'N/A')})")
    
    print("\n" + "="*60)
    print("BENCHMARKING MONGODB")
    print("="*60)
    benchmark_df = mongo.benchmark_queries(iterations=5)
    print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    main()