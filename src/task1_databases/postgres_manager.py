"""
Task 1: PostgreSQL Database Implementation
Handles schema design, data loading, and query execution for relational database.
"""

import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time
from contextlib import contextmanager
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database_if_not_exists(host="localhost", port=5432, user="postgres", password="Hafil2004", dbname="amazon_reviews"):
    """Create database if it doesn't exist"""
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        exists = cur.fetchone()
        
        if not exists:
            cur.execute(f"CREATE DATABASE {dbname}")
            logger.info(f"✓ Created database: {dbname}")
        else:
            logger.info(f"✓ Database already exists: {dbname}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False


class PostgresManager:
    """PostgreSQL implementation for Amazon Reviews 2023."""
    
    def __init__(self, host="localhost", port=5432, user="postgres", 
                 password="Hafil2004", dbname="amazon_reviews"):
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": dbname
        }
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def create_schema(self):
        """Create optimized schema for the 5 required queries."""
        logger.info("Creating PostgreSQL schema...")
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            cur.execute("""
                DROP TABLE IF EXISTS reviews CASCADE;
                DROP TABLE IF EXISTS products CASCADE;
                DROP TABLE IF EXISTS users CASCADE;
            """)
            
            # Products table
            cur.execute("""
                CREATE TABLE products (
                    parent_asin VARCHAR(20) PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    main_category VARCHAR(100),
                    average_rating DECIMAL(3,2),
                    rating_number INTEGER,
                    price DECIMAL(10,2),
                    store VARCHAR(255),
                    features JSONB,
                    categories JSONB,
                    details JSONB,
                    bought_together JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Full-text search index
            cur.execute("""
                CREATE INDEX idx_products_fts ON products 
                USING gin(to_tsvector('english', title || ' ' || COALESCE(description, '')));
            """)
            
            # Users table
            cur.execute("""
                CREATE TABLE users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    review_count INTEGER DEFAULT 0,
                    avg_rating DECIMAL(3,2),
                    first_review_date TIMESTAMP,
                    last_review_date TIMESTAMP
                );
            """)
            
            # Reviews table - partitioned
            cur.execute("""
                CREATE TABLE reviews (
                    review_id BIGSERIAL,
                    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
                    parent_asin VARCHAR(20) NOT NULL REFERENCES products(parent_asin),
                    rating SMALLINT CHECK (rating BETWEEN 1 AND 5),
                    review_title TEXT,
                    review_text TEXT,
                    helpful_vote INTEGER DEFAULT 0,
                    verified_purchase BOOLEAN DEFAULT FALSE,
                    review_timestamp TIMESTAMP NOT NULL,
                    has_image BOOLEAN DEFAULT FALSE,
                    text_length INTEGER,
                    year INTEGER,
                    month INTEGER,
                    PRIMARY KEY (review_id, review_timestamp)
                ) PARTITION BY RANGE (review_timestamp);
            """)
            
            # Create partitions
            for year in range(2000, 2025):
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS reviews_y{year} 
                    PARTITION OF reviews
                    FOR VALUES FROM ('{year}-01-01') TO ('{year+1}-01-01');
                """)
            
            # Indexes
            cur.execute("""
                CREATE INDEX idx_reviews_product_time ON reviews(parent_asin, review_timestamp DESC);
                CREATE INDEX idx_reviews_user_time ON reviews(user_id, review_timestamp DESC);
                CREATE INDEX idx_reviews_product_rating ON reviews(parent_asin, rating);
                CREATE INDEX idx_reviews_timestamp ON reviews(review_timestamp);
                CREATE INDEX idx_products_category ON products(main_category);
                CREATE INDEX idx_products_store ON products(store);
            """)
            
            logger.info("✓ Schema created with partitions and indexes")
    
    def load_data(self, processed_dir: str = "data/processed", 
                  sample_size: Optional[int] = None):
        """Load processed Parquet data into PostgreSQL."""
        processed_path = Path(processed_dir)
        
        def to_json_serializable(obj):
            """Convert NumPy arrays to JSON-safe Python objects"""
            if obj is None:
                return None
            if isinstance(obj, (np.ndarray, list, tuple)):
                return [to_json_serializable(x) for x in obj]
            if np.issubdtype(type(obj), np.integer):
                return int(obj)
            if np.issubdtype(type(obj), np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, dict):
                return {k: to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (str, int, float, bool)):
                return obj
            return str(obj)
        
        def safe_json_dump(obj):
            """Safely convert to JSON string"""
            try:
                serializable = to_json_serializable(obj)
                return json.dumps(serializable) if serializable is not None else '{}'
            except (TypeError, ValueError):
                return '{}'
        
        # Load products
        logger.info("Loading products...")
        products_df = pd.read_parquet(processed_path / "All_Beauty_metadata_cleaned.parquet")
        
        if sample_size:
            products_df = products_df.head(sample_size)
        
        # Store the ASINs we loaded for filtering reviews later
        loaded_asins = set(products_df['parent_asin'].tolist())
        logger.info(f"Will load {len(loaded_asins):,} products")
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            # Batch insert products
            products_data = []
            for _, row in products_df.iterrows():
                avg_rating = row.get('average_rating')
                if pd.isna(avg_rating):
                    avg_rating = None
                else:
                    avg_rating = float(avg_rating)
                
                rating_number = row.get('rating_number')
                if pd.isna(rating_number):
                    rating_number = 0
                else:
                    rating_number = int(rating_number)
                
                price = row.get('price')
                if pd.isna(price):
                    price = None
                else:
                    price = float(price)
                
                products_data.append((
                    str(row['parent_asin']),
                    str(row['title']) if pd.notna(row['title']) else '',
                    str(row['description']) if pd.notna(row.get('description')) else '',
                    str(row['main_category']) if pd.notna(row.get('main_category')) else '',
                    avg_rating,
                    rating_number,
                    price,
                    str(row['store']) if pd.notna(row.get('store')) else '',
                    safe_json_dump(row.get('features')),
                    safe_json_dump(row.get('categories')),
                    safe_json_dump(row.get('details')),
                    safe_json_dump(row.get('bought_together'))
                ))
            
            execute_batch(cur, """
                INSERT INTO products (parent_asin, title, description, main_category,
                                    average_rating, rating_number, price, store,
                                    features, categories, details, bought_together)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (parent_asin) DO NOTHING
            """, products_data)
            
            logger.info(f"✓ Loaded {len(products_data):,} products")
        
        # Load reviews - ONLY for products we loaded
        logger.info("Loading reviews...")
        reviews_df = pd.read_parquet(processed_path / "All_Beauty_reviews_cleaned.parquet")
        
        # CRITICAL: Filter reviews to only include loaded products
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(loaded_asins)]
        logger.info(f"Filtered to {len(reviews_df):,} reviews for loaded products")
        
        # Pre-compute user statistics (only for filtered reviews)
        user_stats = reviews_df.groupby('user_id').agg({
            'rating': ['count', 'mean'],
            'timestamp': ['min', 'max']
        }).reset_index()
        user_stats.columns = ['user_id', 'review_count', 'avg_rating', 'first_review', 'last_review']
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            # Insert users
            users_data = []
            for _, row in user_stats.iterrows():
                avg_rating = row['avg_rating']
                if pd.isna(avg_rating):
                    avg_rating = None
                else:
                    avg_rating = float(avg_rating)
                
                users_data.append((
                    str(row['user_id']),
                    int(row['review_count']),
                    avg_rating,
                    row['first_review'],
                    row['last_review']
                ))
            
            execute_batch(cur, """
                INSERT INTO users (user_id, review_count, avg_rating, 
                                 first_review_date, last_review_date)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, users_data)
            
            logger.info(f"✓ Loaded {len(users_data):,} users")
            
            # Insert reviews in batches
            batch_size = 10000
            reviews_data = []
            
            for i, (_, row) in enumerate(reviews_df.iterrows()):
                helpful_vote = row.get('helpful_vote', 0)
                if pd.isna(helpful_vote):
                    helpful_vote = 0
                else:
                    helpful_vote = int(helpful_vote)
                
                text_length = row.get('text_length', 0)
                if pd.isna(text_length):
                    text_length = 0
                else:
                    text_length = int(text_length)
                
                reviews_data.append((
                    str(row['user_id']),
                    str(row['parent_asin']),
                    int(row['rating']),
                    str(row['title']) if pd.notna(row.get('title')) else '',
                    str(row['text']) if pd.notna(row.get('text')) else '',
                    helpful_vote,
                    bool(row['verified_purchase']) if pd.notna(row.get('verified_purchase')) else False,
                    row['timestamp'],
                    bool(row['has_image']) if pd.notna(row.get('has_image')) else False,
                    text_length,
                    int(row['year']),
                    int(row['month'])
                ))
                
                if len(reviews_data) >= batch_size:
                    execute_batch(cur, """
                        INSERT INTO reviews (user_id, parent_asin, rating, review_title,
                                           review_text, helpful_vote, verified_purchase,
                                           review_timestamp, has_image, text_length, year, month)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, reviews_data)
                    reviews_data = []
                    
                    if (i + 1) % 50000 == 0:
                        logger.info(f"  ... {i + 1:,} reviews inserted")
            
            # Insert remaining
            if reviews_data:
                execute_batch(cur, """
                    INSERT INTO reviews (user_id, parent_asin, rating, review_title,
                                       review_text, helpful_vote, verified_purchase,
                                       review_timestamp, has_image, text_length, year, month)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, reviews_data)
            
            logger.info(f"✓ Loaded {len(reviews_df):,} reviews")
    
    # QUERY 1: Product Information Retrieval
    def query_product_info(self, parent_asin: str) -> Optional[Dict]:
        """Query 1: Get product title and description by parent ASIN."""
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT parent_asin, title, description, main_category,
                       average_rating, rating_number, price, store
                FROM products
                WHERE parent_asin = %s
            """, (parent_asin,))
            
            result = cur.fetchone()
            return dict(result) if result else None
    
    # QUERY 2: Recent N Reviews by Product
    def query_recent_reviews(self, parent_asin: str, n: int = 10) -> List[Dict]:
        """Query 2: Get recent N reviews for a product."""
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT rating, review_title, review_text, review_timestamp,
                       helpful_vote, verified_purchase, user_id
                FROM reviews
                WHERE parent_asin = %s
                ORDER BY review_timestamp DESC
                LIMIT %s
            """, (parent_asin, n))
            
            return [dict(row) for row in cur.fetchall()]
    
    # QUERY 3: Keyword Search in Titles/Descriptions
    def query_keyword_search(self, keyword: str, limit: int = 20) -> List[Dict]:
        """Query 3: Full-text search in product titles and descriptions."""
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT parent_asin, title, description, main_category, price,
                       ts_rank(to_tsvector('english', title || ' ' || COALESCE(description, '')), 
                               plainto_tsquery('english', %s)) as relevance
                FROM products
                WHERE to_tsvector('english', title || ' ' || COALESCE(description, '')) 
                      @@ plainto_tsquery('english', %s)
                ORDER BY relevance DESC
                LIMIT %s
            """, (keyword, keyword, limit))
            
            return [dict(row) for row in cur.fetchall()]
    
    # QUERY 4: User Review History
    def query_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Query 4: Get user's review history with product details."""
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT r.parent_asin, p.title, r.rating, r.review_timestamp,
                       r.review_title, r.verified_purchase
                FROM reviews r
                JOIN products p ON r.parent_asin = p.parent_asin
                WHERE r.user_id = %s
                ORDER BY r.review_timestamp DESC
                LIMIT %s
            """, (user_id, limit))
            
            return [dict(row) for row in cur.fetchall()]
    
    # QUERY 5: Product Statistics
    def query_product_statistics(self, parent_asin: str) -> Optional[Dict]:
        """Query 5: Get comprehensive product statistics."""
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT 
                    parent_asin,
                    COUNT(*) as total_reviews,
                    AVG(rating) as avg_rating,
                    STDDEV(rating) as rating_std,
                    COUNT(CASE WHEN rating = 1 THEN 1 END) as star_1,
                    COUNT(CASE WHEN rating = 2 THEN 1 END) as star_2,
                    COUNT(CASE WHEN rating = 3 THEN 1 END) as star_3,
                    COUNT(CASE WHEN rating = 4 THEN 1 END) as star_4,
                    COUNT(CASE WHEN rating = 5 THEN 1 END) as star_5,
                    SUM(CASE WHEN verified_purchase THEN 1 ELSE 0 END) as verified_count,
                    AVG(CASE WHEN verified_purchase THEN rating END) as verified_avg_rating
                FROM reviews
                WHERE parent_asin = %s
                GROUP BY parent_asin
            """, (parent_asin,))
            
            result = cur.fetchone()
            if result:
                stats = dict(result)
                total = stats['total_reviews']
                for i in range(1, 6):
                    key = f'star_{i}'
                    stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
                return stats
            return None
    
    def benchmark_queries(self, iterations: int = 10) -> pd.DataFrame:
        """Benchmark all 5 queries."""
        import random
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT parent_asin FROM products ORDER BY RANDOM() LIMIT 5")
            sample_products = [row[0] for row in cur.fetchall()]
            
            cur.execute("SELECT user_id FROM users ORDER BY RANDOM() LIMIT 5")
            sample_users = [row[0] for row in cur.fetchall()]
        
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
            'description': 'Primary key lookup'
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
            'description': 'Index scan + sort'
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
            'description': 'Full-text search (GIN index)'
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
            'description': 'Index scan + join'
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
            'description': 'Aggregation with index'
        })
        
        df = pd.DataFrame(results)
        df.to_csv('docs/postgres_benchmark.csv', index=False)
        logger.info("✓ Benchmark results saved to docs/postgres_benchmark.csv")
        
        return df


def main():
    """Demo of PostgreSQL implementation"""
    
    logger.info("Checking database...")
    create_database_if_not_exists(password="Hafil2004")
    
    pg = PostgresManager(password="Hafil2004")
    
    pg.create_schema()
    
    # Load data - reviews are automatically filtered to match products
    pg.load_data(sample_size=10000)
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    with pg.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parent_asin FROM products LIMIT 1")
        sample_asin = cur.fetchone()[0]
    
    print(f"\nQ1: Product Info for {sample_asin}")
    result = pg.query_product_info(sample_asin)
    print(f"  Title: {result['title'][:60]}...")
    
    print(f"\nQ2: Recent 5 reviews")
    reviews = pg.query_recent_reviews(sample_asin, n=5)
    print(f"  Found {len(reviews)} reviews")
    for r in reviews[:2]:
        print(f"    {r['rating']}★: {r['review_title'][:40]}...")
    
    print(f"\nQ3: Keyword search 'conditioner'")
    products = pg.query_keyword_search("conditioner", limit=3)
    print(f"  Found {len(products)} products")
    for p in products:
        print(f"    {p['title'][:50]}... (relevance: {p['relevance']:.3f})")
    
    print("\n" + "="*60)
    print("BENCHMARKING")
    print("="*60)
    benchmark_df = pg.benchmark_queries(iterations=5)
    print(benchmark_df.to_string(index=False))


if __name__ == "__main__":
    main()
