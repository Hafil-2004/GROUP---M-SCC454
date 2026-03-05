"""
Task 1: Neo4j Graph Database Implementation
Graph store for relationships: users, products, reviews, co-purchases.
"""

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_python_types(obj):
    """Convert NumPy types to Python native types"""
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


class Neo4jManager:
    """
    Neo4j implementation for Amazon Reviews 2023.
    Optimized for relationship queries and graph analytics.
    """
    
    def __init__(self, uri="neo4j://127.0.0.1:7687", user="neo4j", password="Hafil2004"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("✓ Database cleared")
    
    def create_schema(self):
        """Create indexes and constraints"""
        with self.driver.session() as session:
            # Constraints for unique IDs
            session.run("CREATE CONSTRAINT product_asin IF NOT EXISTS FOR (p:Product) REQUIRE p.asin IS UNIQUE")
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
            
            # Indexes for performance
            session.run("CREATE INDEX product_category IF NOT EXISTS FOR (p:Product) ON (p.category)")
            session.run("CREATE INDEX review_timestamp IF NOT EXISTS FOR (r:Review) ON (r.timestamp)")
            
            logger.info("✓ Schema created with constraints and indexes")
    
    def load_data(self, processed_dir: str = "data/processed",
                  sample_size: Optional[int] = None):
        """Load data into Neo4j as a graph"""
        processed_path = Path(processed_dir)
        
        self.clear_database()
        self.create_schema()
        
        # Load products
        logger.info("Loading products into Neo4j...")
        products_df = pd.read_parquet(processed_path / "All_Beauty_metadata_cleaned.parquet")
        
        if sample_size:
            products_df = products_df.head(sample_size)
        
        # Create product nodes
        with self.driver.session() as session:
            for _, row in products_df.iterrows():
                session.run("""
                    CREATE (p:Product {
                        asin: $asin,
                        title: $title,
                        description: $description,
                        category: $category,
                        average_rating: $avg_rating,
                        price: $price,
                        store: $store
                    })
                """, {
                    "asin": str(row['parent_asin']),
                    "title": str(row['title']) if pd.notna(row['title']) else '',
                    "description": str(row['description']) if pd.notna(row.get('description')) else '',
                    "category": str(row['main_category']) if pd.notna(row.get('main_category')) else '',
                    "avg_rating": float(row['average_rating']) if pd.notna(row.get('average_rating')) else None,
                    "price": float(row['price']) if pd.notna(row.get('price')) else None,
                    "store": str(row['store']) if pd.notna(row.get('store')) else ''
                })
            
            logger.info(f"✓ Created {len(products_df):,} product nodes")
        
        loaded_asins = set(products_df['parent_asin'].tolist())
        
        # Load reviews and create user nodes + relationships
        logger.info("Loading reviews into Neo4j...")
        reviews_df = pd.read_parquet(processed_path / "All_Beauty_reviews_cleaned.parquet")
        reviews_df = reviews_df[reviews_df['parent_asin'].isin(loaded_asins)]
        
        # Create users and reviews in batches
        batch_size = 1000
        total = len(reviews_df)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = reviews_df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    session.run("""
                        MERGE (u:User {user_id: $user_id})
                        WITH u
                        MATCH (p:Product {asin: $asin})
                        CREATE (u)-[:REVIEWED {
                            rating: $rating,
                            title: $title,
                            text: $text,
                            timestamp: $timestamp,
                            helpful_vote: $helpful_vote,
                            verified: $verified
                        }]->(p)
                    """, {
                        "user_id": str(row['user_id']),
                        "asin": str(row['parent_asin']),
                        "rating": int(row['rating']),
                        "title": str(row['title']) if pd.notna(row.get('title')) else '',
                        "text": str(row['text']) if pd.notna(row.get('text')) else '',
                        "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                        "helpful_vote": int(row['helpful_vote']) if pd.notna(row.get('helpful_vote')) else 0,
                        "verified": bool(row['verified_purchase']) if pd.notna(row.get('verified_purchase')) else False
                    })
                
                if (i + batch_size) % 10000 == 0 or (i + batch_size) >= total:
                    logger.info(f"  ... {min(i + len(batch), total):,} reviews processed")
            
            logger.info(f"✓ Created {total:,} review relationships")
        
        # Create co-purchase relationships (from bought_together)
        logger.info("Creating co-purchase relationships...")
        with self.driver.session() as session:
            for _, row in products_df.iterrows():
                bought_together = to_python_types(row.get('bought_together', []))
                if bought_together:
                    for other_asin in bought_together:
                        if other_asin in loaded_asins:
                            session.run("""
                                MATCH (p1:Product {asin: $asin1})
                                MATCH (p2:Product {asin: $asin2})
                                WHERE p1 <> p2
                                MERGE (p1)-[r:BOUGHT_TOGETHER]->(p2)
                                ON CREATE SET r.frequency = 1
                                ON MATCH SET r.frequency = r.frequency + 1
                            """, {
                                "asin1": str(row['parent_asin']),
                                "asin2": str(other_asin)
                            })
            
            logger.info("✓ Created co-purchase relationships")
    
    # QUERY 1: Product Information
    def query_product_info(self, parent_asin: str) -> Optional[Dict]:
        """Get product by ASIN"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Product {asin: $asin})
                RETURN p {
                    .asin, .title, .description, .category,
                    .average_rating, .price, .store
                } as product
            """, asin=parent_asin)
            
            record = result.single()
            return dict(record["product"]) if record else None
    
    # QUERY 2: Recent N Reviews by Product
    def query_recent_reviews(self, parent_asin: str, n: int = 10) -> List[Dict]:
        """Get recent reviews for a product"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[r:REVIEWED]->(p:Product {asin: $asin})
                RETURN u.user_id as user_id, r.rating as rating,
                       r.title as review_title, r.text as review_text,
                       r.timestamp as timestamp, r.verified as verified_purchase
                ORDER BY r.timestamp DESC
                LIMIT $limit
            """, asin=parent_asin, limit=n)
            
            return [dict(record) for record in result]
    
    # QUERY 3: Keyword Search
    def query_keyword_search(self, keyword: str, limit: int = 20) -> List[Dict]:
        """Search products by keyword in title/description"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Product)
                WHERE p.title CONTAINS $keyword OR p.description CONTAINS $keyword
                RETURN p.asin as parent_asin, p.title as title,
                       p.description as description, p.category as main_category,
                       p.price as price
                LIMIT $limit
            """, keyword=keyword, limit=limit)
            
            return [dict(record) for record in result]
    
    # QUERY 4: User Review History
    def query_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's review history"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {user_id: $user_id})-[r:REVIEWED]->(p:Product)
                RETURN p.asin as parent_asin, p.title as title,
                       r.rating as rating, r.timestamp as review_timestamp,
                       r.title as review_title, r.verified as verified_purchase
                ORDER BY r.timestamp DESC
                LIMIT $limit
            """, user_id=user_id, limit=limit)
            
            return [dict(record) for record in result]
    
    # QUERY 5: Product Statistics
    def query_product_statistics(self, parent_asin: str) -> Optional[Dict]:
        """Get product statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[r:REVIEWED]->(p:Product {asin: $asin})
                RETURN p.asin as parent_asin,
                       count(r) as total_reviews,
                       avg(r.rating) as avg_rating,
                       stdev(r.rating) as rating_std,
                       sum(CASE WHEN r.rating = 1 THEN 1 ELSE 0 END) as star_1,
                       sum(CASE WHEN r.rating = 2 THEN 1 ELSE 0 END) as star_2,
                       sum(CASE WHEN r.rating = 3 THEN 1 ELSE 0 END) as star_3,
                       sum(CASE WHEN r.rating = 4 THEN 1 ELSE 0 END) as star_4,
                       sum(CASE WHEN r.rating = 5 THEN 1 ELSE 0 END) as star_5,
                       sum(CASE WHEN r.verified THEN 1 ELSE 0 END) as verified_count,
                       avg(CASE WHEN r.verified THEN r.rating END) as verified_avg_rating
            """, asin=parent_asin)
            
            record = result.single()
            if record and record["total_reviews"] > 0:
                stats = dict(record)
                total = stats["total_reviews"]
                for i in range(1, 6):
                    key = f'star_{i}'
                    stats[f'{key}_pct'] = (stats[key] / total * 100) if total > 0 else 0
                return stats
            return None
    
    # BONUS: Graph-specific queries for Task 2 (Similarity)
    def find_similar_users(self, user_id: str, n: int = 5) -> List[Dict]:
        """Find users with similar taste (Jaccard similarity)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u1:User {user_id: $user_id})-[r1:REVIEWED]->(p:Product)<-[r2:REVIEWED]-(u2:User)
                WHERE u1 <> u2
                WITH u2, count(p) as common_products, 
                     avg(abs(r1.rating - r2.rating)) as rating_diff
                RETURN u2.user_id as user_id, common_products,
                       1.0 / (1.0 + rating_diff) as similarity
                ORDER BY common_products DESC, similarity DESC
                LIMIT $limit
            """, user_id=user_id, limit=n)
            
            return [dict(record) for record in result]
    
    def find_similar_products(self, parent_asin: str, n: int = 5) -> List[Dict]:
        """Find similar products by co-purchase and shared reviewers"""
        with self.driver.session() as session:
            result = session.run("""
                // Co-purchase first
                MATCH (p:Product {asin: $asin})-[:BOUGHT_TOGETHER]-(similar:Product)
                RETURN similar.asin as asin, similar.title as title, 
                       'co_purchase' as reason, 1.0 as score
                LIMIT $limit
                
                UNION
                
                // Shared reviewers
                MATCH (p:Product {asin: $asin})<-[:REVIEWED]-(u:User)-[:REVIEWED]->(similar:Product)
                WHERE p <> similar
                WITH similar, count(u) as shared_reviewers, avg(u.rating) as avg_rating
                RETURN similar.asin as asin, similar.title as title,
                       'shared_reviewers' as reason, 
                       shared_reviewers * 0.1 as score
                ORDER BY score DESC
                LIMIT $limit
            """, asin=parent_asin, limit=n)
            
            return [dict(record) for record in result]
    
    def benchmark_queries(self, iterations: int = 10) -> pd.DataFrame:
        """Benchmark all 5 queries"""
        import random
        
        # Get samples
        with self.driver.session() as session:
            products = [r["p.asin"] for r in session.run("MATCH (p:Product) RETURN p.asin LIMIT 5")]
            users = [r["u.user_id"] for r in session.run("MATCH (u:User) RETURN u.user_id LIMIT 5")]
        
        results = []
        
        # Q1
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_product_info(random.choice(products))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q1: Product Info',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Node lookup by index'
        })
        
        # Q2
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_recent_reviews(random.choice(products), n=10)
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q2: Recent Reviews',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Relationship traversal + sort'
        })
        
        # Q3
        times = []
        keywords = ['conditioner', 'shampoo', 'cream']
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_keyword_search(random.choice(keywords))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q3: Keyword Search',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'String contains scan'
        })
        
        # Q4
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_user_history(random.choice(users))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q4: User History',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Relationship traversal'
        })
        
        # Q5
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.query_product_statistics(random.choice(products))
            times.append((time.perf_counter() - start) * 1000)
        results.append({
            'query': 'Q5: Product Stats',
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'description': 'Aggregation with pattern'
        })
        
        df = pd.DataFrame(results)
        df.to_csv('docs/neo4j_benchmark.csv', index=False)
        logger.info("✓ Benchmark results saved to docs/neo4j_benchmark.csv")
        
        return df


def main():
    """Demo of Neo4j implementation"""
    
    neo4j = Neo4jManager(password="Hafil2004")
    
    # Load data
    neo4j.load_data(sample_size=1000)  # Smaller for Neo4j (slower inserts)
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING NEO4J QUERIES")
    print("="*60)
    
    with neo4j.driver.session() as session:
        sample_product = session.run("MATCH (p:Product) RETURN p.asin LIMIT 1").single()[0]
        sample_user = session.run("MATCH (u:User) RETURN u.user_id LIMIT 1").single()[0]
    
    print(f"\nQ1: Product Info for {sample_product}")
    result = neo4j.query_product_info(sample_product)
    print(f"  Title: {result['title'][:60]}...")
    
    print(f"\nQ2: Recent 5 reviews")
    reviews = neo4j.query_recent_reviews(sample_product, n=5)
    print(f"  Found {len(reviews)} reviews")
    for r in reviews[:2]:
        print(f"    {r['rating']}★: {r['review_title'][:40]}...")
    
    print(f"\nQ3: Keyword search 'conditioner'")
    products = neo4j.query_keyword_search("conditioner", limit=3)
    print(f"  Found {len(products)} products")
    for p in products:
        print(f"    {p['title'][:50]}...")
    
    print(f"\nBONUS: Similar users to {sample_user}")
    similar = neo4j.find_similar_users(sample_user, n=3)
    print(f"  Found {len(similar)} similar users")
    for u in similar:
        print(f"    {u['user_id'][:20]}... (similarity: {u['similarity']:.3f})")
    
    print("\n" + "="*60)
    print("BENCHMARKING NEO4J")
    print("="*60)
    benchmark_df = neo4j.benchmark_queries(iterations=5)
    print(benchmark_df.to_string(index=False))
    
    neo4j.close()


if __name__ == "__main__":
    main()