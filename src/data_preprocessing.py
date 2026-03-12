"""
Data Preprocessing Pipeline for Amazon Reviews 2023
Cleans, transforms, and prepares data for downstream tasks.
"""

import gzip
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Iterator, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses Amazon Reviews 2023 data for Big Data Analytics coursework.
    Handles cleaning, deduplication, and feature engineering.
    """
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'raw_reviews': 0,
            'cleaned_reviews': 0,
            'duplicates_removed': 0,
            'missing_dropped': 0
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text fields"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize unicode
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        return text.strip()
    
    def parse_timestamp(self, ts: int) -> datetime:
        """Convert Unix timestamp (ms) to datetime"""
        return datetime.fromtimestamp(ts / 1000)
    
    def process_reviews(self, category: str = "All_Beauty", 
                       sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process reviews: clean, deduplicate, engineer features.
        
        Args:
            category: Product category
            sample_size: Limit to N records (None for all)
        """
        input_file = self.input_dir / f"{category}_reviews.jsonl.gz"
        output_file = self.output_dir / f"{category}_reviews_cleaned.parquet"
        
        logger.info(f"Processing reviews from {input_file}")
        
        records = []
        seen_ids = set()  # For deduplication
        
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                
                self.stats['raw_reviews'] += 1
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Create unique key for deduplication
                dup_key = (record.get('user_id'), record.get('parent_asin'), 
                          record.get('timestamp'))
                
                if dup_key in seen_ids:
                    self.stats['duplicates_removed'] += 1
                    continue
                
                seen_ids.add(dup_key)
                
                # Clean and transform
                cleaned = {
                    'review_id': f"{record.get('user_id')}_{record.get('parent_asin')}_{record.get('timestamp')}",
                    'user_id': record.get('user_id'),
                    'parent_asin': record.get('parent_asin'),
                    'rating': float(record.get('rating', 0)),
                    'title': self.clean_text(record.get('title')),
                    'text': self.clean_text(record.get('text')),
                    'text_length': len(self.clean_text(record.get('text', ''))),
                    'has_image': len(record.get('images', [])) > 0,
                    'helpful_vote': int(record.get('helpful_vote', 0)),
                    'verified_purchase': bool(record.get('verified_purchase', False)),
                    'timestamp': self.parse_timestamp(record.get('timestamp', 0)),
                    'year': self.parse_timestamp(record.get('timestamp', 0)).year,
                    'month': self.parse_timestamp(record.get('timestamp', 0)).month,
                }
                
                # Quality filters
                if cleaned['rating'] < 1 or cleaned['rating'] > 5:
                    self.stats['missing_dropped'] += 1
                    continue
                
                if not cleaned['user_id'] or not cleaned['parent_asin']:
                    self.stats['missing_dropped'] += 1
                    continue
                
                records.append(cleaned)
                self.stats['cleaned_reviews'] += 1
                
                if (i + 1) % 50000 == 0:
                    logger.info(f"Processed {i + 1:,} records...")
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save as Parquet (efficient for downstream processing)
        df.to_parquet(output_file, compression='snappy')
        logger.info(f"✓ Saved cleaned reviews: {output_file}")
        logger.info(f"  Records: {len(df):,}")
        
        return df
    
    def process_metadata(self, category: str = "All_Beauty") -> pd.DataFrame:
        """Process product metadata"""
        input_file = self.input_dir / f"{category}_meta.jsonl.gz"
        output_file = self.output_dir / f"{category}_metadata_cleaned.parquet"
        
        logger.info(f"Processing metadata from {input_file}")
        
        records = []
        
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                
                # Extract and clean fields
                cleaned = {
                    'parent_asin': record.get('parent_asin'),
                    'title': self.clean_text(record.get('title')),
                    'main_category': record.get('main_category'),
                    'average_rating': float(record.get('average_rating', 0)) if record.get('average_rating') else None,
                    'rating_number': int(record.get('rating_number', 0)) if record.get('rating_number') else 0,
                    'price': self._parse_price(record.get('price')),
                    'store': record.get('store'),
                    'has_image': bool(record.get('images')),
                    'has_video': bool(record.get('videos')),
                    'description': self.clean_text(' '.join(record.get('description', []))),
                    'features': record.get('features', []),
                    'categories': record.get('categories', []),
                    'bought_together': record.get('bought_together', []),
                }
                
                records.append(cleaned)
        
        df = pd.DataFrame(records)
        df.to_parquet(output_file, compression='snappy')
        
        logger.info(f"✓ Saved cleaned metadata: {output_file}")
        logger.info(f"  Records: {len(df):,}")
        
        return df
    
    def _parse_price(self, price) -> Optional[float]:
        """Extract numeric price from various formats"""
        if price is None:
            return None
        
        if isinstance(price, (int, float)):
            return float(price)
        
        if isinstance(price, str):
            # Remove $ and commas
            price_str = re.sub(r'[^\d.]', '', price)
            try:
                return float(price_str) if price_str else None
            except ValueError:
                return None
        
        return None
    
    def create_train_test_split(self, reviews_df: pd.DataFrame, 
                                test_size: float = 0.2) -> tuple:
        """
        Create temporal train/test split (NOT random - respects time order).
        Critical for recommendation systems to avoid data leakage.
        """
        logger.info("Creating temporal train/test split...")
        
        # Sort by timestamp
        df = reviews_df.sort_values('timestamp')
        
        # Split point
        split_idx = int(len(df) * (1 - test_size))
        
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        # Save splits
        train.to_parquet(self.output_dir / "train_reviews.parquet")
        test.to_parquet(self.output_dir / "test_reviews.parquet")
        
        logger.info(f"✓ Train set: {len(train):,} reviews ({train['timestamp'].min()} to {train['timestamp'].max()})")
        logger.info(f"✓ Test set: {len(test):,} reviews ({test['timestamp'].min()} to {test['timestamp'].max()})")
        
        return train, test
    
    def generate_lookup_tables(self, reviews_df: pd.DataFrame, 
                              meta_df: pd.DataFrame):
        """Generate lookup tables for fast access"""
        
        # User lookup
        user_stats = reviews_df.groupby('user_id').agg({
            'rating': ['count', 'mean'],
            'parent_asin': 'nunique',
            'timestamp': ['min', 'max']
        }).reset_index()
        user_stats.columns = ['user_id', 'review_count', 'avg_rating', 
                             'unique_products', 'first_review', 'last_review']
        user_stats.to_parquet(self.output_dir / "user_lookup.parquet")
        
        # Product lookup
        product_stats = reviews_df.groupby('parent_asin').agg({
            'rating': ['count', 'mean'],
            'user_id': 'nunique',
            'helpful_vote': 'sum'
        }).reset_index()
        product_stats.columns = ['parent_asin', 'review_count', 'avg_rating',
                                'unique_users', 'total_helpful_votes']
        product_stats.to_parquet(self.output_dir / "product_lookup.parquet")
        
        logger.info("✓ Generated lookup tables")
    
    def print_stats(self):
        """Print preprocessing statistics"""
        print("\n" + "="*60)
        print("PREPROCESSING STATISTICS")
        print("="*60)
        for key, value in self.stats.items():
            print(f"  {key}: {value:,}")
        print("="*60)


def main():
    """Run full preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Process full dataset (or limit for testing)
    reviews_df = preprocessor.process_reviews(sample_size=None)  # None for all
    meta_df = preprocessor.process_metadata()
    
    # Create splits
    train, test = preprocessor.create_train_test_split(reviews_df)
    
    # Generate lookups
    preprocessor.generate_lookup_tables(reviews_df, meta_df)
    
    # Print stats
    preprocessor.print_stats()
    
    print("\n✓ Preprocessing complete!")
    print("Output files in data/processed/:")
    for f in preprocessor.output_dir.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()