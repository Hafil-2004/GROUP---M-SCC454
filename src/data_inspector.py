"""
Data Inspector for Amazon Reviews 2023
Analyzes schema, statistics, and quality of downloaded data.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import statistics

import pandas as pd
import numpy as np
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class FieldSchema:
    """Schema information for a single field"""
    name: str
    dtype: str
    nullable: bool
    sample_values: List[Any]
    unique_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DatasetSchema:
    """Complete schema for a dataset"""
    name: str
    total_records: int
    fields: List[FieldSchema]
    file_size_mb: float
    memory_estimate_mb: float
    
    def to_dict(self):
        return {
            'name': self.name,
            'total_records': self.total_records,
            'file_size_mb': self.file_size_mb,
            'memory_estimate_mb': self.memory_estimate_mb,
            'fields': [f.to_dict() for f in self.fields]
        }


class DataInspector:
    """
    Inspects Amazon Reviews 2023 data files to understand schema and quality.
    Uses streaming to handle large files without loading everything into memory.
    """
    
    def __init__(self, file_path: str, sample_size: int = 10000):
        self.file_path = Path(file_path)
        self.sample_size = sample_size
        self.records_sample = []
        self.field_stats = defaultdict(lambda: {
            'types': Counter(),
            'null_count': 0,
            'values': [],
            'numeric_values': [],
            'lengths': []
        })
        
    def analyze(self) -> DatasetSchema:
        """Main analysis method"""
        logger.info(f"Analyzing {self.file_path}...")
        
        file_size = self.file_path.stat().st_size
        file_size_mb = file_size / 1024 / 1024
        
        # Stream through file
        record_count = 0
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                record_count += 1
                
                # Collect sample for detailed analysis
                if len(self.records_sample) < self.sample_size:
                    self.records_sample.append(record)
                    self._analyze_record(record)
                
                # Progress logging
                if record_count % 100000 == 0:
                    logger.info(f"Processed {record_count:,} records...")
        
        # Build schema from collected stats
        fields = self._build_field_schemas(record_count)
        
        # Estimate memory usage (rough approximation)
        avg_record_size = file_size / record_count
        memory_estimate_mb = (avg_record_size * record_count) / 1024 / 1024
        
        return DatasetSchema(
            name=self.file_path.stem.replace('.jsonl', ''),
            total_records=record_count,
            fields=fields,
            file_size_mb=file_size_mb,
            memory_estimate_mb=memory_estimate_mb
        )
    
    def _analyze_record(self, record: Dict):
        """Analyze a single record and update statistics"""
        for field, value in record.items():
            stats = self.field_stats[field]
            
            # Track data types
            dtype = type(value).__name__
            if value is None:
                dtype = 'null'
                stats['null_count'] += 1
            stats['types'][dtype] += 1
            
            # Collect sample values (non-null)
            if value is not None and len(stats['values']) < 100:
                stats['values'].append(value)
            
            # Numeric statistics
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats['numeric_values'].append(value)
            
            # String lengths
            if isinstance(value, str):
                stats['lengths'].append(len(value))
            elif isinstance(value, (list, dict)):
                stats['lengths'].append(len(value))
    
    def _build_field_schemas(self, total_records: int) -> List[FieldSchema]:
        """Build field schemas from collected statistics"""
        fields = []
        
        for field_name, stats in self.field_stats.items():
            # Determine primary type (most common non-null)
            primary_type = stats['types'].most_common(1)[0][0]
            if primary_type == 'null' and len(stats['types']) > 1:
                primary_type = stats['types'].most_common(2)[1][0]
            
            # Check if nullable
            nullable = stats['null_count'] > 0
            
            # Calculate unique count if manageable
            unique_count = None
            if len(stats['values']) <= 100:
                unique_count = len(set(str(v) for v in stats['values']))
            
            # Min/max for numeric fields
            min_val, max_val = None, None
            if stats['numeric_values']:
                min_val = min(stats['numeric_values'])
                max_val = max(stats['numeric_values'])
            
            # Format sample values for display
            samples = []
            for v in stats['values'][:5]:
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:100] + '...'
                samples.append(v_str)
            
            fields.append(FieldSchema(
                name=field_name,
                dtype=primary_type,
                nullable=nullable,
                sample_values=samples,
                unique_count=unique_count,
                min_value=min_val,
                max_value=max_val
            ))
        
        return fields
    
    def generate_data_dictionary(self) -> pd.DataFrame:
        """Generate a pandas DataFrame with field descriptions"""
        schema = self.analyze()
        
        data = []
        for field in schema.fields:
            data.append({
                'Field Name': field.name,
                'Data Type': field.dtype,
                'Nullable': field.nullable,
                'Unique Values (sample)': field.unique_count,
                'Min': field.min_value,
                'Max': field.max_value,
                'Sample Values': '; '.join(str(v) for v in field.sample_values[:3])
            })
        
        return pd.DataFrame(data)
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check for common data quality issues"""
        issues = []
        warnings = []
        
        schema = self.analyze()
        
        # Check for high null rates
        for field in schema.fields:
            null_rate = self.field_stats[field.name]['null_count'] / schema.total_records
            if null_rate > 0.5:
                issues.append(f"Field '{field.name}' has {null_rate:.1%} null values")
            elif null_rate > 0.1:
                warnings.append(f"Field '{field.name}' has {null_rate:.1%} null values")
        
        # Check for empty strings
        for field in schema.fields:
            if field.dtype == 'str':
                empty_count = sum(1 for v in self.field_stats[field.name]['values'] if v == '')
                if empty_count > 0:
                    warnings.append(f"Field '{field.name}' has {empty_count} empty strings in sample")
        
        # Check for duplicates (on sample)
        if len(self.records_sample) > 1:
            # Check for duplicate records (based on a subset of fields)
            id_fields = ['user_id', 'parent_asin', 'timestamp']
            available_id_fields = [f for f in id_fields if f in [fld.name for fld in schema.fields]]
            
            if available_id_fields:
                seen = set()
                duplicates = 0
                for record in self.records_sample:
                    key = tuple(record.get(f) for f in available_id_fields)
                    if key in seen:
                        duplicates += 1
                    seen.add(key)
                
                if duplicates > 0:
                    dup_rate = duplicates / len(self.records_sample)
                    warnings.append(f"Approximately {dup_rate:.1%} duplicate records detected (based on sample)")
        
        return {
            'total_records': schema.total_records,
            'issues': issues,
            'warnings': warnings,
            'is_valid': len(issues) == 0
        }


def print_schema_report(schema: DatasetSchema):
    """Pretty print schema report"""
    print("\n" + "="*80)
    print(f"DATASET SCHEMA REPORT: {schema.name}")
    print("="*80)
    print(f"Total Records: {schema.total_records:,}")
    print(f"File Size: {schema.file_size_mb:.2f} MB")
    print(f"Estimated Memory: {schema.memory_estimate_mb:.2f} MB")
    print(f"Fields: {len(schema.fields)}")
    print("-"*80)
    
    for field in schema.fields:
        print(f"\n📌 {field.name}")
        print(f"   Type: {field.dtype} | Nullable: {field.nullable}")
        if field.unique_count:
            print(f"   Unique Values (in sample): {field.unique_count}")
        if field.min_value is not None:
            print(f"   Range: {field.min_value} to {field.max_value}")
        print(f"   Samples: {', '.join(str(v) for v in field.sample_values[:3])}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect Amazon Reviews 2023 data')
    parser.add_argument('file', help='Path to .jsonl.gz file')
    parser.add_argument('--output', '-o', help='Output JSON file for schema')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of records to sample for analysis')
    
    args = parser.parse_args()
    
    inspector = DataInspector(args.file, sample_size=args.sample_size)
    
    # Generate schema
    schema = inspector.analyze()
    print_schema_report(schema)
    
    # Data quality check
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)
    quality = inspector.check_data_quality()
    print(f"Valid: {quality['is_valid']}")
    if quality['issues']:
        print("\n❌ Issues:")
        for issue in quality['issues']:
            print(f"  - {issue}")
    if quality['warnings']:
        print("\n⚠️  Warnings:")
        for warning in quality['warnings']:
            print(f"  - {warning}")
    if not quality['issues'] and not quality['warnings']:
        print("✓ No major issues detected")
    
    # Save schema if requested
    if args.output:
        import json as jsonlib
        with open(args.output, 'w') as f:
            jsonlib.dump(schema.to_dict(), f, indent=2, default=str)
        print(f"\n💾 Schema saved to {args.output}")
    
    # Generate data dictionary
    print("\n" + "="*80)
    print("DATA DICTIONARY")
    print("="*80)
    df = inspector.generate_data_dictionary()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()