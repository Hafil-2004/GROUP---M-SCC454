# Amazon Reviews 2023: Large-Scale Data Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-orange.svg)](https://amazon-reviews-2023.github.io/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/Hafil-2004/amazon-reviews-2023-beauty-processed)

A comprehensive data analysis platform implementing multi-database architectures, similarity computation, clustering analysis, and recommendation systems for the Amazon Reviews 2023 dataset (All Beauty category).

## 👥 Team
- HAFIL ABDUL KADHAR
- Teammate 1
- Teammate 2
- Teammate 3
  
## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tasks](#-tasks)
  - [Task 1: Database System Comparison](#-task-1-database-system-comparison)
  - [Task 2: Similarity Computation](#-task-2-similarity-computation)
  - [Task 3: Clustering Analysis](#-task-3-clustering-analysis)
  - [Task 4: Recommendation System](#-task-4-recommendation-system)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results Summary](#-results-summary)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a complete data analysis pipeline for e-commerce product recommendations, comparing multiple database technologies, evaluating similarity algorithms, performing customer/product clustering, and building hybrid recommendation engines.

**Dataset:** Amazon Reviews 2023 - All Beauty Category
- **701K+ reviews**
- **112K+ products**
- **632K+ users**

## 🧱 Architecture

We implement a **hybrid database architecture** leveraging the strengths of three database systems:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      ┌─────────┐  ┌──────────┐  ┌──────────┐
      │ MongoDB │  │PostgreSQL│  │  Neo4j   │
      │(Primary)│  │(Analytics│  │ (Graph)  │
      │ Catalog │  │Reporting)│  │Recommend │
      └─────────┘  └──────────┘  └──────────┘
           │            │            │
           └────────────┴────────────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
    ┌────────────┐           ┌──────────────┐
    │   FAISS    │           │   Scikit-    │
    │ (Similarity│           │   Learn      │
    │   Search)  │           │ (Clustering) │
    └────────────┘           └──────────────┘
```

## 📁 Project Structure

```
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── All_Beauty_meta.jsonl
│   │   └── All_Beauty_reviews.jsonl
│   ├── processed/
│   │   ├── All_Beauty_metadata_cleaned.parquet
│   │   ├── All_Beauty_reviews_cleaned.parquet
│   │   ├── product_clusters_hierarchical.parquet
│   │   ├── product_clusters_kmeans.parquet
│   │   ├── product_lookup.parquet
│   │   ├── test_reviews.parquet
│   │   ├── train_reviews.parquet
│   │   ├── user_clusters_dbscan.parquet
│   │   ├── user_clusters_kmeans.parquet
│   │   └── user_lookup.parquet
│   └── cache/
├── src/
│   ├── task1_databases/
│   │   ├── mongo_manager.py
│   │   ├── neo4j_manager.py
│   │   └── postgres_manager.py
│   ├── task2_similarity/
│   │   ├── feature_extractors.py
│   │   ├── similarity_service.py
│   │   └── vector_stores.py
│   ├── task3_clustering/
│   │   └── clustering_service.py
│   ├── task4_recommendation/
│   │   └── recommendation_service.py
│   ├── data_inspector.py
│   └── data_preprocessing.py
├── docs/
│   ├── task1_database_comparison.md
│   ├── task2_similarity_evaluation.md
│   ├── task3_clustering_report.md
│   └── task4_recommendation_evaluation.md
├── notebooks/
│   ├── 01_eda.py
│   └── task2_visualization.py
├── requirements.txt
└── README.md
```
## 🎯 Tasks
## 💾 Task 1: Database System Comparison

Implemented three database systems with comprehensive performance benchmarking:

| Database | Type | Best For | Avg Query Time |
|----------|------|----------|----------------|
| **MongoDB** | Document Store | Fast reads, flexible schema | **0.41-2.93 ms** |
| PostgreSQL | Relational | Complex analytics, ACID | 56-107 ms |
| Neo4j | Graph | Relationship queries | 3.4-55.1 ms |

**Key Finding:** MongoDB is **137x faster** than PostgreSQL for primary key lookups, making it ideal for product catalog serving.

**Implementation Details:**
- PostgreSQL: Partitioned tables by year, GIN indexes for full-text search
- MongoDB: Native JSON documents, compound indexes for common queries
- Neo4j: Graph model for user-product relationships and similarity queries

📄 [Full Report](docs/task1_database_comparison.md)

## 🔍 Task 2: Similarity Computation

Evaluated four product similarity approaches and three user similarity methods:

### Product Similarity Results

| Method | Best Score | Build Time | Status |
|--------|-----------|------------|--------|
| **Hybrid (TF-IDF + BERT)** | **0.87-0.93** | ~47s | ✅ Recommended |
| TF-IDF + FAISS | 0.67-0.83 | ~2s | ✅ Fast fallback |
| BERT + FAISS | 0.584 | ~45s | ⚠️ Semantic only |
| Metadata + Annoy | ~0.04 | <1s | ❌ Poor performance |

**Recommendation:** Deploy hybrid approach with TF-IDF fallback for real-time queries.

📄 [Full Report](docs/task2_similarity_evaluation.md)

## 📊 Task 3: Clustering Analysis

Comprehensive clustering of 632K users and 2K products using multiple algorithms:

### Customer Segments Discovered

| Segment | Size | Characteristics | Business Use |
|---------|------|-----------------|--------------|
| **Casual Reviewers** | 580K (91.8%) | Avg rating 3.95, moderate engagement | Mass marketing, convenience focus |
| **Enthusiastic Reviewers** | 52K (8.2%) | Avg rating 4.07, highly engaged | Loyalty programs, early access |

### Product Clusters

- **BERT-based:** 3 price tiers ($17.92, $20.79, $31.07) - weak separation
- **Metadata-based:** 5 quality/engagement clusters - better separation (silhouette: 0.381)

**Key Metrics:**
- User K-Means: Silhouette **0.762** (excellent)
- Product Hierarchical: Silhouette **0.381** (moderate)

📄 [Full Report](docs/task3_clustering_report.md)

## 🎯 Task 4: Recommendation System

Multi-approach recommendation engine with cold-start handling:

### Approaches Implemented

| Method | Type | Strengths |
|--------|------|-----------|
| User-Based CF | Collaborative | Community preferences |
| Item-Based CF | Collaborative | Stable, good coverage |
| Content-Based | TF-IDF | No cold-start |
| **Hybrid (α=0.6)** | **Combined** | **Best overall** |
| Popularity | Fallback | Cold-start handling |

**Cold-Start Strategy:** New users receive popularity-based recommendations (rating count × avg rating) until sufficient interaction history is collected.

**Performance:**
- Matrix sparsity: 99.97%
- Query latency: <300ms (hybrid)
- Coverage: 5,874 products, 5,000 users

📄 [Full Report](docs/task4_recommendation_evaluation.md)

## 🚀 Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 14+
- MongoDB 5.0+
- Neo4j 5.x
- 16GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/amazon-reviews-analysis.git
cd amazon-reviews-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Setup databases
python src/task1_database/setup_all.py
```

### Requirements

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
pymongo>=4.3.0
psycopg2-binary>=2.9.0
neo4j>=5.0.0
umap-learn>=0.5.0
matplotlib>=3.5.0
seaborn>=0.12.0
tqdm>=4.64.0
```

## 💻 Usage

### Task 1: Database Queries

```python
from src.task1_databases import MongoManager, PostgresManager, Neo4jManager

# MongoDB - Fast product lookup
mongo = MongoManager()
product = mongo.get_product("B08P4GRYY8")

# PostgreSQL - Complex analytics
pg = PostgresManager()
stats = pg.get_product_stats("B08P4GRYY8")

# Neo4j - Relationship queries
neo4j = Neo4jManager()
similar = neo4j.find_similar_users("user_123")
```

### Task 2: Similarity Search

```python
from src.task2_similarity import SimilarityService

# Initialize hybrid model
sim = SimilarityService()
sim.build_indices(products_df)

# Search similar products
results = sim.search("leather conditioner", top_k=5)
# Returns: [(product_id, similarity_score, product_name), ...]
```

### Task 3: Clustering

```python
from src.task3_clustering import ClusteringService

# User clustering
clustering = ClusteringService()
user_clusters = clustering.cluster_users_kmeans(user_features, n_clusters=2)

# Product clustering
product_clusters = clustering.cluster_products_hierarchical(product_features, n_clusters=5)
```

### Task 4: Recommendations

```python
from src.task4_recommendation import RecommendationService

# Initialize recommender
rec = RecommendationService()
rec.fit(interaction_matrix, product_features)

# Get recommendations
recommendations = rec.recommend(user_id="user_123", n=10)
```

## 📈 Results Summary

| Task | Key Achievement | Performance |
|------|----------------|-------------|
| Database | Hybrid architecture design | MongoDB 137x faster than PostgreSQL |
| Similarity | Hybrid TF-IDF + BERT | 0.93 similarity score |
| Clustering | 632K users segmented | Silhouette 0.762 (excellent) |
| Recommendation | 4-method hybrid | <300ms latency |

## 🧰 Technologies Used

- **Databases:** PostgreSQL, MongoDB, Neo4j
- **ML/AI:** scikit-learn, sentence-transformers, FAISS, UMAP
- **Data:** pandas, NumPy, PyArrow
- **Visualization:** matplotlib, seaborn
- **Similarity:** FAISS, Annoy, cosine similarity

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) by McAuley et al., UC San Diego
- **Course:** SCC 454 - Large Scale Platforms for AI and Data Analysis, Lancaster University
- **Tools:** scikit-learn, FAISS, sentence-transformers communities

> **Note:** This is an academic project for educational purposes. The dataset is used under the original license terms.
