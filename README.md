# Amazon Reviews 2023: Large-Scale Data Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-orange.svg)](https://amazon-reviews-2023.github.io/)

A comprehensive data analysis platform implementing multi-database architectures, similarity computation, clustering analysis, and recommendation systems for the Amazon Reviews 2023 dataset (All Beauty category).

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task 1: Database System Comparison](#task-1-database-system-comparison)
  - [Task 2: Similarity Computation](#task-2-similarity-computation)
  - [Task 3: Clustering Analysis](#task-3-clustering-analysis)
  - [Task 4: Recommendation System](#task-4-recommendation-system)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete data analysis pipeline for e-commerce product recommendations, comparing multiple database technologies, evaluating similarity algorithms, performing customer/product clustering, and building hybrid recommendation engines.

**Dataset:** Amazon Reviews 2023 - All Beauty Category
- **701K+ reviews**
- **112K+ products**
- **632K+ users**

## 🏗️ Architecture

We implement a **hybrid database architecture** leveraging the strengths of three database systems:

┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
└───────────────────────┬─────────────────────────────────────┘
│
┌───────────────┼───────────────┐
▼               ▼               ▼
┌─────────┐    ┌──────────┐    ┌──────────┐
│ MongoDB │    │PostgreSQL│    │  Neo4j   │
│(Primary)│    │(Analytics│    │ (Graph)  │
│ Catalog │    │Reporting)│    │Recommend │
└─────────┘    └──────────┘    └──────────┘
│               │               │
└───────────────┴───────────────┘
│
┌─────────┴──────────┐
▼                    ▼
┌────────────┐      ┌──────────────┐
│   FAISS    │      │   Scikit-    │
│ (Similarity│      │   Learn      │
│   Search)  │      │ (Clustering) │
└────────────┘      └──────────────┘


## 📁 Project Structure
├── configs/
│   └──config
├── data/
│   ├── raw/
│   │   ├── All_Beauty_meta.jsonl
│   │   ├── All_Beauty_reviews.jsonl
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
│   │   ├── clustering_service.py
│   ├── task4_recommendation/   
│   │   └── recommendation_service.py
│   ├── data_inspector.py
│   └── data_preprocessing,py
├── docs/
│   ├── task1_database_comparison.md
│   ├── task2_similarity_evaluation.md
│   ├── task3_clustering_report.md
│   └── task4_recommendation_evaluation.md
├── notebooks/
│   ├── 01_eda.py
│   └── task 2 visualization.py
├── requirements.txt
└── README.md


## 🗄️ Task 1: Database System Comparison

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
