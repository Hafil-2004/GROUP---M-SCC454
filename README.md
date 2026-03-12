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

