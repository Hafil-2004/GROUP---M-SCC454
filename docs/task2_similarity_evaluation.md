# Task 2: Similarity Computation Evaluation

## Executive Summary

This report evaluates four product similarity approaches and three user similarity approaches implemented for the Amazon beauty products dataset (5,000 products, 39,495 users). The hybrid approach combining TF-IDF and BERT embeddings demonstrated superior performance, achieving similarity scores of 0.87-0.93 compared to individual methods.

## Product Similarity Approaches

### 1. TF-IDF + FAISS (Baseline)
**Strengths:** Excellent at keyword matching, fast index building (~2 seconds), interpretable results. For the leather conditioner query, it correctly identified similar conditioners (Citre Shine, Ogx, Pantene) with scores of 0.67-0.83.

**Weaknesses:** Misses semantic relationships - cannot distinguish between "leather conditioner" (furniture care) and "hair conditioner" without context. Limited to exact word matches.

### 2. BERT Embeddings + FAISS (Semantic)
**Strengths:** Captures semantic meaning beyond keywords. Identified "PRO Leather Lotion" as similar to leather conditioner (score: 0.584), showing understanding of product purpose. Handles synonyms and paraphrases effectively.

**Weaknesses:** Higher computational cost (45 seconds for 5,000 products). Occasionally returns off-target results (e.g., suggesting wig grips for eye pencils), likely due to generic product descriptions in the beauty domain.

### 3. Metadata Features + Annoy (Structured)
**Strengths:** Fastest build time, memory efficient with 10 trees.

**Weaknesses:** **Poorest performance** (scores ~0.000-0.04). The simple hash-based category encoding and sparse features (price, rating, has_description) fail to capture meaningful product relationships. Results were essentially random, indicating metadata alone is insufficient for similarity in this domain.

### 4. Hybrid Approach (Recommended)
**Strengths:** **Best overall performance** (scores 0.87-0.93). Combines TF-IDF's keyword precision with BERT's semantic understanding. For leather conditioner, top match achieved 0.930 similarity. Truncated embeddings (128-dim each) maintain efficiency while maximizing accuracy.

**Weaknesses:** Requires building multiple indices, doubling storage requirements.

## User Similarity Approaches

All three methods (Rating Patterns, Category Preferences, Text Embeddings) achieved perfect scores (1.000) for top matches, indicating highly clustered user behaviors in the sampled dataset. Text embeddings showed more granularity (0.748-0.858 range), suggesting nuanced review content differences.

## Performance Analysis

| Method | Build Time | Query Time | Memory |
|--------|-----------|-----------|--------|
| TF-IDF | ~2s | <1ms | Low |
| BERT | ~45s | <1ms | Medium |
| Metadata | <1s | <1ms | Lowest |
| Hybrid | ~47s | <1ms | High |

## Recommendations

**For Production:** Deploy the **Hybrid approach** for product similarity, with TF-IDF as fallback for real-time queries. BERT embeddings should be pre-computed and cached. Metadata features require engineering improvement (e.g., proper category embeddings instead of hashing) before production use.

**Future Work:** Implement approximate nearest neighbor (ANN) methods like HNSW for million-scale deployment, and explore multi-modal features (product images) for enhanced similarity.

## Conclusion

The hybrid TF-IDF + BERT approach delivers optimal similarity matching for beauty products, combining computational efficiency with semantic accuracy. Pure metadata features are inadequate without significant feature engineering investment.