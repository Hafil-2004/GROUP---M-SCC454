"""
Vector database implementations for fast similarity search.
FAISS, Annoy, and brute-force comparison.
"""

import numpy as np
import faiss
from annoy import AnnoyIndex
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaissIndex:
    """
    Facebook AI Similarity Search (FAISS) implementation.
    Best for: Exact and approximate nearest neighbor search at scale.
    """
    
    def __init__(self, dimension: int, index_type: str = 'Flat'):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Maps internal index to original ID
        self.reverse_map = {}
        
    def build_index(self, vectors: np.ndarray, ids: List[str]):
        """Build FAISS index."""
        logger.info(f"Building FAISS {self.index_type} index...")
        
        # Ensure float32 and contiguous
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        
        # Create index
        if self.index_type == 'Flat':
            # Exact L2 search
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'FlatIP':
            # Exact inner product (cosine if normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == 'IVF':
            # Inverted file index (faster, approximate)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.train(vectors)
        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
        
        # Add vectors
        self.index.add(vectors)
        
        # Build ID mapping
        self.id_map = {i: id_ for i, id_ in enumerate(ids)}
        self.reverse_map = {id_: i for i, id_ in enumerate(ids)}
        
        logger.info(f"✓ FAISS index built with {self.index.ntotal} vectors")
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.
        Returns: List of (id, distance) tuples.
        """
        if self.index is None:
            raise ValueError("Index not built")
        
        # Ensure correct format
        query = np.ascontiguousarray(query_vector.reshape(1, -1).astype('float32'))
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Map back to original IDs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch search for efficiency."""
        query_vectors = np.ascontiguousarray(query_vectors.astype('float32'))
        distances, indices = self.index.search(query_vectors, k)
        
        results = []
        for i in range(len(query_vectors)):
            batch_results = []
            for idx, dist in zip(indices[i], distances[i]):
                if idx != -1 and idx in self.id_map:
                    batch_results.append((self.id_map[idx], float(dist)))
            results.append(batch_results)
        
        return results


class AnnoyIndexWrapper:
    """
    Annoy (Approximate Nearest Neighbors Oh Yeah) implementation.
    Best for: Memory-efficient approximate search with fast build times.
    """
    
    def __init__(self, dimension: int, metric: str = 'angular'):
        self.dimension = dimension
        self.metric = metric  # 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'
        self.index = None
        self.id_map = {}
        self.reverse_map = {}
        
    def build_index(self, vectors: np.ndarray, ids: List[str], n_trees: int = 10):
        """Build Annoy index."""
        logger.info(f"Building Annoy index with {n_trees} trees...")
        
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.id_map = {i: id_ for i, id_ in enumerate(ids)}
        self.reverse_map = {id_: i for i, id_ in enumerate(ids)}
        
        # Add items
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)
        
        # Build index
        self.index.build(n_trees)
        
        logger.info(f"✓ Annoy index built with {len(ids)} vectors")
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if self.index is None:
            raise ValueError("Index not built")
        
        indices, distances = self.index.get_nns_by_vector(
            query_vector, k, include_distances=True
        )
        
        return [(self.id_map[idx], float(dist)) for idx, dist in zip(indices, distances)]
    
    def save(self, filename: str):
        """Save index to disk."""
        if self.index:
            self.index.save(filename)
            logger.info(f"✓ Annoy index saved to {filename}")
    
    def load(self, filename: str):
        """Load index from disk."""
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.index.load(filename)
        logger.info(f"✓ Annoy index loaded from {filename}")


class BruteForceIndex:
    """
    Brute-force similarity computation.
    Best for: Small datasets, exact results, baseline comparison.
    """
    
    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
        self.vectors = None
        self.id_map = {}
        
    def build_index(self, vectors: np.ndarray, ids: List[str]):
        """Store vectors for brute-force search."""
        logger.info("Building brute-force index...")
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.vectors = vectors / norms
        else:
            self.vectors = vectors
        
        self.id_map = {i: id_ for i, id_ in enumerate(ids)}
        logger.info(f"✓ Brute-force index built with {len(ids)} vectors")
        
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Compute exact similarities."""
        if self.vectors is None:
            raise ValueError("Index not built")
        
        # Normalize query
        if self.metric == 'cosine':
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
        
        # Compute similarities (dot product for normalized vectors = cosine)
        similarities = np.dot(self.vectors, query_vector)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return [(self.id_map[idx], float(similarities[idx])) for idx in top_indices]