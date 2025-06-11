# rag_system/retrieval.py
import logging
import numpy as np
from typing import List
from nltk.tokenize import word_tokenize

from .indexing import VectorStore
from .data_structures import RetrievalResult
from .config import config

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines dense and sparse retrieval methods to find relevant documents."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.stop_words = set(word_tokenize("english"))

    def retrieve(self, query: str) -> List[RetrievalResult]:
        if not self.vector_store.index or not self.vector_store.chunks:
            logger.error("Vector store not initialized. Cannot retrieve.")
            return []

        dense_results = self._dense_retrieval(query, config.TOP_K_RETRIEVAL)
        if not config.USE_HYBRID_SEARCH:
            return dense_results
        
        sparse_results = self._sparse_retrieval(query, config.TOP_K_RETRIEVAL)
        combined_results = self._combine_and_rerank(dense_results, sparse_results)
        
        filtered = [r for r in combined_results if r.combined_score >= config.MIN_RELEVANCE_SCORE]
        logger.info(f"Retrieved {len(filtered)} relevant chunks for query.")
        return filtered[:config.TOP_K_RERANK]

    def _dense_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Performs dense search using FAISS."""
        query_embedding = self.vector_store.model.encode([query])
        np.linalg.norm(query_embedding, ord=2, axis=1, keepdims=True)
        
        scores, indices = self.vector_store.index.search(query_embedding.astype('float32'), top_k)
        results = [
            RetrievalResult(chunk=self.vector_store.chunks[idx], dense_score=float(score), rank=i)
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])) if idx != -1
        ]
        return results

    def _sparse_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Performs sparse search using BM25."""
        query_tokens = [w for w in word_tokenize(query.lower()) if w.isalpha()]
        scores = self.vector_store.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            RetrievalResult(chunk=self.vector_store.chunks[idx], dense_score=0.0, sparse_score=scores[idx], rank=i)
            for i, idx in enumerate(top_indices)
        ]
        return results
    
    def _combine_and_rerank(self, dense: List[RetrievalResult], sparse: List[RetrievalResult]) -> List[RetrievalResult]:
        """Combines results using weighted score normalization."""
        all_results = {}
        
        # Process and normalize dense results
        if dense:
            # The dense scores (from IndexFlatIP) are already normalized dot products (cosine similarity)
            # and are typically in a good range. We can use them directly.
            for r in dense:
                chunk_id = (r.chunk.source_file, r.chunk.chunk_id)
                all_results[chunk_id] = {'dense': r.dense_score, 'sparse': 0.0, 'chunk': r.chunk}

        # Process and normalize sparse results
        if sparse:
            # BM25 scores are not bounded, so we need to normalize them
            max_sparse_score = max(r.sparse_score for r in sparse) if sparse else 1.0
            for r in sparse:
                chunk_id = (r.chunk.source_file, r.chunk.chunk_id)
                normalized_sparse = r.sparse_score / max_sparse_score
                if chunk_id in all_results:
                    all_results[chunk_id]['sparse'] = normalized_sparse
                else:
                    all_results[chunk_id] = {'dense': 0.0, 'sparse': normalized_sparse, 'chunk': r.chunk}

        # Calculate a weighted combined score
        final_results = []
        for chunk_id, scores in all_results.items():
            # Adjust weights here if needed (e.g., 0.6 for dense, 0.4 for sparse)
            dense_weight = 0.7
            sparse_weight = 0.3
            
            combined_score = (scores['dense'] * dense_weight) + (scores['sparse'] * sparse_weight)
            
            final_results.append(RetrievalResult(
                chunk=scores['chunk'],
                dense_score=scores['dense'],
                sparse_score=scores['sparse'], # Storing normalized score for consistency
                combined_score=combined_score
            ))

        # Sort by the new combined score
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Re-assign rank after sorting
        for i, res in enumerate(final_results):
            res.rank = i
            
        return final_results