# rag_system/retrieval.py
import logging
import numpy as np
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from .indexing import VectorStore
from .data_structures import RetrievalResult
from .config import config

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines dense and sparse retrieval methods to find relevant documents."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        # Proper English stop-words set for sparse retrieval
        self.stop_words = set(stopwords.words("english"))

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
        # Encode and normalize the query embedding
        query_embedding = self.vector_store.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        scores, indices = self.vector_store.index.search(
            query_embedding.astype("float32"), top_k
        )
        results = [
            RetrievalResult(chunk=self.vector_store.chunks[idx],
                            dense_score=float(score),
                            rank=i)
            for i, (idx, score) in enumerate(zip(indices[0], scores[0]))
            if idx != -1
        ]
        return results

    def _sparse_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Performs sparse search using BM25."""
        # Strip stop-words before BM25 scoring
        query_tokens = [
            w for w in word_tokenize(query.lower())
            if w.isalpha() and w not in self.stop_words
        ]
        scores = self.vector_store.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            RetrievalResult(chunk=self.vector_store.chunks[idx],
                            dense_score=0.0,
                            sparse_score=scores[idx],
                            rank=i)
            for i, idx in enumerate(top_indices)
        ]
        return results

    def _combine_and_rerank(self,
                            dense: List[RetrievalResult],
                            sparse: List[RetrievalResult]
                           ) -> List[RetrievalResult]:
        """Combines results using weighted score normalization."""
        all_results = {}

        # Normalize dense results
        if dense:
            max_dense = max(r.dense_score for r in dense) or 1.0
            for r in dense:
                cid = (r.chunk.source_file, r.chunk.chunk_id)
                all_results[cid] = {
                    "dense": r.dense_score / max_dense,
                    "sparse": 0.0,
                    "chunk": r.chunk
                }

        # Normalize sparse results
        if sparse:
            max_sparse = max(r.sparse_score for r in sparse) or 1.0
            for r in sparse:
                cid = (r.chunk.source_file, r.chunk.chunk_id)
                normalized = r.sparse_score / max_sparse
                if cid in all_results:
                    all_results[cid]["sparse"] = normalized
                else:
                    all_results[cid] = {"dense": 0.0, "sparse": normalized, "chunk": r.chunk}

        # Combine with weights
        final_results = []
        dense_w, sparse_w = 0.7, 0.3
        for scores in all_results.values():
            combined = scores["dense"] * dense_w + scores["sparse"] * sparse_w
            final_results.append(RetrievalResult(
                chunk=scores["chunk"],
                dense_score=scores["dense"],
                sparse_score=scores["sparse"],
                combined_score=combined
            ))

        # Sort and re-rank
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, res in enumerate(final_results):
            res.rank = i

        return final_results
