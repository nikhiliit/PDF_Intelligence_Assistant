# tests/test_retrieval.py

import numpy as np
import pytest
import nltk

from rag_system.indexing import VectorStore
from rag_system.retrieval import HybridRetriever

# Ensure NLTK data is present when tests run in CI
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

@pytest.fixture(scope="session")
def tiny_vector_store(tmp_path_factory):
    # Build a tiny in-memory store from two sentences
    store = VectorStore()
    dummy_texts = ["Cats drink milk.", "Dogs chase cats."]
    # Create chunks
    store.chunks = store._create_chunks(dummy_texts)            # type: ignore
    # Encode embeddings
    embeddings = store.model.encode(dummy_texts)
    # Normalize embeddings to unit length
    import faiss
    faiss.normalize_L2(embeddings)
    # Create FAISS index and add embeddings
    store.index = faiss.IndexFlatIP(embeddings.shape[1])
    store.index.add(embeddings.astype("float32"))
    # Build sparse BM25 index
    from rank_bm25 import BM25Okapi
    tokenised = [nltk.word_tokenize(t.lower()) for t in dummy_texts]
    store.bm25 = BM25Okapi(tokenised)
    return store

def test_query_vector_is_unit_length(tiny_vector_store):
    retriever = HybridRetriever(tiny_vector_store)
    # Encode a dummy query
    emb = tiny_vector_store.model.encode(["unit test"])
    # Manually normalize
    normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    # Norm of the normalized vector should be ~1
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-6)

def test_stopwords_removed(tiny_vector_store):
    retriever = HybridRetriever(tiny_vector_store)
    # Perform sparse retrieval on a query composed solely of stop-words + a keyword
    result = retriever._sparse_retrieval("the and a cat", 1)[0]
    tokens = result.chunk.text.lower()
    # The stop-words "the", "and", "a" should not appear in the chosen chunk's text
    assert "the" not in tokens and "and" not in tokens and "a" not in tokens
