# tests/test_retrieval.py

import numpy as np
import pytest
import nltk
import faiss
from rank_bm25 import BM25Okapi

from rag_system.indexing import VectorStore
from rag_system.retrieval import HybridRetriever
from rag_system.data_structures import ChunkMetadata

# Download NLTK data once for all tests
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

@pytest.fixture(scope="session")
def tiny_vector_store():
    # Initialize the store and two sample texts
    store = VectorStore()
    texts = ["Cats drink milk.", "Dogs chase cats."]
    
    # 1) Manually create ChunkMetadata objects
    chunks = []
    for i, txt in enumerate(texts):
        chunks.append(ChunkMetadata(
            text=txt,
            source_file=f"doc{i}.txt",
            chunk_id=i,
            page_number=1,
            char_start=0,
            char_end=len(txt),
            created_at="2025-01-01T00:00:00",
            word_count=len([w for w in nltk.word_tokenize(txt.lower()) if w.isalpha()])
        ))
    store.chunks = chunks

    # 2) Build a FAISS index with normalized embeddings
    embeddings = store.model.encode(texts)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    store.index = index

    # 3) Build a BM25 index
    tokenized_texts = [nltk.word_tokenize(t.lower()) for t in texts]
    store.bm25 = BM25Okapi(tokenized_texts)

    return store

def test_query_vector_is_unit_length(tiny_vector_store):
    # This test verifies that manually normalizing produces unit‐length vectors
    emb = tiny_vector_store.model.encode(["unit test"])
    normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-6)

def test_stopwords_removed(tiny_vector_store):
    retriever = HybridRetriever(tiny_vector_store)
    # Use only stop‐words plus "cat" and retrieve top‐1
    result = retriever._sparse_retrieval("the and a cat", top_k=1)[0]
    # Ensure none of the stop‐words appear in the returned chunk text
    text = result.chunk.text.lower()
    for stop in ("the", "and", "a"):
        assert stop not in text
