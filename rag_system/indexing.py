# rag_system/indexing.py
import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from .data_structures import ChunkMetadata
from .data_processing import PDFProcessor
from .config import config

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages the creation, caching, and loading of vector and keyword indexes."""
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[ChunkMetadata] = []
        self.bm25: Optional[BM25Okapi] = None
        self.cache_dir = Path(config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, pdf_files: List[str]) -> str:
        """Generates a unique key based on file names and modification times."""
        info = "".join(f"{f}{os.path.getmtime(f)}" for f in sorted(pdf_files))
        return hashlib.md5(info.encode()).hexdigest()

    def _save_cache(self, key: str):
        """Saves the index, chunks, and BM25 model to disk."""
        try:
            index_path = self.cache_dir / f"index_{key}.faiss"
            data_path = self.cache_dir / f"data_{key}.pkl"
            faiss.write_index(self.index, str(index_path))
            with open(data_path, 'wb') as f:
                pickle.dump({'chunks': self.chunks, 'bm25': self.bm25}, f)
            logger.info(f"Cache saved successfully with key {key}.")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_cache(self, key: str) -> bool:
        """Loads indexes and data from cache if available."""
        index_path = self.cache_dir / f"index_{key}.faiss"
        data_path = self.cache_dir / f"data_{key}.pkl"
        if not index_path.exists() or not data_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(index_path))
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.bm25 = data['bm25']
            logger.info(f"Loaded {len(self.chunks)} chunks from cache.")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return False

    def build_index(self, pdf_paths: List[str], force_rebuild: bool = False) -> bool:
        """Builds or loads the FAISS and BM25 indexes."""
        cache_key = self._get_cache_key(pdf_paths)
        if not force_rebuild and self._load_cache(cache_key):
            return True

        logger.info("Building new index from PDFs...")
        processor = PDFProcessor()
        all_chunks = [chunk for path in pdf_paths for chunk in processor.process_pdf(path)]

        if not all_chunks:
            logger.error("No chunks were created from any PDF files. Halting index build.")
            return False

        self.chunks = all_chunks
        texts = [chunk.text for chunk in self.chunks]

        # Dense Index (FAISS)
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))

        # Sparse Index (BM25)
        logger.info("Creating BM25 index...")
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        self._save_cache(cache_key)
        logger.info(f"Index built and cached successfully. Total chunks: {len(self.chunks)}")
        return True