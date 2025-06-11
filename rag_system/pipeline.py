# rag_system/pipeline.py
import os
import logging
from pathlib import Path
from typing import Dict, List

from .config import config
from .indexing import VectorStore
from .retrieval import HybridRetriever
from .generation import EnhancedGenerator

logger = logging.getLogger(__name__)

class RAGPipeline:
    """The main RAG system orchestrator."""
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.generator = EnhancedGenerator()
        self.initialized = False

    def initialize(self, force_rebuild: bool = False) -> bool:
        """Initializes the system by building the vector index."""
        logger.info("Initializing RAG pipeline...")
        pdf_dir = Path(config.PDF_DIR)
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            logger.error(f"PDF directory not found: {config.PDF_DIR}")
            return False
        
        pdf_paths = [str(f) for f in pdf_dir.glob("*.pdf")]
        if not pdf_paths:
            logger.error(f"No PDF files found in {config.PDF_DIR}")
            return False

        logger.info(f"Found {len(pdf_paths)} PDF files.")
        if self.vector_store.build_index(pdf_paths, force_rebuild):
            self.initialized = True
            logger.info("RAG pipeline initialized successfully.")
            return True
        else:
            logger.error("Failed to build vector store index.")
            return False

    def query(self, question: str) -> Dict:
        """Processes a query and returns the answer and sources."""
        if not self.initialized:
            return {"answer": "System is not initialized.", "sources": []}

        retrieval_results = self.retriever.retrieve(question)
        answer = self.generator.generate_answer(question, retrieval_results)
        
        sources = [
            {
                "file": Path(res.chunk.source_file).name,
                "page": res.chunk.page_number,
                "score": round(res.combined_score, 4),
                "text": res.chunk.text[:150] + "..."
            }
            for res in retrieval_results
        ]
        return {"answer": answer, "sources": sources}

    def get_stats(self) -> Dict:
        """Returns statistics about the loaded data."""
        if not self.initialized:
            return {"error": "System not initialized."}
        return {
            "total_chunks": len(self.vector_store.chunks),
            "pdf_files": list(set(c.source_file for c in self.vector_store.chunks)),
            "embedding_model": config.EMBEDDING_MODEL,
            "search_mode": "Hybrid" if config.USE_HYBRID_SEARCH else "Dense"
        }