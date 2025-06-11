# rag_system/config.py
from dataclasses import dataclass

@dataclass
class Config:
    """Central configuration for the RAG system."""
    # Directories
    PDF_DIR: str = "./docs"
    CACHE_DIR: str = "./cache"
    LOG_FILE: str = "rag_system.log"

    # Chunking parameters
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 100

    # Model settings
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"

    # Retrieval settings
    TOP_K_RETRIEVAL: int = 5
    TOP_K_RERANK: int = 3
    MIN_RELEVANCE_SCORE: float = 0.3
    USE_HYBRID_SEARCH: bool = True

    # Generation & Conversation settings
    ENABLE_CONVERSATION_HISTORY: bool = True
    MAX_CONTEXT_LENGTH: int = 4000
    MAX_HISTORY_MESSAGES: int = 5

# Create a single instance to be imported by other modules
config = Config()