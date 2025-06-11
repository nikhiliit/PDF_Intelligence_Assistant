# rag_system/data_structures.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ChunkMetadata:
    """Data structure for a single text chunk."""
    text: str
    source_file: str
    chunk_id: int
    page_number: int
    char_start: int
    char_end: int
    created_at: str
    word_count: int

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChunkMetadata':
        return cls(**data)

@dataclass
class RetrievalResult:
    """Data structure for a single retrieval result."""
    chunk: ChunkMetadata
    dense_score: float
    sparse_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0