# rag_system/data_processing.py
import os
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

from .data_structures import ChunkMetadata
from .config import config

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class PDFProcessor:
    """Handles loading, parsing, and chunking of PDF documents."""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def process_pdf(self, pdf_path: str) -> List[ChunkMetadata]:
        """Loads, chunks, and creates metadata for a single PDF."""
        logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
        full_text, page_metadata = self._load_pdf_with_metadata(pdf_path)
        if not full_text.strip():
            logger.warning(f"No text extracted from {os.path.basename(pdf_path)}")
            return []
        
        chunks = self._smart_chunk_text(full_text, pdf_path, page_metadata)
        logger.info(f"Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        return chunks

    def _load_pdf_with_metadata(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Loads text from a PDF and tracks character offsets for each page."""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            page_metadata = []
            for page_num, page in enumerate(doc):
                page_text = ' '.join(page.get_text().split())
                if not page_text.strip():
                    continue

                page_info = {
                    'page_number': page_num + 1,
                    'char_start': len(full_text),
                    'char_end': len(full_text) + len(page_text),
                }
                page_metadata.append(page_info)
                full_text += page_text + "\n"
            doc.close()
            return full_text, page_metadata
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return "", []

    def _smart_chunk_text(self, text: str, source_file: str, page_metadata: List[Dict]) -> List[ChunkMetadata]:
        """Splits text into chunks based on sentences and overlap."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_start_char = 0
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > config.CHUNK_SIZE and current_chunk:
                chunks.append(self._create_chunk_metadata(current_chunk, source_file, chunk_id, current_start_char, page_metadata))
                
                overlap_text = self._get_overlap_text(current_chunk)
                current_start_char += len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                if not current_chunk:
                    current_start_char = text.find(sentence, current_start_char)
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(self._create_chunk_metadata(current_chunk, source_file, chunk_id, current_start_char, page_metadata))

        return [c for c in chunks if c is not None]

    def _get_overlap_text(self, text: str) -> str:
        """Gets the last `CHUNK_OVERLAP` characters, respecting word boundaries."""
        if len(text) <= config.CHUNK_OVERLAP:
            return text
        overlap_text = text[-config.CHUNK_OVERLAP:]
        space_idx = overlap_text.find(' ')
        return overlap_text[space_idx + 1:] if space_idx != -1 else overlap_text

    def _create_chunk_metadata(self, text_chunk: str, source: str, cid: int, start: int, p_meta: List[Dict]) -> Optional[ChunkMetadata]:
        """Creates a ChunkMetadata object."""
        text_chunk = text_chunk.strip()
        if len(text_chunk) < 20: return None # Skip very short chunks

        page_number = next((p['page_number'] for p in p_meta if start >= p['char_start'] and start < p['char_end']), 1)
        
        return ChunkMetadata(
            text=text_chunk,
            source_file=source,
            chunk_id=cid,
            page_number=page_number,
            char_start=start,
            char_end=start + len(text_chunk),
            created_at=datetime.now().isoformat(),
            word_count=len([w for w in word_tokenize(text_chunk.lower()) if w.isalpha()])
        )