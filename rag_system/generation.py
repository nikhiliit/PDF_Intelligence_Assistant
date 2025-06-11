# rag_system/generation.py
import logging
import requests
from typing import List, Dict
from datetime import datetime
from pathlib import Path

from .data_structures import RetrievalResult
from .config import config

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages the history of the conversation."""
    def __init__(self):
        self.history: List[Dict] = []

    def add_exchange(self, question: str, answer: str):
        self.history.append({'question': question, 'answer': answer})
        if len(self.history) > config.MAX_HISTORY_MESSAGES:
            self.history.pop(0)

    def get_history_prompt(self) -> str:
        if not self.history: return ""
        
        history_str = "\n".join([f"Previous Q: {h['question']}\nPrevious A: {h['answer']}" for h in self.history])
        return f"Previous conversation history:\n{history_str}\n\n"

class EnhancedGenerator:
    """Generates answers using a large language model (Ollama)."""
    def __init__(self):
        self.conversation = ConversationManager() if config.ENABLE_CONVERSATION_HISTORY else None

    def generate_answer(self, question: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "I couldn't find any relevant information to answer your question."
        
        context = self._prepare_context(results)
        prompt = self._build_prompt(question, context)

        answer = self._query_ollama(prompt)
        
        if self.conversation:
            self.conversation.add_exchange(question, answer)
        
        return answer

    def _prepare_context(self, results: List[RetrievalResult]) -> str:
        """Formats retrieved chunks into a context string."""
        context_parts = []
        for res in results:
            source_info = f"[Source: {Path(res.chunk.source_file).name}, Page: {res.chunk.page_number}]"
            context_parts.append(f"{res.chunk.text}\n{source_info}")
        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        history_prompt = self.conversation.get_history_prompt() if self.conversation else ""
        
        return (
            f"{history_prompt}"
            "Based on the following context, please provide a comprehensive and accurate answer to the question. "
            "Cite the sources provided with each piece of context using the format [Source: file, Page: number].\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

    def _query_ollama(self, prompt: str) -> str:
        """Sends a request to the Ollama API and handles retries."""
        for attempt in range(3):
            try:
                response = requests.post(
                    config.OLLAMA_URL,
                    json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                    timeout=45
                )
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except requests.RequestException as e:
                logger.warning(f"Ollama request failed on attempt {attempt+1}: {e}")
                if attempt == 2:
                    logger.error("Ollama connection failed after multiple retries.")
                    return "Error: Could not connect to the language model."
        return "Error: Failed to generate a response."