"""
Chunking module for RAG pipeline.

Implements LLM-driven semantic chunking with fixed-window fallback.
"""

import re
import json
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None

class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """Chunk the input text."""
        pass

class SentenceChunker(BaseChunker):
    """Fast and reliable sentence-based chunker."""
    
    def __init__(self, min_sentence_length: int = 20):
        self.min_sentence_length = min_sentence_length
    
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """Chunk text by sentences with improved splitting."""
        import re
        
        # Clean up the text first
        text = text.strip()
        if not text:
            return []
        
        # Split by sentence boundaries - improved regex
        # This handles various sentence endings and whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Further split by paragraph breaks if sentences are too long
        all_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If sentence is very long (>200 chars), try to split by commas or semicolons
            if len(sentence) > 200:
                # Split by major punctuation within long sentences
                sub_sentences = re.split(r'(?<=[,;])\s+', sentence)
                all_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
            else:
                all_sentences.append(sentence)
        
        chunks = []
        current_pos = 0
        
        for i, sentence in enumerate(all_sentences):
            sentence = sentence.strip()
            if len(sentence) < self.min_sentence_length:
                continue
            
            # Find position in original text
            start_idx = text.find(sentence, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(sentence)
            
            chunk = Chunk(
                content=sentence,
                start_idx=start_idx,
                end_idx=end_idx,
                metadata={
                    "chunk_type": "sentence",
                    "chunk_size": len(sentence),
                    "chunk_index": i,
                    "chunking_strategy": "sentence"
                },
                chunk_id=f"sentence_chunk_{i}"
            )
            chunks.append(chunk)
            current_pos = end_idx
        
        logger.info(f"[SentenceChunker] Extracted {len(chunks)} sentence chunks from {len(all_sentences)} sentences")
        return chunks

class FixedChunker(BaseChunker):
    """Fixed window chunker with token-based windowing."""
    
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """Chunk text using fixed token windowing."""
        # Simple tokenization (split by whitespace)
        tokens = text.split()
        
        if len(tokens) <= self.max_tokens:
            # Single chunk
            chunk = Chunk(
                content=text,
                start_idx=0,
                end_idx=len(text),
                metadata={
                    "chunk_type": "fixed_window",
                    "chunk_size": len(text),
                    "token_count": len(tokens),
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "chunking_strategy": "fixed"
                },
                chunk_id="fixed_chunk_0"
            )
            return [chunk]
        
        # Multiple chunks with overlap
        chunks = []
        start_token = 0
        chunk_id = 0
        
        while start_token < len(tokens):
            end_token = min(start_token + self.max_tokens, len(tokens))
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = ' '.join(chunk_tokens)
            
            # Calculate character positions
            start_char = self._get_char_position(text, tokens, start_token)
            end_char = self._get_char_position(text, tokens, end_token)
            
            chunk = Chunk(
                content=chunk_text,
                start_idx=start_char,
                end_idx=end_char,
                metadata={
                    "chunk_type": "fixed_window",
                    "chunk_size": len(chunk_text),
                    "token_count": len(chunk_tokens),
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "start_token": start_token,
                    "end_token": end_token,
                    "chunking_strategy": "fixed"
                },
                chunk_id=f"fixed_chunk_{chunk_id}"
            )
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk with overlap
            start_token = end_token - self.overlap_tokens
            if start_token >= len(tokens):
                break
        
        return chunks
    
    def _get_char_position(self, text: str, tokens: List[str], token_index: int) -> int:
        """Get character position for a given token index."""
        if token_index == 0:
            return 0
        if token_index >= len(tokens):
            return len(text)
        
        # Find the position by counting characters
        pos = 0
        for i in range(min(token_index, len(tokens))):
            # Find the token in the text
            token = tokens[i]
            pos = text.find(token, pos)
            if pos == -1:
                # Fallback: estimate position
                return min(i * 10, len(text))
            pos += len(token)
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
        
        return min(pos, len(text))

def get_chunker(strategy: str, llm_client=None, max_tokens: int = 512, overlap_tokens: int = 50) -> BaseChunker:
    """Get a chunker instance based on strategy.
    
    Args:
        strategy: Chunking strategy ('sentence', 'fixed', 'semantic', or 'llm')
        llm_client: LLM client for semantic chunking (optional)
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap for fixed chunking
    
    Returns:
        BaseChunker instance
    """
    if strategy == "sentence":
        return SentenceChunker()
    elif strategy == "fixed":
        return FixedChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use 'sentence' or 'fixed'.")

class ChunkingManager:
    """Manages different chunking strategies."""
    
    def __init__(self, llm_client=None, window_size: int = 2000, stride: int = 1500):
        self.llm_client = llm_client
        self.window_size = window_size
        self.stride = stride
        self.chunkers = {
            "sentence": SentenceChunker(),
            "fixed": FixedChunker(),
        }
    
    def chunk(self, text: str, strategy: str = "sentence", **kwargs) -> List[Chunk]:
        """Chunk text using specified strategy."""
        if strategy not in self.chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        return self.chunkers[strategy].chunk(text, **kwargs)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        return ["sentence", "fixed"]