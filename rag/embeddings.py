"""
Embedding module for RAG pipeline.

Supports multiple embedding models with multi-GPU acceleration.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    multi_gpu: bool = True
    batch_size: int = 32

class EmbeddingManager:
    """Manages embedding models and operations."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 multi_gpu: bool = True, device: str = "auto"):
        self.model_name = model_name
        self.multi_gpu = multi_gpu
        self.device = device
        self.model = None
        self.tokenizer = None
        self.device_count = 1
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    if self.multi_gpu and torch.cuda.device_count() > 1:
                        self.device_count = torch.cuda.device_count()
                        logger.info(f"Multi-GPU mode: {self.device_count} GPUs available")
                else:
                    self.device = "cpu"
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Note: Multi-GPU for sentence-transformers is handled internally
            # DataParallel wrapping can cause issues with the encode method
            if self.multi_gpu and torch.cuda.device_count() > 1:
                logger.info(f"Multi-GPU acceleration available on {torch.cuda.device_count()} GPUs")
            
            logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
            
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            if not texts:
                return np.array([])
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        try:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True
            )
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "multi_gpu": self.multi_gpu,
            "device_count": self.device_count,
            "dimension": self.model.get_sentence_embedding_dimension() if self.model else None
        }

def get_default_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingManager:
    """Get default embedding manager."""
    return EmbeddingManager(model_name=model_name)
