"""
Retrieval module for RAG pipeline.

Implements semantic, hybrid, and re-ranking retrieval strategies.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rag.vectorstore import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for retrieval strategies."""
    strategy: str = "semantic"  # "semantic", "hybrid", "rerank"
    top_k: int = 5
    similarity_threshold: float = 0.25  # Lower threshold for better recall
    min_hits: int = 3
    use_reranking: bool = True
    rerank_top_k: int = 20

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[SearchResult]:
        """Retrieve relevant documents for query."""
        pass

class SemanticRetriever(BaseRetriever):
    """Semantic retrieval using vector similarity."""
    
    def __init__(self, vectorstore, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[SearchResult]:
        """Retrieve documents using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Search vector store
            results = self.vectorstore.search(query_embedding, k=k)
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []

class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining semantic and keyword search."""
    
    def __init__(self, vectorstore, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5, **kwargs) -> List[SearchResult]:
        """Retrieve documents using hybrid approach."""
        try:
            # Semantic search
            query_embedding = self.embedder.embed_query(query)
            semantic_results = self.vectorstore.search(query_embedding, k=k*2)
            
            # Simple keyword matching (can be enhanced)
            keyword_results = self._keyword_search(query, k)
            
            # Combine and re-rank results
            combined_results = self._combine_results(semantic_results, keyword_results)
            
            # Return top k results
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        """Simple keyword search implementation."""
        # This is a placeholder - in practice, you'd use a proper keyword search
        return []
    
    def _combine_results(self, semantic_results: List[SearchResult], 
                        keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Combine and re-rank results from different retrieval methods."""
        # Simple combination - can be enhanced with more sophisticated ranking
        all_results = semantic_results + keyword_results
        
        # Remove duplicates based on content
        seen_content = set()
        unique_results = []
        for result in all_results:
            if result.content not in seen_content:
                seen_content.add(result.content)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results

class RetrievalManager:
    """Manages different retrieval strategies."""
    
    def __init__(self, vectorstore, embedder, config: Optional[RetrievalConfig] = None):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.retriever = self._create_retriever()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create appropriate retriever based on config."""
        if self.config.strategy == "semantic":
            return SemanticRetriever(self.vectorstore, self.embedder)
        elif self.config.strategy == "hybrid":
            return HybridRetriever(self.vectorstore, self.embedder)
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.config.strategy}")
    
    def retrieve(self, query: str, k: Optional[int] = None, **kwargs) -> List[SearchResult]:
        """Retrieve relevant documents for query."""
        k = k or self.config.top_k
        return self.retriever.retrieve(query, k=k, **kwargs)
    
    def update_config(self, config: RetrievalConfig):
        """Update retrieval configuration."""
        self.config = config
        self.retriever = self._create_retriever()
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies."""
        return ["semantic", "hybrid"]