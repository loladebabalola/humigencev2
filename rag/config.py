"""
RAG Configuration module.

Defines configuration classes and default settings for RAG pipeline components.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Pipeline identification
    pipeline_name: str = "humigence_rag"
    description: str = "Humigence RAG Pipeline"
    
    # Document processing
    chunking_strategy: str = "sentence"  # "sentence", "fixed", "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 100
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    multi_gpu_embeddings: bool = True
    
    # Vector store configuration
    vector_store: str = "chroma"  # "chroma", "pinecone", "qdrant"
    collection_name: str = "humigence_rag"
    namespace: str = "default"
    persist_directory: str = "./chroma_db"
    reset_collection: bool = True
    
    # LLM configuration
    llm_model: str = "microsoft/Phi-2"
    llm_model_type: str = "local"  # "local", "api", "huggingface", "gguf"
    llm_device: str = "auto"
    llm_model_path: Optional[str] = None
    llm_quantized: bool = False
    context_length: int = 8192
    max_tokens: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Retrieval configuration
    retrieval_strategy: str = "semantic"  # "semantic", "hybrid", "rerank"
    top_k: int = 5
    similarity_threshold: float = 0.25  # Lower threshold for better recall
    min_hits: int = 3
    use_reranking: bool = True
    rerank_top_k: int = 20
    
    # Semantic chunking
    semantic_chunking_model: str = "microsoft/phi-2"
    window_size: int = 2000
    stride: int = 1500
    
    # Performance settings
    vector_store_batch_size: int = 100