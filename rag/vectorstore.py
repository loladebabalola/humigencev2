"""
Vector store module for RAG pipeline.

Supports ChromaDB with namespace support and external vector stores.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    store_type: str = "chroma"  # "chroma", "pinecone", "weaviate", etc.
    collection_name: str = "humigence_rag"
    persist_directory: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    dimension: int = 384
    distance_metric: str = "cosine"
    namespace: str = "default"  # Namespace for pipeline profiles
    reset_collection: bool = True  # Reset collection before adding new documents

@dataclass
class SearchResult:
    """Represents a search result from vector store."""
    content: str
    metadata: Dict[str, Any]
    score: float
    id: str
    namespace: str = "default"

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation with namespace support."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection with namespace support."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Set up persist directory with namespace
            persist_dir = self.config.persist_directory or "./chroma_db"
            namespace_dir = Path(persist_dir) / self.config.namespace
            os.makedirs(namespace_dir, exist_ok=True)
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=str(namespace_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collection name with namespace
            collection_name = f"{self.config.collection_name}_{self.config.namespace}"
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": self.config.distance_metric,
                        "namespace": self.config.namespace,
                        "dimension": self.config.dimension
                    }
                )
                logger.info(f"Created new collection: {collection_name} in namespace: {self.config.namespace}")
            
        except ImportError:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            if self.collection:
                collection_name = self.collection.name
                logger.info(f"[VectorStore] Resetting collection '{collection_name}' before ingestion")
                
                # Delete the existing collection
                self.client.delete_collection(collection_name)
                
                # Recreate the collection
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                
                logger.info(f"[VectorStore] Collection '{collection_name}' reset successfully")
            else:
                logger.warning("[VectorStore] No collection to reset")
        except Exception as e:
            logger.error(f"[VectorStore] Failed to reset collection: {e}")
            raise

    def _normalize_metadata(self, md: dict, default_source: str = None) -> dict:
        """Normalize metadata to ensure consistent structure."""
        md = (md or {}).copy()
        # 1) Normalize 'source'
        src = md.get("source") or md.get("filename") or md.get("title") or md.get("document_id") or default_source or "unknown"
        # keep just the basename if it's a path-like
        src_str = str(src)
        md["source"] = os.path.basename(src_str) if "/" in src_str or "\\" in src_str else src_str

        # 2) Normalize 'chunk_id'
        if "chunk_id" not in md and "chunk_index" in md:
            md["chunk_id"] = md["chunk_index"]

        # 3) Optional fields (do NOT enforce)
        # file_path, page are optional; keep them if present, don't fail if missing

        # 4) Namespace default fallback
        md.setdefault("namespace", "default")
        
        # 5) Remove None values (ChromaDB doesn't accept them)
        md = {k: v for k, v in md.items() if v is not None}
        
        return md

    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        """Add documents to ChromaDB with batch processing and real-time progress logging."""
        try:
            # Reset collection if configured to do so
            if self.config.reset_collection:
                self.reset_collection()
            
            n = len(documents)
            total_batches = (n + batch_size - 1) // batch_size
            
            # Normalize metadata with soft validation
            strict = os.getenv("RAG_METADATA_STRICT", "0") == "1"
            normalized_metadatas = []
            for i, md in enumerate(metadatas or [{}]*len(documents)):
                nmd = self._normalize_metadata(md, default_source=None)
                # Soft checks: require at least a non-empty 'source'
                if not nmd.get("source"):
                    msg = f"[VectorStore] WARN: missing 'source' in metadata for item {i}; using 'unknown'."
                    print(msg)
                    nmd["source"] = "unknown"
                normalized_metadatas.append(nmd)

            # Only if strict is ON, enforce hard errors:
            if strict:
                for i, nmd in enumerate(normalized_metadatas):
                    if not nmd.get("source"):
                        raise ValueError(f"Missing required metadata field 'source' at index {i}")
            
            # Debug: Show first 2 normalized metadatas
            if normalized_metadatas:
                if os.getenv('RAG_DEBUG') == '1':
                    print("[DEBUG] First 2 normalized metadatas:", normalized_metadatas[:2])
            
            # Generate all IDs with proper format: filename::p{page}::c{chunk_id}
            all_ids = []
            for i, metadata in enumerate(normalized_metadatas):
                # Get filename and page information
                filename = metadata.get('source', metadata.get('file_name', metadata.get('filename', 'doc')))
                page = metadata.get('page', 0)
                chunk_id = metadata.get('chunk_id', i)
                
                # Create ID in format: filename::p{page}::c{chunk_id}
                all_ids.append(f"{filename}::p{page}::c{chunk_id}")
            
            # Add namespace to all metadata
            enhanced_metadatas = []
            for metadata in normalized_metadatas:
                enhanced_metadata = metadata.copy()
                enhanced_metadata["namespace"] = self.config.namespace
                enhanced_metadatas.append(enhanced_metadata)
            
            # Convert numpy array to list
            embeddings_list = embeddings.tolist()
            
            # Process in batches
            for i in range(0, n, batch_size):
                batch_texts = documents[i:i + batch_size]
                batch_embeddings = embeddings_list[i:i + batch_size]
                batch_metadatas = enhanced_metadatas[i:i + batch_size]
                batch_ids = all_ids[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Log progress
                print(f"[VectorStore] Inserting batch {batch_num}/{total_batches} ({i + len(batch_texts)}/{n} documents)")
                
                try:
                    # Add batch to collection
                    self.collection.add(
                        documents=batch_texts,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    
                    logger.info(f"Successfully inserted batch {batch_num}/{total_batches} ({len(batch_texts)} documents)")
                    
                except Exception as batch_error:
                    # Check if this is a dimension mismatch error
                    error_str = str(batch_error).lower()
                    if "dimension" in error_str and "expecting" in error_str:
                        logger.warning(f"[VectorStore] Embedding dimension mismatch detected in batch {batch_num}: {batch_error}")
                        
                        # Extract expected and actual dimensions from error message
                        try:
                            # Parse error message like "Collection expecting embedding with dimension of 384, got 768"
                            import re
                            match = re.search(r'dimension of (\d+), got (\d+)', str(batch_error))
                            if match:
                                expected_dim = int(match.group(1))
                                actual_dim = int(match.group(2))
                            else:
                                # Fallback: use collection metadata and actual embedding dimension
                                expected_dim = self.collection.metadata.get("dimension", "unknown")
                                actual_dim = len(embeddings[0]) if len(embeddings) > 0 else "unknown"
                            
                            logger.warning(f"[VectorStore] Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. Reinitializing collection...")
                            
                            # Reinitialize collection with correct dimension
                            self._reinitialize_collection(actual_dim)
                            
                            # Retry current batch
                            self.collection.add(
                                documents=batch_texts,
                                embeddings=batch_embeddings,
                                metadatas=batch_metadatas,
                                ids=batch_ids
                            )
                            
                            logger.info(f"✅ Collection reinitialized with embedding dimension: {actual_dim}")
                            logger.info(f"✅ Batch {batch_num} inserted successfully after reinitialization")
                            
                        except Exception as reinit_error:
                            logger.error(f"Failed to reinitialize collection: {reinit_error}")
                            raise
                    else:
                        # Re-raise non-dimension related errors
                        print(f"❌ Failed to insert batch {batch_num}: {str(batch_error)}")
                        logger.error(f"Failed to insert batch {batch_num}: {batch_error}")
                        raise
            
            # Success message
            print(f"[VectorStore] ✅ Successfully ingested {n} documents into collection '{self.collection.name}'")
            logger.info(f"Successfully ingested {n} documents into collection '{self.collection.name}' in {total_batches} batches")
            return all_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in ChromaDB with namespace filtering."""
        try:
            # Add namespace filter
            if filter is None:
                filter = {"namespace": self.config.namespace}
            else:
                filter = {**filter, "namespace": self.config.namespace}
            
            # Convert numpy array to list
            query_embedding_list = query_embedding.tolist()
            
            # Perform search with metadata included
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Debug: Show retrieved metadata only if RAG_DEBUG=1
            if os.getenv('RAG_DEBUG') == '1' and results.get('metadatas') and results['metadatas'][0]:
                print(f"[DEBUG] Retrieved metadatas: {results['metadatas'][0]}")
                logger.info(f"📋 Retrieved metadata: {results['metadatas'][0]}")
            
            # Convert to SearchResult objects
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    score = 1.0 - distance if distance <= 1.0 else 0.0
                    
                    search_results.append(SearchResult(
                        content=doc,
                        metadata=metadata or {},
                        score=score,
                        id=results['ids'][0][i],
                        namespace=self.config.namespace
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from namespace: {self.config.namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.config.collection_name,
                "namespace": self.config.namespace,
                "store_type": "chroma",
                "dimension": self.config.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def _reinitialize_collection(self, new_dimension: int):
        """Reinitialize collection with new embedding dimension."""
        try:
            collection_name = f"{self.config.collection_name}_{self.config.namespace}"
            
            # Delete existing collection
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete collection (may not exist): {e}")
            
            # Update config dimension
            self.config.dimension = new_dimension
            
            # Create new collection with correct dimension
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "dimension": new_dimension,
                    "namespace": self.config.namespace
                }
            )
            
            logger.info(f"Created new collection with dimension: {new_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to reinitialize collection: {e}")
            raise
    
    def list_namespaces(self) -> List[str]:
        """List available namespaces."""
        try:
            collections = self.client.list_collections()
            namespaces = set()
            for collection in collections:
                if collection.metadata and "namespace" in collection.metadata:
                    namespaces.add(collection.metadata["namespace"])
            return list(namespaces)
        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            return []

class VectorStoreManager:
    """Manages vector store operations with namespace support."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or self._get_default_config()
        self.store = self._create_vector_store()
    
    def _get_default_config(self) -> VectorStoreConfig:
        """Get default vector store configuration."""
        return VectorStoreConfig(
            store_type="chroma",
            collection_name="humigence_rag",
            persist_directory="./chroma_db",
            dimension=384,
            distance_metric="cosine",
            namespace="default"
        )
    
    def _create_vector_store(self) -> BaseVectorStore:
        """Create appropriate vector store based on config."""
        if self.config.store_type == "chroma":
            return ChromaVectorStore(self.config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.store_type}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        """Add documents to vector store with batch processing."""
        return self.store.add_documents(documents, embeddings, metadatas, batch_size)
    
    def search(self, query_embedding: np.ndarray, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents."""
        return self.store.search(query_embedding, k, filter)
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        return self.store.delete(ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.store.get_stats()
    
    def update_config(self, config: VectorStoreConfig):
        """Update vector store configuration."""
        self.config = config
        self.store = self._create_vector_store()
    
    def reinitialize(self, new_dimension: int):
        """Reinitialize vector store with new embedding dimension."""
        try:
            if hasattr(self.store, '_reinitialize_collection'):
                self.store._reinitialize_collection(new_dimension)
                logger.info(f"✅ Vector store reinitialized with dimension: {new_dimension}")
            else:
                logger.warning("Current vector store does not support reinitialization")
        except Exception as e:
            logger.error(f"Failed to reinitialize vector store: {e}")
            raise
    
    def set_namespace(self, namespace: str):
        """Set namespace for the vector store."""
        self.config.namespace = namespace
        self.store = self._create_vector_store()
    
    def list_namespaces(self) -> List[str]:
        """List available namespaces."""
        if hasattr(self.store, 'list_namespaces'):
            return self.store.list_namespaces()
        return [self.config.namespace]
    
    @staticmethod
    def get_available_stores() -> List[str]:
        """Get list of available vector store types."""
        return ["chroma"]
    
    def save_config(self, path: str):
        """Save vector store configuration to file."""
        config_dict = {
            "store_type": self.config.store_type,
            "collection_name": self.config.collection_name,
            "persist_directory": self.config.persist_directory,
            "api_key": self.config.api_key,
            "api_base": self.config.api_base,
            "dimension": self.config.dimension,
            "distance_metric": self.config.distance_metric,
            "namespace": self.config.namespace
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'VectorStoreManager':
        """Load vector store configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = VectorStoreConfig(**config_dict)
        return cls(config)

# Convenience functions for easy access
def store_embeddings(chunks: List[str], embeddings: np.ndarray, namespace: str = "default", 
                    collection_name: str = "humigence_rag", persist_directory: str = "./chroma_db") -> List[str]:
    """Convenience function to store embeddings in ChromaDB with namespace support.
    
    Args:
        chunks: List of text chunks to store
        embeddings: numpy array of embeddings
        namespace: Namespace for the pipeline profile
        collection_name: Name of the collection
        persist_directory: Directory to persist the database
    
    Returns:
        List of document IDs
    """
    config = VectorStoreConfig(
        store_type="chroma",
        collection_name=collection_name,
        persist_directory=persist_directory,
        dimension=embeddings.shape[1],
        namespace=namespace
    )
    
    manager = VectorStoreManager(config)
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    return manager.add_documents(chunks, embeddings, metadatas)

def get_default_vectorstore(namespace: str = "default") -> VectorStoreManager:
    """Get default vector store manager.
    
    Args:
        namespace: Namespace for the pipeline profile
    
    Returns:
        VectorStoreManager instance
    """
    config = VectorStoreConfig(
        store_type="chroma",
        collection_name="humigence_rag",
        persist_directory="./chroma_db",
        dimension=384,
        namespace=namespace
    )
    
    return VectorStoreManager(config)