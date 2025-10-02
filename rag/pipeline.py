"""
RAG Pipeline module.

Orchestrates the complete RAG pipeline with ingestion, chunking, embedding, and retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .config import RAGConfig
from .chunking import ChunkingManager, get_chunker
from .embeddings import EmbeddingManager
from .vectorstore import VectorStoreManager, VectorStoreConfig
from .llm import LLMManager, LLMConfig
from .retrieval import RetrievalManager, RetrievalConfig

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, config: RAGConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        
        # Initialize components
        self.chunker = ChunkingManager()
        self.embedder = EmbeddingManager(
            model_name=config.embedding_model,
            multi_gpu=config.multi_gpu_embeddings,
            device=config.embedding_device
        )
        
        # Initialize vector store
        vectorstore_config = VectorStoreConfig(
            store_type=config.vector_store,
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            dimension=config.embedding_dimension,
            namespace=config.namespace
        )
        self.vectorstore = VectorStoreManager(vectorstore_config)
        
        # Initialize LLM if not provided
        if not self.llm_client:
            llm_config = LLMConfig(
                model_name=config.llm_model,
                model_type=config.llm_model_type,
                device=config.llm_device,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
            self.llm_client = LLMManager(llm_config)
        
        # Initialize retrieval
        retrieval_config = RetrievalConfig(
            strategy=config.retrieval_strategy,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            min_hits=config.min_hits
        )
        self.retriever = RetrievalManager(
            self.vectorstore.store,
            self.embedder,
            retrieval_config
        )
    
    def ingest(self, documents: List[str], chunking_strategy: str = "sentence", 
               embed_model: str = None, vectorstore: str = None, 
               metadatas: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest documents into the RAG pipeline."""
        try:
            # Update config if provided
            if embed_model:
                self.config.embedding_model = embed_model
                self.embedder = EmbeddingManager(
                    model_name=embed_model,
                    multi_gpu=self.config.multi_gpu_embeddings,
                    device=self.config.embedding_device
                )
            
            if vectorstore:
                self.config.vector_store = vectorstore
                vectorstore_config = VectorStoreConfig(
                    store_type=vectorstore,
                    collection_name=self.config.collection_name,
                    persist_directory=self.config.persist_directory,
                    dimension=self.config.embedding_dimension,
                    namespace=self.config.namespace
                )
                self.vectorstore = VectorStoreManager(vectorstore_config)
            
            # Process documents
            all_chunks = []
            all_chunk_metadatas = []
            
            for i, doc in enumerate(documents):
                # Chunk document
                chunks = self.chunker.chunk(doc, strategy=chunking_strategy)
                
                # Prepare metadata for each chunk
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                for j, chunk in enumerate(chunks):
                    # Merge parent doc metadata and set chunk fields
                    chunk_meta = {**(metadata or {})}
                    chunk_meta["chunk_id"] = j  # always present
                    chunk_meta["chunk_index"] = j
                    chunk_meta["chunk_size"] = len(chunk.content)
                    chunk_meta.setdefault("chunk_type", "sentence")
                    chunk_meta.setdefault("chunking_strategy", "sentence")
                    # If source still missing, leave it as-is (vectorstore will warn and set 'unknown' if needed)
                    
                    # Preserve document-level metadata
                    file_path = metadata.get("file_path", "")
                    source_name = metadata.get("source", f"document_{i}")
                    
                    # Create stable document ID based on file path
                    if file_path:
                        import hashlib
                        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
                    else:
                        doc_id = f"doc_{i}"
                    
                    # Get page information if available
                    page = None
                    if "page_metadata" in metadata:
                        page_info = self._find_page_for_chunk(chunk, metadata["page_metadata"])
                        if page_info:
                            page = page_info["page"]
                    
                    chunk_meta.update({
                        "source": source_name,
                        "file_path": file_path,
                        "source_type": metadata.get("source_type", "document"),
                        "page": page,
                        "chunking_strategy": chunking_strategy,
                        "document_id": doc_id,
                        "document_length": len(doc),
                        **metadata
                    })
                    
                    chunk.metadata = chunk_meta
                    
                    all_chunks.append(chunk)
                    all_chunk_metadatas.append(chunk.metadata)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedder.embed_texts(chunk_texts)
            
            # Store in vector database
            doc_ids = self.vectorstore.add_documents(
                chunk_texts,
                embeddings,
                all_chunk_metadatas,
                batch_size=self.config.vector_store_batch_size
            )
            
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(documents)} documents")
            
            return {
                "status": "success",
                "chunks_created": len(all_chunks),
                "documents_processed": len(documents),
                "embedding_model": self.config.embedding_model,
                "chunking_strategy": chunking_strategy,
                "vector_store": self.config.vector_store
            }
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "chunks_created": 0,
                "documents_processed": 0
            }
    
    def ingest_pdf(self, pdf_path: str, chunking_strategy: str = "sentence") -> Dict[str, Any]:
        """Ingest a single PDF file with proper page metadata."""
        try:
            import PyPDF2
            
            # Extract text from PDF with page information
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                page_count = len(pdf_reader.pages)
                page_metadata = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                            page_metadata.append({
                                "page": page_num + 1,
                                "text": page_text,
                                "start_char": len(text) - len(page_text),
                                "end_char": len(text)
                            })
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                        continue
            
            if not text.strip():
                return {
                    "status": "error",
                    "error": "No text extracted from PDF",
                    "chunks_created": 0
                }
            
            # Prepare metadata with proper filename
            filename = os.path.basename(pdf_path)
            metadata = {
                "source": filename,
                "file_path": pdf_path,
                "source_type": "document",
                "page_count": page_count
            }
            
            # Ingest using main pipeline
            result = self.ingest([text.strip()], chunking_strategy=chunking_strategy, metadatas=[metadata])
            
            if result["status"] == "success":
                result["source_type"] = "pdf"
                result["page_count"] = page_count
                result["filename"] = filename
            
            return result
            
        except Exception as e:
            logger.error(f"PDF ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "chunks_created": 0
            }
    
    def ingest_multiple_pdfs(self, pdf_paths: List[str], chunking_strategy: str = "sentence") -> Dict[str, Any]:
        """Ingest multiple PDF files with proper metadata."""
        try:
            results = []
            total_chunks = 0
            successful_files = 0
            
            # Reset vectorstore once at the beginning for multi-PDF ingestion
            if self.vectorstore and hasattr(self.vectorstore, 'reset_collection'):
                self.vectorstore.reset_collection()
            
            for pdf_path in pdf_paths:
                # Temporarily disable reset for individual PDF ingestion
                original_reset = self.vectorstore.config.reset_collection if self.vectorstore else True
                if self.vectorstore:
                    self.vectorstore.config.reset_collection = False
                
                result = self.ingest_pdf(pdf_path, chunking_strategy)
                
                # Restore original reset setting
                if self.vectorstore:
                    self.vectorstore.config.reset_collection = original_reset
                
                results.append({
                    "file": os.path.basename(pdf_path),
                    "status": result["status"],
                    "chunks": result.get("chunks_created", 0),
                    "pages": result.get("page_count", 0),
                    "error": result.get("error", "")
                })
                
                if result["status"] == "success":
                    total_chunks += result["chunks_created"]
                    successful_files += 1
            
            return {
                "status": "success" if successful_files > 0 else "error",
                "results": results,
                "total_chunks": total_chunks,
                "successful_files": successful_files,
                "total_files": len(pdf_paths)
            }
            
        except Exception as e:
            logger.error(f"Multiple PDF ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total_chunks": 0
            }
    
    def query(self, question: str, top_k: int = None, **kwargs) -> Dict[str, Any]:
        """Query the RAG pipeline with clean, direct answers."""
        try:
            # Retrieve relevant documents
            top_k = top_k or self.config.top_k
            results = self.retriever.retrieve(question, k=top_k)
            
            if not results:
                return {
                    "answer": "I cannot answer based on the provided documents.",
                    "sources": [],
                    "retrieval_count": 0
                }
            
            # Filter results by similarity threshold
            filtered_results = [
                result for result in results 
                if result.score >= self.config.similarity_threshold
            ]
            
            if not filtered_results:
                return {
                    "answer": "I cannot answer based on the provided documents.",
                    "sources": [],
                    "retrieval_count": 0
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for result in filtered_results:
                context_parts.append(result.content)
                sources.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score
                })
            
            context = "\n\n".join(context_parts)
            
            # Generate answer with clean prompting
            prompt = self._build_prompt(question, context)
            answer = self.llm_client.generate(prompt, **kwargs)
            
            # Format clean sources
            clean_sources = self._format_sources(sources)
            
            # Debug mode output
            if os.environ.get("RAG_DEBUG", "0") == "1":
                debug_info = {
                    "prompt_sent_to_llm": prompt,
                    "raw_output": answer,
                    "retrieved_chunks": [
                        {
                            "content": r.content[:100] + "..." if len(r.content) > 100 else r.content,
                            "score": r.score,
                            "metadata": r.metadata
                        } for r in filtered_results
                    ]
                }
                return {
                    "answer": answer,
                    "sources": clean_sources,
                    "retrieval_count": len(filtered_results),
                    "context": context,
                    "debug": debug_info
                }
            
            return {
                "answer": answer,
                "sources": clean_sources,
                "retrieval_count": len(filtered_results),
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": "I cannot answer based on the provided documents.",
                "sources": [],
                "retrieval_count": 0
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build clean prompt for LLM generation."""
        return f"""You are a concise AI assistant.
Use ONLY the provided context to answer.
Rules:
• Give direct, clear answers (2–5 sentences).
• Cite sources at the end.
• If context is insufficient, say: "I cannot answer based on the provided documents."
• Do NOT use meta-commentary ("The context says…").

Context:
{context}

Question: {question}

Answer:"""
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Format sources for clean display."""
        if not sources:
            return []
        
        clean_sources = []
        for source in sources:
            metadata = source.get('metadata', {})
            source_name = metadata.get('source', 'unknown')
            page = metadata.get('page')
            
            if page is not None:
                clean_sources.append(f"{source_name} (p.{page})")
            else:
                clean_sources.append(source_name)
        
        # Remove duplicates while preserving order
        unique_sources = list(dict.fromkeys(clean_sources))
        return unique_sources
    
    def _find_page_for_chunk(self, chunk, page_metadata):
        """Find page number for a chunk based on page metadata."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated page detection
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "vectorstore": self.vectorstore.get_stats(),
            "embedder": self.embedder.get_model_info(),
            "llm": self.llm_client.get_model_info(),
            "config": {
                "chunking_strategy": self.config.chunking_strategy,
                "embedding_model": self.config.embedding_model,
                "vector_store": self.config.vector_store,
                "retrieval_strategy": self.config.retrieval_strategy
            }
        }