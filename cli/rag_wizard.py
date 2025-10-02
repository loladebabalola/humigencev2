"""
RAG Implementation Wizard CLI.

Interactive wizard for setting up and using RAG pipelines.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add the parent directory to the path so we can import from rag
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from rag.pipeline import RAGPipeline
from rag.config import RAGConfig
from rag.llm import LLMManager, LLMConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

console = Console()

class RAGWizard:
    """Interactive RAG Implementation Wizard."""
    
    def __init__(self):
        self.config = RAGConfig()
        self.pipeline = None
        self.llm_client = None
        self.documents = []
        self.document_sources = []
        self.ingestion_report = []
    
    def run(self):
        """Run the RAG Implementation Wizard."""
        self._show_welcome_header()
        
        # Step-by-step wizard flow
        self._step_configure_pipeline()
        self._step_load_documents()
        self._step_configure_chunking()
        self._step_configure_embeddings()
        self._step_configure_vectorstore()
        self._step_configure_llm()
        self._step_configure_retrieval()
        self._step_test_pipeline()
        
        # Launch interactive query mode
        self._launch_interactive_mode()
    
    def _show_welcome_header(self):
        """Display the welcome header with ASCII art."""
        console.print("\n" + "="*80)
        console.print("╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮  ╭─╮")
        console.print("│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │")
        console.print("╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯  ╰─╯")
        console.print("\n🤖 RAG Implementation Wizard")
        console.print("Set up and use Retrieval-Augmented Generation pipelines")
        console.print("="*80)
    
    def _step_configure_pipeline(self):
        """Step 1: Configure Pipeline"""
        console.print("\n🚀 [bold blue]Step 1: Configure Pipeline[/bold blue]")
        console.print("-" * 50)
        
        self.config.pipeline_name = Prompt.ask("Pipeline name", default=self.config.pipeline_name)
        self.config.description = Prompt.ask("Description", default=self.config.description)
        
        console.print(f"\n✅ [green]Pipeline configured: {self.config.pipeline_name}[/green]")
        console.print(f"📝 Description: {self.config.description}")
    
    def _step_load_documents(self):
        """Step 2: Load Documents"""
        console.print("\n📚 [bold blue]Step 2: Load Documents[/bold blue]")
        console.print("-" * 50)
        
        # Check for default docs folder first
        docs_folder = Path.home() / "humigence" / "docs"
        
        if docs_folder.exists():
            pdf_files = list(docs_folder.glob("*.pdf"))
            if pdf_files:
                console.print(f"🔍 Found {len(pdf_files)} PDF(s) in default docs folder:")
                for i, pdf_file in enumerate(pdf_files, 1):
                    file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
                    console.print(f"  {i}. {pdf_file.name} ({file_size:.1f} MB)")
                
                use_default = Confirm.ask("\nUse these PDFs?", default=True)
                if use_default:
                    self._process_pdf_files(pdf_files)
                    return
        
        # Manual document loading
        console.print("\n📄 Manual Document Loading")
        doc_type = Prompt.ask("Document type", choices=["text", "pdf"], default="text")
        
        if doc_type == "text":
            text = Prompt.ask("Enter document text")
            if text:
                self.documents.append(text)
                self.document_sources.append({"type": "text", "content": text[:100] + "..."})
                console.print("✅ [green]Text document added[/green]")
        elif doc_type == "pdf":
            pdf_path = Prompt.ask("Enter PDF file path")
            if os.path.exists(pdf_path):
                self._process_pdf_files([Path(pdf_path)])
            else:
                console.print("❌ [red]PDF file not found[/red]")
    
    def _process_pdf_files(self, pdf_files: List[Path]):
        """Process PDF files for ingestion with improved error handling."""
        console.print(f"\n🚀 Processing {len(pdf_files)} PDF(s)...")
        
        # Initialize pipeline if not already done
        if not self.pipeline:
            self.pipeline = RAGPipeline(self.config, llm_client=self.llm_client)
        
        # Use the new multi-PDF ingestion method
        pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
        result = self.pipeline.ingest_multiple_pdfs(pdf_paths, chunking_strategy="sentence")
        
        if result["status"] == "success":
            console.print(f"✅ Successfully ingested {result['successful_files']}/{result['total_files']} files")
            console.print(f"📊 Total chunks created: {result['total_chunks']}")
            
            # Display individual file results
            for file_result in result["results"]:
                if file_result["status"] == "success":
                    console.print(f"  ✅ {file_result['file']} – {file_result['chunks']} chunks, {file_result['pages']} pages")
                else:
                    console.print(f"  ❌ {file_result['file']} – {file_result['error']}")
        else:
            console.print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
        
        # Update documents list for testing
        self.documents = [f"PDF: {r['file']}" for r in result["results"] if r["status"] == "success"]
        self.document_sources = [{"type": "pdf", "filename": r["file"], "path": str(pdf_files[0].parent / r["file"])} for r in result["results"] if r["status"] == "success"]
        self.ingestion_report = result["results"]
    
    def _step_configure_chunking(self):
        """Step 3: Configure Chunking Strategy"""
        console.print("\n🔧 [bold blue]Step 3: Configure Chunking Strategy[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n📋 Available chunking strategies:")
        console.print("  • sentence - Split by sentences (recommended for most use cases)")
        console.print("  • fixed - Fixed window size with overlap")
        console.print("  • semantic - Semantic boundary detection (experimental)")
        
        strategy = Prompt.ask(
            "Chunking strategy",
            choices=["sentence", "fixed", "semantic"],
            default=self.config.chunking_strategy
        )
        
        self.config.chunking_strategy = strategy
        logger.debug(f"[DEBUG] Selected chunking strategy: {strategy}")
        
        if strategy == "fixed":
            self.config.chunk_size = int(Prompt.ask("Chunk size (characters)", default=str(self.config.chunk_size)))
            self.config.chunk_overlap = int(Prompt.ask("Chunk overlap (characters)", default=str(self.config.chunk_overlap)))
            logger.debug(f"[DEBUG] Fixed chunking: size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        elif strategy == "semantic":
            console.print("⚠️ [yellow]Semantic chunking is experimental and may be slower[/yellow]")
            self.config.semantic_chunk_size = int(Prompt.ask("Target chunk size", default="1000"))
            logger.debug(f"[DEBUG] Semantic chunking: target_size={self.config.semantic_chunk_size}")
        
        console.print(f"\n✅ [green]Chunking configured: {strategy}[/green]")
        if strategy == "fixed":
            console.print(f"📏 Chunk size: {self.config.chunk_size} characters")
            console.print(f"🔄 Overlap: {self.config.chunk_overlap} characters")
    
    def _step_configure_embeddings(self):
        """Step 4: Configure Embeddings"""
        console.print("\n🧠 [bold blue]Step 4: Configure Embeddings[/bold blue]")
        console.print("-" * 50)
        
        # Check for GPU availability
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                console.print(f"🚀 [green]GPU acceleration available: {gpu_count} GPU(s) detected[/green]")
                use_gpu = Confirm.ask("Use GPU acceleration for embeddings?", default=True)
                if use_gpu:
                    self.config.embedding_device = "cuda"
                    logger.debug(f"[DEBUG] Using GPU acceleration for embeddings: {gpu_count} GPUs")
            else:
                console.print("💻 [yellow]No GPU detected - using CPU for embeddings[/yellow]")
                self.config.embedding_device = "cpu"
        except ImportError:
            console.print("💻 [yellow]PyTorch not available - using CPU for embeddings[/yellow]")
            self.config.embedding_device = "cpu"
        
        # Model selection
        console.print("\n📋 Available embedding models:")
        console.print("  • sentence-transformers/all-MiniLM-L6-v2 (default, fast)")
        console.print("  • sentence-transformers/all-mpnet-base-v2 (better quality)")
        console.print("  • sentence-transformers/all-MiniLM-L12-v2 (balanced)")
        
        model = Prompt.ask(
            "Embedding model",
            default=self.config.embedding_model
        )
        
        self.config.embedding_model = model
        self.config.embedding_dimension = int(Prompt.ask("Embedding dimension", default=str(self.config.embedding_dimension)))
        
        # Batch size configuration
        if self.config.embedding_device == "cuda":
            self.config.embedding_batch_size = int(Prompt.ask("Batch size for GPU processing", default="32"))
        else:
            self.config.embedding_batch_size = int(Prompt.ask("Batch size for CPU processing", default="8"))
        
        logger.debug(f"[DEBUG] Embedding config: model={model}, dim={self.config.embedding_dimension}, device={self.config.embedding_device}, batch_size={self.config.embedding_batch_size}")
        
        console.print(f"\n✅ [green]Embeddings configured: {model}[/green]")
        console.print(f"🔧 Device: {self.config.embedding_device}")
        console.print(f"📏 Dimension: {self.config.embedding_dimension}")
        console.print(f"📦 Batch size: {self.config.embedding_batch_size}")
    
    def _step_configure_vectorstore(self):
        """Step 5: Configure Vector Store"""
        console.print("\n🗄️ [bold blue]Step 5: Configure Vector Store[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n📋 Available vector stores:")
        console.print("  • chroma - ChromaDB (default, local persistence)")
        console.print("  • faiss - Facebook AI Similarity Search (memory-based)")
        
        store_type = Prompt.ask(
            "Vector store type",
            choices=["chroma", "faiss"],
            default=self.config.vector_store
        )
        
        self.config.vector_store = store_type
        logger.debug(f"[DEBUG] Selected vector store: {store_type}")
        
        if store_type == "chroma":
            self.config.collection_name = Prompt.ask("Collection name", default=self.config.collection_name)
            self.config.persist_directory = Prompt.ask("Persist directory", default=self.config.persist_directory)
            
            # Namespace configuration
            self.config.namespace = Prompt.ask("Namespace (for document separation)", default="default")
            
            console.print(f"\n📁 Collection: {self.config.collection_name}")
            console.print(f"💾 Persist directory: {self.config.persist_directory}")
            console.print(f"🏷️ Namespace: {self.config.namespace}")
            
            logger.debug(f"[VectorStore] ChromaDB config: collection={self.config.collection_name}, persist_dir={self.config.persist_directory}, namespace={self.config.namespace}")
        
        console.print(f"\n✅ [green]Vector store configured: {store_type}[/green]")
    
    def _step_configure_llm(self):
        """Step 6: Configure LLM"""
        console.print("\n🤖 [bold blue]Step 6: Configure LLM[/bold blue]")
        console.print("-" * 50)
        
        # Auto-detect available models
        console.print("\n🔍 Auto-detecting available models...")
        
        # Check for local models
        local_models = []
        try:
            import torch
            if torch.cuda.is_available():
                # Check for common local model paths
                model_paths = [
                    "/home/joshua/models",
                    "/home/joshua/humigence/models",
                    Path.home() / "models"
                ]
                
                for model_path in model_paths:
                    if Path(model_path).exists():
                        local_models.extend([f for f in Path(model_path).iterdir() if f.is_dir()])
        except ImportError:
            pass
        
        # Model selection
        console.print("\n📋 Available models:")
        console.print("  • gpt-3.5-turbo (OpenAI API)")
        console.print("  • gpt-4 (OpenAI API)")
        console.print("  • claude-3-sonnet (Anthropic API)")
        console.print("  • claude-3-haiku (Anthropic API)")
        
        if local_models:
            console.print("  • Local models:")
            for model in local_models[:5]:  # Show first 5
                console.print(f"    - {model.name}")
        
        model = Prompt.ask("LLM model", default=self.config.llm_model)
        self.config.llm_model = model
        
        # Model-specific configuration
        if "gpt" in model.lower():
            self.config.llm_provider = "openai"
            self.config.llm_model_type = "chat"
        elif "claude" in model.lower():
            self.config.llm_provider = "anthropic"
            self.config.llm_model_type = "chat"
        else:
            self.config.llm_provider = "local"
            self.config.llm_model_type = "completion"
        
        # Advanced settings
        self.config.max_tokens = int(Prompt.ask("Max tokens", default=str(self.config.max_tokens)))
        self.config.temperature = float(Prompt.ask("Temperature (0.0-1.0)", default=str(self.config.temperature)))
        
        # Device selection for local models
        if self.config.llm_provider == "local":
            try:
                import torch
                if torch.cuda.is_available():
                    self.config.llm_device = Prompt.ask("Device", choices=["cuda", "cpu"], default="cuda")
                else:
                    self.config.llm_device = "cpu"
            except ImportError:
                self.config.llm_device = "cpu"
        else:
            self.config.llm_device = "api"
        
        logger.debug(f"[DEBUG] LLM config: model={model}, provider={self.config.llm_provider}, type={self.config.llm_model_type}, device={self.config.llm_device}")
        
        console.print(f"\n✅ [green]LLM configured: {model}[/green]")
        console.print(f"🔧 Provider: {self.config.llm_provider}")
        console.print(f"📏 Max tokens: {self.config.max_tokens}")
        console.print(f"🌡️ Temperature: {self.config.temperature}")
        if self.config.llm_provider == "local":
            console.print(f"💻 Device: {self.config.llm_device}")
    
    def _step_configure_retrieval(self):
        """Step 7: Configure Retrieval Strategy"""
        console.print("\n🔍 [bold blue]Step 7: Configure Retrieval Strategy[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n📋 Available retrieval strategies:")
        console.print("  • semantic - Pure semantic similarity search")
        console.print("  • hybrid - Combines semantic + keyword search")
        console.print("  • rerank - Semantic search + re-ranking for better precision")
        
        strategy = Prompt.ask(
            "Retrieval strategy",
            choices=["semantic", "hybrid", "rerank"],
            default=self.config.retrieval_strategy
        )
        
        self.config.retrieval_strategy = strategy
        logger.debug(f"[DEBUG] Selected retrieval strategy: {strategy}")
        
        # Common parameters
        self.config.top_k = int(Prompt.ask("Top K results to retrieve", default=str(self.config.top_k)))
        self.config.similarity_threshold = float(Prompt.ask("Similarity threshold (0.0-1.0)", default=str(self.config.similarity_threshold)))
        
        # Strategy-specific parameters
        if strategy == "hybrid":
            self.config.hybrid_alpha = float(Prompt.ask("Hybrid alpha (semantic vs keyword weight)", default="0.7"))
            console.print(f"⚖️ Hybrid alpha: {self.config.hybrid_alpha} (semantic weight)")
        
        elif strategy == "rerank":
            self.config.rerank_top_k = int(Prompt.ask("Initial retrieval count (before re-ranking)", default="20"))
            self.config.rerank_final_k = int(Prompt.ask("Final results after re-ranking", default=str(self.config.top_k)))
            console.print(f"🔄 Re-ranking: {self.config.rerank_top_k} → {self.config.rerank_final_k} results")
        
        console.print(f"\n✅ [green]Retrieval configured: {strategy}[/green]")
        console.print(f"📊 Top K: {self.config.top_k}")
        console.print(f"🎯 Similarity threshold: {self.config.similarity_threshold}")
    
    def _step_test_pipeline(self):
        """Step 8: Test Pipeline"""
        console.print("\n🧪 [bold blue]Step 8: Test Pipeline[/bold blue]")
        console.print("-" * 50)
        
        if not self.pipeline:
            console.print("🔧 [yellow]Initializing pipeline...[/yellow]")
            self.pipeline = RAGPipeline(self.config, llm_client=self.llm_client)
        
        # Initialize LLM if not already done
        if not self.llm_client:
            console.print("🤖 [yellow]Initializing LLM client...[/yellow]")
            llm_config = LLMConfig(
                model_name=self.config.llm_model,
                model_type=self.config.llm_model_type,
                device=self.config.llm_device
            )
            self.llm_client = LLMManager(llm_config)
            self.pipeline.llm_client = self.llm_client
        
        # Test queries
        test_queries = [
            "What is this document about?",
            "What are the main topics discussed?",
            "Can you summarize the key points?"
        ]
        
        console.print("\n🔍 [bold]Running test queries...[/bold]")
        
        for i, question in enumerate(test_queries, 1):
            console.print(f"\n📝 [bold]Test Query {i}:[/bold] {question}")
            console.print("-" * 60)
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing query...", total=None)
                    
                    # Process query
                    progress.update(task, description="🔍 Retrieving relevant documents...")
                    result = self.pipeline.query(question)
                    
                    progress.update(task, description="✅ Query processed")
                
                # Display clean results
                console.print(f"\n✅ [green]Answer:[/green]")
                console.print(Panel(result['answer'], title="Response", border_style="green"))
                
                # Show retrieved chunks info
                console.print(f"\n📊 [blue]Retrieved {result['retrieval_count']} relevant chunks[/blue]")
                
                # Show clean sources
                if result.get('sources'):
                    console.print(f"\n📚 [cyan]Sources:[/cyan]")
                    for j, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
                        console.print(f"   {j}. {source}")
                
                # Debug mode output
                if result.get('debug'):
                    console.print(f"\n🔍 [yellow]Debug Information:[/yellow]")
                    debug = result['debug']
                    console.print(f"   Retrieved chunks: {len(debug['retrieved_chunks'])}")
                    for i, chunk in enumerate(debug['retrieved_chunks'][:2], 1):
                        console.print(f"   Chunk {i}: {chunk['content']} (score: {chunk['score']:.3f})")
                
            except Exception as e:
                console.print(f"❌ [red]Query failed: {e}[/red]")
                logger.error(f"[DEBUG] Test query {i} failed: {e}")
        
        console.print("\n✅ [green]Test validation complete![/green]")
    
    def _launch_interactive_mode(self):
        """Launch interactive query mode."""
        console.print("\n🚀 [bold green]Launching Interactive Query Mode[/bold green]")
        console.print("=" * 60)
        console.print("Enter queries to test your RAG pipeline. Press Enter on an empty line to exit.")
        console.print("=" * 60)
        
        query_count = 0
        
        while True:
            try:
                query = Prompt.ask("\n🔍 Enter a query (or press Enter to quit)")
                
                if not query.strip():
                    console.print("👋 [blue]Exiting interactive mode...[/blue]")
                    break
                
                query_count += 1
                console.print(f"\n🔍 [bold]Query {query_count}: {query}[/bold]")
                console.print("-" * 50)
                
                # Process the query
                try:
                    result = self.pipeline.query(query, k=self.config.top_k)
                    
                    if result.get("answer"):
                        console.print(f"\n✅ [green]Answer:[/green]")
                        console.print(Panel(result['answer'], title="Response", border_style="green"))
                        
                        # Show retrieved chunks info
                        console.print(f"\n📊 [blue]Retrieved {result['retrieval_count']} relevant chunks[/blue]")
                        
                        # Show clean sources
                        if result.get('sources'):
                            console.print(f"\n📚 [cyan]Sources:[/cyan]")
                            for i, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
                                console.print(f"   {i}. {source}")
                        
                        # Debug mode output
                        if result.get('debug'):
                            console.print(f"\n🔍 [yellow]Debug Information:[/yellow]")
                            debug = result['debug']
                            console.print(f"   Retrieved chunks: {len(debug['retrieved_chunks'])}")
                            for i, chunk in enumerate(debug['retrieved_chunks'][:2], 1):
                                console.print(f"   Chunk {i}: {chunk['content']} (score: {chunk['score']:.3f})")
                    else:
                        console.print("❌ [red]No answer generated[/red]")
                
                except Exception as e:
                    console.print(f"❌ [red]Query failed: {e}[/red]")
                    console.print("💡 [yellow]Try rephrasing your question or check the pipeline configuration[/yellow]")
            
            except KeyboardInterrupt:
                console.print("\n👋 [blue]Exiting interactive mode...[/blue]")
                break
            except Exception as e:
                console.print(f"❌ [red]Unexpected error: {e}[/red]")
                break
        
        if query_count > 0:
            console.print(f"\n✅ [green]Interactive session completed! Processed {query_count} queries.[/green]")
        else:
            console.print("\n👋 [blue]No queries processed.[/blue]")
    
    def _display_ingestion_summary(self, results: List[Dict], total_chunks: int):
        """Display ingestion summary table."""
        console.print("\n📊 [bold]Ingestion Report[/bold]")
        
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Chunks", justify="right")
        table.add_column("Pages", justify="right")
        table.add_column("Error", style="red")
        
        successful = 0
        failed = 0
        
        for result in results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            chunks = str(result.get("chunks", 0)) if result["status"] == "success" else "-"
            pages = str(result.get("pages", "-")) if result["status"] == "success" else "-"
            error = result.get("error", "") if result["status"] == "failed" else ""
            
            table.add_row(
                result["file"],
                f"{status_icon} {result['status'].title()}",
                chunks,
                pages,
                error
            )
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
        
        console.print(table)
        console.print(f"\n[bold]Total: {successful} successful, {failed} failed[/bold]")
        console.print(f"[bold]Total chunks ingested: {total_chunks}[/bold]")
        
        # Debug logging for ingestion report
        logger.debug(f"[INGEST] Ingestion complete: {successful} successful, {failed} failed, {total_chunks} total chunks")

def main():
    """Main entry point for the RAG wizard."""
    wizard = RAGWizard()
    wizard.run()

if __name__ == "__main__":
    main()
