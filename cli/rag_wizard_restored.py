"""
RAG Implementation Wizard CLI - Restored Version.

Interactive wizard for setting up and using RAG pipelines with the original
user-friendly step-by-step flow while preserving recent ingestion/metadata fixes.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class RAGWizardRestored:
    """Restored Interactive RAG Implementation Wizard with original flow."""
    
    def __init__(self):
        self.config = RAGConfig()
        self.pipeline = None
        self.llm_client = None
        self.documents = []
        self.document_sources = []
        self.ingestion_report = []
        self.chunking_strategy = "sentence"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.vector_store_type = "chroma"
        self.llm_model = "microsoft/Phi-2"
        self.retrieval_strategy = "semantic"
    
    def run(self):
        """Run the restored RAG Implementation Wizard."""
        self._show_welcome_header()
        
        # Step-by-step wizard flow
        self._step_1_document_selection()
        self._step_2_chunking_selection()
        self._step_3_embedding_selection()
        self._step_4_vectorstore_setup()
        self._step_5_llm_selection()
        self._step_6_retrieval_selection()
        self._step_7_test_validation()
        
        # Launch interactive query mode
        self._launch_interactive_mode()
    
    def _show_welcome_header(self):
        """Display the welcome header."""
        console.print("\n" + "="*80)
        console.print("🤖 RAG Implementation Wizard")
        console.print("Set up and use Retrieval-Augmented Generation pipelines")
        console.print("="*80)
    
    def _step_1_document_selection(self):
        """Step 1: Document Source Selection"""
        console.print("\n📚 [bold blue]Step 1: Document Source Selection[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n📋 Available document sources:")
        console.print("  1. Enter text directly")
        console.print("  2. Load from text file")
        console.print("  3. Load from PDF file")
        console.print("  4. Load from URL")
        console.print("  5. Load multiple documents")
        console.print("  6. Load from default docs folder (~/humigence/docs/)")
        
        choice = Prompt.ask("Select document source", default="6")
        try:
            choice = int(choice)
        except ValueError:
            console.print("❌ [red]Invalid selection. Please enter a number between 1-6.[/red]")
            return
        
        if choice == 1:
            self._load_text_directly()
        elif choice == 2:
            self._load_text_file()
        elif choice == 3:
            self._load_single_pdf()
        elif choice == 4:
            self._load_from_url()
        elif choice == 5:
            self._load_multiple_documents()
        elif choice == 6:
            self._load_from_default_folder()
        
        console.print(f"\n✅ [green]Documents loaded: {len(self.documents)} document(s)[/green]")
        if self.ingestion_report:
            total_chunks = sum(r.get('chunks', 0) for r in self.ingestion_report if r.get('status') == 'success')
            console.print(f"📊 Total chunks created: {total_chunks}")
    
    def _load_text_directly(self):
        """Load text directly from user input."""
        text = Prompt.ask("Enter document text")
        if text:
            self.documents.append(text)
            self.document_sources.append({
                "type": "text", 
                "content": text[:100] + "..." if len(text) > 100 else text
            })
            console.print("✅ [green]Text document added[/green]")
    
    def _load_text_file(self):
        """Load text from a file."""
        file_path = Prompt.ask("Enter text file path")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.documents.append(text)
                self.document_sources.append({
                    "type": "text_file",
                    "filename": os.path.basename(file_path),
                    "path": file_path
                })
                console.print(f"✅ [green]Text file loaded: {os.path.basename(file_path)}[/green]")
            except Exception as e:
                console.print(f"❌ [red]Failed to load text file: {e}[/red]")
        else:
            console.print("❌ [red]Text file not found[/red]")
    
    def _load_single_pdf(self):
        """Load a single PDF file."""
        pdf_path = Prompt.ask("Enter PDF file path")
        if os.path.exists(pdf_path):
            self._process_pdf_files([Path(pdf_path)])
        else:
            console.print("❌ [red]PDF file not found[/red]")
    
    def _load_from_url(self):
        """Load document from URL (placeholder)."""
        url = Prompt.ask("Enter document URL")
        console.print("⚠️ [yellow]URL loading not implemented yet[/yellow]")
        console.print("Please use one of the other options for now.")
    
    def _load_multiple_documents(self):
        """Load multiple documents."""
        console.print("\n📁 Multiple Document Loading")
        console.print("Enter file paths (one per line, empty line to finish):")
        
        file_paths = []
        while True:
            path = Prompt.ask("File path (or press Enter to finish)", default="")
            if not path:
                break
            if os.path.exists(path):
                file_paths.append(Path(path))
            else:
                console.print(f"⚠️ [yellow]File not found: {path}[/yellow]")
        
        if file_paths:
            pdf_files = [f for f in file_paths if f.suffix.lower() == '.pdf']
            text_files = [f for f in file_paths if f.suffix.lower() in ['.txt', '.md']]
            
            if pdf_files:
                self._process_pdf_files(pdf_files)
            
            if text_files:
                for text_file in text_files:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                        self.documents.append(text)
                        self.document_sources.append({
                            "type": "text_file",
                            "filename": text_file.name,
                            "path": str(text_file)
                        })
                        console.print(f"✅ [green]Text file loaded: {text_file.name}[/green]")
                    except Exception as e:
                        console.print(f"❌ [red]Failed to load {text_file.name}: {e}[/red]")
    
    def _load_from_default_folder(self):
        """Load from default docs folder."""
        docs_folder = Path.home() / "humigence" / "docs"
        
        if not docs_folder.exists():
            console.print(f"❌ [red]Default docs folder not found: {docs_folder}[/red]")
            console.print("Please create the folder and add PDF files, or use another option.")
            return
        
        pdf_files = list(docs_folder.glob("*.pdf"))
        if not pdf_files:
            console.print(f"❌ [red]No PDF files found in {docs_folder}[/red]")
            console.print("Please add PDF files to the folder, or use another option.")
            return
        
        console.print(f"🔍 Found {len(pdf_files)} PDF(s) in default docs folder:")
        for i, pdf_file in enumerate(pdf_files, 1):
            file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
            console.print(f"  {i}. {pdf_file.name} ({file_size:.1f} MB)")
        
        console.print("\n📋 Selection options:")
        console.print("  • Enter 'all' to load all PDFs")
        console.print("  • Enter specific numbers (e.g., '1,3,5') to load selected PDFs")
        console.print("  • Enter '1-3' to load a range of PDFs")
        
        selection = Prompt.ask("Select PDFs to load", default="all")
        
        selected_files = self._parse_file_selection(selection, pdf_files)
        
        if selected_files:
            self._process_pdf_files(selected_files)
        else:
            console.print("❌ [red]No valid files selected[/red]")
    
    def _parse_file_selection(self, selection: str, files: List[Path]) -> List[Path]:
        """Parse file selection string."""
        selected_files = []
        try:
            if selection.lower() == "all":
                selected_files = files
            elif '-' in selection:
                # Range selection (e.g., "1-3")
                start, end = map(int, selection.split('-'))
                selected_files = files[start-1:end]
            else:
                # Comma-separated selection (e.g., "1,3,5")
                indices = [int(x.strip()) for x in selection.split(',')]
                selected_files = [files[i-1] for i in indices if 1 <= i <= len(files)]
        except (ValueError, IndexError) as e:
            console.print(f"❌ [red]Invalid selection format: {e}[/red]")
            return []
        
        return selected_files
    
    def _process_pdf_files(self, pdf_files: List[Path]):
        """Process PDF files for ingestion with proper metadata."""
        console.print(f"\n🚀 Processing {len(pdf_files)} PDF(s)...")
        
        # Initialize pipeline if not already done
        if not self.pipeline:
            self.pipeline = RAGPipeline(self.config, llm_client=self.llm_client)
        
        # Use the multi-PDF ingestion method
        pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
        result = self.pipeline.ingest_multiple_pdfs(pdf_paths, chunking_strategy=self.chunking_strategy)
        
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
    
    def _step_2_chunking_selection(self):
        """Step 2: Chunking Strategy Selection"""
        console.print("\n🔧 [bold blue]Step 2: Chunking Strategy Selection[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n📋 Available chunking strategies:")
        console.print("  1. Sentence-based (default, recommended)")
        console.print("  2. Fixed-window")
        console.print("  3. Semantic with Phi-2")
        
        choice = Prompt.ask("Select chunking strategy", default="1")
        try:
            choice = int(choice)
        except ValueError:
            console.print("❌ [red]Invalid selection. Please enter a number between 1-3.[/red]")
            return
        
        if choice == 1:
            self.chunking_strategy = "sentence"
            self.config.chunking_strategy = "sentence"
            console.print("✅ [green]Chunking configured: sentence-based[/green]")
        elif choice == 2:
            self.chunking_strategy = "fixed"
            self.config.chunking_strategy = "fixed"
            chunk_size = Prompt.ask("Chunk size (characters)", default="512")
            chunk_overlap = Prompt.ask("Chunk overlap (characters)", default="100")
            try:
                self.config.chunk_size = int(chunk_size)
                self.config.chunk_overlap = int(chunk_overlap)
            except ValueError:
                console.print("❌ [red]Invalid chunk size/overlap. Using defaults.[/red]")
                self.config.chunk_size = 512
                self.config.chunk_overlap = 100
            console.print(f"✅ [green]Chunking configured: fixed-window (size: {self.config.chunk_size}, overlap: {self.config.chunk_overlap})[/green]")
        elif choice == 3:
            self.chunking_strategy = "semantic"
            self.config.chunking_strategy = "semantic"
            console.print("⚠️ [yellow]Semantic chunking is experimental and may be slower[/yellow]")
            console.print("✅ [green]Chunking configured: semantic with Phi-2[/green]")
    
    def _step_3_embedding_selection(self):
        """Step 3: Embedding Model Selection"""
        console.print("\n🧠 [bold blue]Step 3: Embedding Model Selection[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n📋 Available embedding models:")
        console.print("  1. Fast, lightweight (384-dim) - all-MiniLM-L6-v2")
        console.print("  2. High quality (768-dim) - all-mpnet-base-v2")
        console.print("  3. Multilingual (384-dim) - paraphrase-multilingual-MiniLM-L12-v2")
        console.print("  4. Enter custom model")
        
        choice = Prompt.ask("Select embedding model", default="1")
        try:
            choice = int(choice)
        except ValueError:
            console.print("❌ [red]Invalid selection. Please enter a number between 1-4.[/red]")
            return
        
        if choice == 1:
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.config.embedding_model = self.embedding_model
            self.config.embedding_dimension = 384
        elif choice == 2:
            self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
            self.config.embedding_model = self.embedding_model
            self.config.embedding_dimension = 768
        elif choice == 3:
            self.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.config.embedding_model = self.embedding_model
            self.config.embedding_dimension = 384
        elif choice == 4:
            self.embedding_model = Prompt.ask("Enter custom model name")
            self.config.embedding_model = self.embedding_model
            dim_input = Prompt.ask("Enter embedding dimension", default="384")
            try:
                self.config.embedding_dimension = int(dim_input)
            except ValueError:
                console.print("❌ [red]Invalid dimension. Using default 384.[/red]")
                self.config.embedding_dimension = 384
        
        # GPU acceleration
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                console.print(f"🚀 [green]GPU acceleration available: {gpu_count} GPU(s) detected[/green]")
                gpu_input = Prompt.ask("Enable multi-GPU acceleration? [y/n]", default="y")
                use_gpu = gpu_input.lower() in ['y', 'yes', 'true', '1']
                if use_gpu:
                    self.config.embedding_device = "cuda"
                    self.config.multi_gpu_embeddings = True
                else:
                    self.config.embedding_device = "cpu"
                    self.config.multi_gpu_embeddings = False
            else:
                console.print("💻 [yellow]No GPU detected - using CPU for embeddings[/yellow]")
                self.config.embedding_device = "cpu"
                self.config.multi_gpu_embeddings = False
        except ImportError:
            console.print("💻 [yellow]PyTorch not available - using CPU for embeddings[/yellow]")
            self.config.embedding_device = "cpu"
            self.config.multi_gpu_embeddings = False
        
        console.print(f"✅ [green]Embeddings configured: {self.embedding_model}[/green]")
        console.print(f"🔧 Device: {self.config.embedding_device}")
        console.print(f"📏 Dimension: {self.config.embedding_dimension}")
        if self.config.multi_gpu_embeddings:
            console.print("🚀 Multi-GPU acceleration: enabled")
    
    def _step_4_vectorstore_setup(self):
        """Step 4: Vector Store Setup"""
        console.print("\n🗄️ [bold blue]Step 4: Vector Store Setup[/bold blue]")
        console.print("-" * 50)
        
        # Use silent defaults for better UX
        self.vector_store_type = "chroma"
        self.config.vector_store = "chroma"
        self.config.collection_name = "humigence_rag"
        self.config.persist_directory = "./chroma_db"
        self.config.namespace = "default"
        
        console.print("✅ [green]Vector store configured: chroma (collection=humigence_rag, namespace=default)[/green]")
    
    def _step_5_llm_selection(self):
        """Step 5: LLM Model Selection"""
        console.print("\n🤖 [bold blue]Step 5: LLM Model Selection[/bold blue]")
        console.print("-" * 50)
        
        console.print("\n🔍 Auto-detecting available models...")
        
        # Check for local models
        local_models = []
        try:
            import torch
            if torch.cuda.is_available():
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
        
        console.print("\n📋 Available models:")
        console.print("  1. microsoft/Phi-2 (local, recommended)")
        console.print("  2. microsoft/DialoGPT-medium (local)")
        console.print("  3. gpt-3.5-turbo (OpenAI API)")
        console.print("  4. gpt-4 (OpenAI API)")
        console.print("  5. claude-3-sonnet (Anthropic API)")
        console.print("  6. Enter custom model")
        
        if local_models:
            console.print("  Local models found:")
            for i, model in enumerate(local_models[:5], 7):
                console.print(f"  {i}. {model.name} (local)")
        
        choice = Prompt.ask("Select LLM model", default="1")
        try:
            choice = int(choice)
        except ValueError:
            console.print("❌ [red]Invalid selection. Please enter a number.[/red]")
            return
        
        if choice == 1:
            self.llm_model = "microsoft/Phi-2"
            self.config.llm_model = self.llm_model
            self.config.llm_model_type = "local"
        elif choice == 2:
            self.llm_model = "microsoft/DialoGPT-medium"
            self.config.llm_model = self.llm_model
            self.config.llm_model_type = "local"
        elif choice == 3:
            self.llm_model = "gpt-3.5-turbo"
            self.config.llm_model = self.llm_model
            self.config.llm_model_type = "api"
            self.config.llm_provider = "openai"
        elif choice == 4:
            self.llm_model = "gpt-4"
            self.config.llm_model = self.llm_model
            self.config.llm_model_type = "api"
            self.config.llm_provider = "openai"
        elif choice == 5:
            self.llm_model = "claude-3-sonnet"
            self.config.llm_model = self.llm_model
            self.config.llm_model_type = "api"
            self.config.llm_provider = "anthropic"
        elif choice == 6:
            self.llm_model = Prompt.ask("Enter custom model name")
            self.config.llm_model = self.llm_model
            self.config.llm_model_type = "local"
        elif choice >= 7 and local_models:
            # Local model selection
            model_index = choice - 7
            if model_index < len(local_models):
                self.llm_model = str(local_models[model_index])
                self.config.llm_model = self.llm_model
                self.config.llm_model_type = "local"
        
        # Device selection for local models - use silent defaults
        if self.config.llm_model_type == "local":
            try:
                import torch
                if torch.cuda.is_available():
                    self.config.llm_device = "cuda"
                else:
                    self.config.llm_device = "cpu"
            except ImportError:
                self.config.llm_device = "cpu"
        else:
            self.config.llm_device = "api"
        
        console.print(f"✅ [green]LLM configured: {self.llm_model}[/green]")
        console.print(f"🔧 Type: {self.config.llm_model_type}")
        if self.config.llm_model_type == "local":
            console.print(f"💻 Device: {self.config.llm_device}")
    
    def _step_6_retrieval_selection(self):
        """Step 6: Retrieval Strategy Selection"""
        console.print("\n🔍 [bold blue]Step 6: Retrieval Strategy Selection[/bold blue]")
        console.print("-" * 50)
        
        # Use semantic search as default for better UX
        self.retrieval_strategy = "semantic"
        self.config.retrieval_strategy = "semantic"
        
        # Use silent defaults for better UX
        self.config.top_k = 5
        self.config.similarity_threshold = 0.25
        
        console.print(f"✅ [green]Retrieval configured: {self.retrieval_strategy} (top-k=5, threshold=0.25)[/green]")
    
    def _step_7_test_validation(self):
        """Step 7: Test Query & Validation"""
        console.print("\n🧪 [bold blue]Step 7: Test Query & Validation[/bold blue]")
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
            "What is the main topic of these documents?",
            "What are the key points discussed?",
            "Can you summarize the content?"
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
                
                # Display results
                if result.get('answer') and result['answer'] != "I cannot answer based on the provided documents.":
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
                    if result.get('debug') and os.environ.get("RAG_DEBUG", "0") == "1":
                        console.print(f"\n🔍 [yellow]Debug Information:[/yellow]")
                        debug = result['debug']
                        console.print(f"   Retrieved chunks: {len(debug['retrieved_chunks'])}")
                        for i, chunk in enumerate(debug['retrieved_chunks'][:2], 1):
                            console.print(f"   Chunk {i}: {chunk['content']} (score: {chunk['score']:.3f})")
                else:
                    console.print("⚠️ [yellow]No relevant information found.[/yellow]")
                
            except Exception as e:
                console.print(f"❌ [red]Query failed: {e}[/red]")
                logger.error(f"Test query {i} failed: {e}")
        
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
                    
                    if result.get("answer") and result['answer'] != "I cannot answer based on the provided documents.":
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
                        if result.get('debug') and os.environ.get("RAG_DEBUG", "0") == "1":
                            console.print(f"\n🔍 [yellow]Debug Information:[/yellow]")
                            debug = result['debug']
                            console.print(f"   Retrieved chunks: {len(debug['retrieved_chunks'])}")
                            for i, chunk in enumerate(debug['retrieved_chunks'][:2], 1):
                                console.print(f"   Chunk {i}: {chunk['content']} (score: {chunk['score']:.3f})")
                    else:
                        console.print("⚠️ [yellow]No relevant information found.[/yellow]")
                
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

def main():
    """Main entry point for the restored RAG wizard."""
    wizard = RAGWizardRestored()
    wizard.run()

if __name__ == "__main__":
    main()
