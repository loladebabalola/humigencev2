# cli/main.py

import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import from pipelines
sys.path.insert(0, str(Path(__file__).parent.parent))

# DO NOT import Unsloth here - delay until after wizard completion
UNSLOTH_AVAILABLE = None  # Will be checked later

from cli.config_wizard import collect_training_config
from cli.atomic_eval import app as atomic_eval_app
from cli.rag_wizard_restored import RAGWizardRestored
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import inquirer
import json
import typer

console = Console()

# Removed download functionality - system now only works with local datasets

def check_unsloth_availability():
    """Check if Unsloth is available (delayed import)"""
    global UNSLOTH_AVAILABLE
    if UNSLOTH_AVAILABLE is None:
        try:
            import unsloth
            UNSLOTH_AVAILABLE = True
        except ImportError:
            UNSLOTH_AVAILABLE = False
    return UNSLOTH_AVAILABLE

def detect_gpus():
    """Detect available GPUs"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpus = []
            for i in range(gpu_count):
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB"
                })
            return gpu_count, gpus
        else:
            return 0, []
    except ImportError:
        return 0, []

def choose_training_mode(gpu_count, gpus):
    """Choose training mode based on available GPUs"""
    if gpu_count == 0:
        console.print("[yellow]⚠️ No GPUs detected - CPU training not supported[/yellow]")
        return None
    elif gpu_count == 1:
        console.print(f"[blue]🔧 Single GPU detected - using GPU 0: {gpus[0]['name']}[/blue]")
        return "single"
    else:
        # Multiple GPUs - prompt user to choose
        console.print(f"[blue]🔧 {gpu_count} GPUs detected - choose training mode[/blue]")
        
        # Display available GPUs
        from rich.table import Table
        gpu_table = Table(show_header=True, box=None)
        gpu_table.add_column("Index", style="cyan", width=6)
        gpu_table.add_column("Name", style="white", width=40)
        gpu_table.add_column("VRAM", style="green", width=10)
        
        for gpu in gpus:
            gpu_table.add_row(str(gpu['index']), gpu['name'], gpu['memory'])
        
        console.print(gpu_table)
        
        choices = [
            "Multi-GPU Training (all available GPUs)",
            "Single GPU Training (choose specific GPU)"
        ]
        
        questions = [
            inquirer.List('training_mode',
                         message="🔧 Training Mode: (Use arrow keys)",
                         choices=choices,
                         default=choices[0])
        ]
        
        answers = inquirer.prompt(questions)
        selected_mode = answers['training_mode']
        
        if "Multi-GPU" in selected_mode:
            return "multi"
        else:
            # Single GPU - let user choose which one
            gpu_choices = []
            for gpu in gpus:
                gpu_choices.append(f"GPU{gpu['index']}: {gpu['name']} ({gpu['memory']})")
            
            questions = [
                inquirer.List('gpu_selection',
                             message="Choose GPU: (Use arrow keys)",
                             choices=gpu_choices,
                             default=gpu_choices[0])
            ]
            
            answers = inquirer.prompt(questions)
            selected_gpu = answers['gpu_selection']
            
            # Extract GPU index
            gpu_index = int(selected_gpu.split("GPU")[1].split(":")[0])
            console.print(f"[blue]Selected GPU {gpu_index}: {gpus[gpu_index]['name']}[/blue]")
            return f"single_{gpu_index}"

def show_menu():
    console.rule("[bold cyan]Humigence — Your AI. Your pipeline. Zero code.")
    print("[dim]A complete MLOps suite built for makers, teams, and enterprises.[/dim]\n")
    print("Options:")
    print("[bold green]1.[/bold green] Supervised Fine-Tuning 🚀")
    print("[bold green]2.[/bold green] RAG Implementation 🔍")
    print("[bold green]3.[/bold green] Query RAG Profile 💬")
    print("[bold yellow]4.[/bold yellow] EnterpriseGPT (coming soon)")
    print("[bold yellow]5.[/bold yellow] Batch Inference (coming soon)")
    print("[bold yellow]6.[/bold yellow] Context Length (coming soon)")
    print("[bold red]7.[/bold red] Exit\n")

def launch_training(config, training_mode, gpus):
    """Launch training based on the selected mode"""
    import os
    import subprocess
    import json
    
    # Change to the humigence directory
    humigence_dir = Path(__file__).parent.parent
    os.chdir(humigence_dir)
    
    # Map model names to Unsloth equivalents
    model_mapping = {
        "Qwen/Qwen2.5-0.5B": "unsloth/Qwen2.5-0.5B-Instruct",
        "microsoft/Phi-2": "unsloth/Phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "unsloth/TinyLlama-1.1B-Chat-v1.0"
    }
    
    # Use Unsloth model if available, otherwise use original
    base_model = config.get("base_model", config.get("model_name", "Qwen/Qwen2.5-0.5B"))
    model_name = model_mapping.get(base_model, base_model)
    
    # Determine dataset parameters
    dataset_path = config["dataset_path"]
    if dataset_path.startswith("local:"):
        # Local dataset - use as custom dataset
        dataset_name = "jsonl"
        dataset_config = dataset_path[6:]  # Remove "local:" prefix
    else:
        # Default to wikitext for demo
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"
    
    # Map training recipe to precision
    training_recipe = config.get("training_recipe", "QLoRA (4-bit NF4)")
    if "QLoRA" in training_recipe:
        precision = "qlora_4bit"
    elif "BF16" in training_recipe:
        precision = "lora_bf16"
    else:
        precision = "lora_fp16"
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if training_mode == "multi":
        # Multi-GPU training with TorchRun
        output_dir = f"./runs/humigence/out_lora_dual_{timestamp}"
        console.print("[bold green]🚀 Launching multi-GPU training with Unsloth...[/bold green]")
        
        cmd = [
            "torchrun", 
            "--nproc_per_node=2", 
            "training/unsloth/train_lora_dual.py",
            "--model", model_name,
            "--dataset", dataset_name,
            "--dataset_config", dataset_config,
            "--out_dir", output_dir,
            "--max_steps", "1000",
            "--per_device_batch", "2",
            "--grad_accum", "4",
            "--learning_rate", "2e-4",
            "--block_size", "1024",
            "--lora_r", "16",
            "--lora_alpha", "32",
            "--lora_dropout", "0.0",
            "--precision", precision
        ]
        
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=humigence_dir)
            console.print("[bold green]✅ Multi-GPU training completed successfully![/bold green]")
            console.print(f"[blue]📁 Output saved to: {output_dir}[/blue]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ Multi-GPU training failed with return code: {e.returncode}[/bold red]")
            console.print("[yellow]🔄 Falling back to single-GPU training...[/yellow]")
            # Fall through to single-GPU fallback
            training_mode = "single"
    
    if training_mode == "single" or training_mode.startswith("single_"):
        # Single-GPU training
        if training_mode.startswith("single_"):
            gpu_index = int(training_mode.split("_")[1])
            output_dir = f"./runs/humigence/out_lora_single_{timestamp}_gpu{gpu_index}"
        else:
            gpu_index = 0
            output_dir = f"./runs/humigence/out_lora_single_{timestamp}"
        
        console.print(f"[bold green]🚀 Launching single-GPU training with Unsloth...[/bold green]")
        console.print(f"[blue]Using GPU {gpu_index}: {gpus[gpu_index]['name']}[/blue]")
        
        cmd = [
            "python3",
            "training/unsloth/train_lora_dual.py",
            "--model", model_name,
            "--dataset", dataset_name,
            "--dataset_config", dataset_config,
            "--out_dir", output_dir,
            "--max_steps", "1000",
            "--per_device_batch", "4",  # Larger batch for single GPU
            "--grad_accum", "2",        # Less accumulation for single GPU
            "--learning_rate", "2e-4",
            "--block_size", "1024",
            "--lora_r", "16",
            "--lora_alpha", "32",
            "--lora_dropout", "0.0",
            "--precision", precision
        ]
        
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        
        # Set environment for specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        
        try:
            result = subprocess.run(cmd, check=True, cwd=humigence_dir, env=env)
            console.print("[bold green]✅ Single-GPU training completed successfully![/bold green]")
            console.print(f"[blue]📁 Output saved to: {output_dir}[/blue]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ Single-GPU training failed with return code: {e.returncode}[/bold red]")
            return False
        except Exception as e:
            console.print(f"[bold red]❌ Single-GPU training failed: {e}[/bold red]")
            return False
    
    return False

def query_rag_profile(profile_name: str):
    """Query a saved RAG profile interactively."""
    console.print(f"\n🔍 [bold blue]Loading RAG Profile: {profile_name}[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Load the profile
        profile_dir = Path.home() / ".humigence" / "rag_profiles"
        profile_path = profile_dir / f"{profile_name}.json"
        
        if not profile_path.exists():
            console.print(f"❌ [red]Profile not found: {profile_path}[/red]")
            console.print("Available profiles:")
            if profile_dir.exists():
                profiles = list(profile_dir.glob("*.json"))
                if profiles:
                    for profile in profiles:
                        console.print(f"  • {profile.stem}")
                else:
                    console.print("  No profiles found")
            else:
                console.print("  No profiles directory found")
            return False
        
        # Load profile configuration
        with open(profile_path, 'r') as f:
            config_dict = json.load(f)
        
        console.print(f"✅ [green]Profile loaded: {config_dict.get('profile_name', profile_name)}[/green]")
        console.print(f"📊 Components: {config_dict.get('components', {})}")
        
        # Initialize RAG pipeline
        from rag.pipeline import RAGPipeline
        from rag.config import RAGConfig
        
        # Create config from profile
        config = RAGConfig(**{k: v for k, v in config_dict.items() 
                             if k not in ["profile_name", "created_at", "version", "pipeline_type", "components"]})
        
        # Initialize pipeline
        console.print("\n🔧 [bold]Initializing RAG pipeline...[/bold]")
        pipeline = RAGPipeline(config)
        
        # Load the profile into the pipeline
        load_result = pipeline.load_profile(str(profile_path))
        if load_result["status"] != "success":
            console.print(f"❌ [red]Failed to load profile: {load_result['error']}[/red]")
            return False
        
        console.print("✅ [green]Pipeline initialized successfully![/green]")
        
        # Interactive query loop
        console.print("\n🔍 [bold]Interactive Query Mode[/bold]")
        console.print("Enter queries to test your RAG pipeline. Press Enter on an empty line to exit.")
        console.print("=" * 60)
        
        query_count = 0
        
        while True:
            try:
                query = Prompt.ask("\nEnter a query (or press Enter to quit)")
                
                if not query.strip():
                    console.print("👋 [blue]Exiting interactive mode...[/blue]")
                    break
                
                query_count += 1
                console.print(f"\n🔍 [bold]Query {query_count}: {query}[/bold]")
                console.print("-" * 50)
                
                # Process the query
                try:
                    result = pipeline.query(query, k=config.top_k)
                    
                    if result.get("answer"):
                        console.print(f"\n✅ [green]Answer:[/green]")
                        console.print(Panel(result['answer'], title="Response", border_style="green"))
                        
                        # Show retrieved chunks info
                        console.print(f"\n📊 [blue]Retrieved {result['retrieved_chunks']} chunks in {result['timing']['total_time']:.2f}s[/blue]")
                        
                        # Show sources if available
                        if result.get('sources'):
                            console.print(f"\n📚 [cyan]Sources:[/cyan]")
                            for i, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
                                score = source.get('score', 0)
                                display = source.get('display', f"Chunk {source.get('chunk_id', 'N/A')}")
                                console.print(f"   {i}. {display} – score: {score:.3f}")
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
        
        return True
        
    except Exception as e:
        console.print(f"❌ [red]Error loading profile: {e}[/red]")
        return False

def main():
    while True:
        show_menu()
        choice = console.input("[bold blue]Select an option[/bold blue]: ")

        if choice == "1":
            console.print("[bold green]Starting Supervised Fine-Tuning...[/bold green]")
            
            # Step 1: Run the configuration wizard (no Unsloth import yet)
            config_path = collect_training_config()
            
            if config_path is None:
                # User cancelled or error occurred
                console.print("[bold red]❌ Training cancelled. Returning to main menu.[/bold red]")
                time.sleep(2)
                continue
            
            # Step 2: Load the configuration from the wizard
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Step 3: NOW check if Unsloth dependencies are available (after wizard completion)
            if not check_unsloth_availability():
                console.print("[bold red]❌ Missing required dependencies: No module named 'unsloth'[/bold red]")
                console.print("[yellow]➡ To install, run:[/yellow]")
                console.print("[cyan]python3 training/unsloth/setup_humigence_unsloth.py[/cyan]")
                time.sleep(2)
                continue
            
            # Step 4: Detect GPUs BEFORE importing Unsloth (to avoid interference)
            gpu_count, gpus = detect_gpus()
            training_mode = choose_training_mode(gpu_count, gpus)
            
            if training_mode is None:
                console.print("[bold red]❌ No suitable training mode available. Returning to main menu.[/bold red]")
                time.sleep(2)
                continue
            
            # Step 5: Launch training
            success = launch_training(config, training_mode, gpus)
            
            if not success:
                console.print("[bold red]❌ Training failed. Check the logs above for details.[/bold red]")
            
            # Ask if user wants to start another training session
            console.print("\n[bold cyan]Training completed![/bold cyan]")
            if console.input("[bold blue]Start another training session? (y/N)[/bold blue]: ").lower() in ['y', 'yes']:
                continue
            else:
                break
        elif choice == "2":
            console.print("[bold green]Starting RAG Implementation Wizard...[/bold green]")
            
            try:
                # Run the restored RAG wizard
                wizard = RAGWizardRestored()
                wizard.run()
                
                console.print("\n[bold green]✅ RAG pipeline setup completed successfully![/bold green]")
                
            except KeyboardInterrupt:
                console.print("\n[bold red]❌ RAG setup cancelled by user.[/bold red]")
            except Exception as e:
                console.print(f"\n[bold red]❌ RAG setup failed: {e}[/bold red]")
            
            # Ask if user wants to configure another pipeline
            console.print("\n[bold cyan]RAG setup completed![/bold cyan]")
            if console.input("[bold blue]Configure another RAG pipeline? (y/N)[/bold blue]: ").lower() in ['y', 'yes']:
                continue
            else:
                break
        elif choice == "3":
            console.print("[bold green]Starting RAG Profile Query...[/bold green]")
            
            # List available profiles
            profile_dir = Path.home() / ".humigence" / "rag_profiles"
            if profile_dir.exists():
                profiles = list(profile_dir.glob("*.json"))
                if profiles:
                    console.print("\n📚 [bold]Available RAG Profiles:[/bold]")
                    for i, profile in enumerate(profiles, 1):
                        console.print(f"  {i}. {profile.stem}")
                    
                    try:
                        profile_choice = int(console.input("\n[bold blue]Select profile number[/bold blue]: "))
                        if 1 <= profile_choice <= len(profiles):
                            selected_profile = profiles[profile_choice - 1].stem
                            success = query_rag_profile(selected_profile)
                            
                            if success:
                                console.print("\n[bold green]✅ Query session completed![/bold green]")
                            else:
                                console.print("\n[bold red]❌ Query session failed![/bold red]")
                        else:
                            console.print("[bold red]❌ Invalid profile selection![/bold red]")
                    except ValueError:
                        console.print("[bold red]❌ Please enter a valid number![/bold red]")
                else:
                    console.print("[bold red]❌ No RAG profiles found![/bold red]")
                    console.print("Please create a RAG pipeline first using option 2.")
            else:
                console.print("[bold red]❌ No RAG profiles directory found![/bold red]")
                console.print("Please create a RAG pipeline first using option 2.")
            
            # Ask if user wants to query another profile
            console.print("\n[bold cyan]Query session completed![/bold cyan]")
            if console.input("[bold blue]Query another RAG profile? (y/N)[/bold blue]: ").lower() in ['y', 'yes']:
                continue
            else:
                break
        elif choice == "7":
            console.print("[bold red]Exiting Humigence CLI. Goodbye![/bold red]")
            time.sleep(1)
            sys.exit()
        else:
            console.print("[yellow]⚠️ Option not implemented yet. Try 1, 2, 3, or 7.[/yellow]\n")
            time.sleep(1)

if __name__ == "__main__":
    main()