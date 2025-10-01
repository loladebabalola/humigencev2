# cli/config_wizard.py

import json
import time
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich import print
import sys
import os
import inquirer

# Add the parent directory to the path so we can import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.device import get_system_info
from utils.dataset_loader import list_local_datasets
from humigence_datasets.download_datasets import get_available_datasets, ensure_demo_dataset

console = Console()

def show_system_info():
    """Display system information as shown in the screenshots"""
    console.print("\n[bold cyan]System Information[/bold cyan]")
    info = get_system_info()
    
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", width=15)
    table.add_column(style="white")
    
    table.add_row("Platform:", info["Platform"])
    table.add_row("Python:", info["Python Version"])
    table.add_row("PyTorch:", info["Torch Version"])
    table.add_row("CUDA:", info["CUDA Version"] if info["CUDA Available"] else "Not available")
    table.add_row("RAM:", info["RAM"])
    table.add_row("CPUs:", str(info["CPUs"]))
    table.add_row("GPUs detected:", str(info["GPU Count"]))
    
    console.print(table)

def choose_setup_mode():
    """Choose between Basic and Advanced setup"""
    console.print("\n[bold red]Fine-tuning Configuration[/bold red]")
    
    questions = [
        inquirer.List('setup_mode',
                     message="? Choose Setup Mode: (Use arrow keys)",
                     choices=[
                         "Basic Setup - Essential configuration only",
                         "Advanced Setup - Full control over all parameters"
                     ],
                     default="Basic Setup - Essential configuration only")
    ]
    
    answers = inquirer.prompt(questions)
    return "basic" if "Basic" in answers['setup_mode'] else "advanced"

def show_gpu_info():
    """Display available GPU information without prompting for selection"""
    info = get_system_info()
    
    if info["GPU Count"] == 0:
        console.print("\n[yellow]⚠️ No GPUs detected - CPU training mode[/yellow]")
        return {
            "gpu_count": 0,
            "gpus": []
        }
    
    # Display available GPUs
    console.print("\n[bold cyan]🖥️ Available GPUs:[/bold cyan]")
    gpu_table = Table(show_header=True, box=None)
    gpu_table.add_column("Index", style="cyan", width=6)
    gpu_table.add_column("Name", style="white", width=40)
    gpu_table.add_column("VRAM", style="green", width=10)
    
    for i, gpu in enumerate(info["GPUs"]):
        gpu_table.add_row(str(i), gpu['name'], gpu['memory'])
    
    console.print(gpu_table)
    console.print("[dim]💡 GPU selection will be made at training launch[/dim]")
    
    return {
        "gpu_count": info["GPU Count"],
        "gpus": info["GPUs"]
    }

def choose_base_model():
    """Choose base model"""
    models = [
        ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B (77M) - Local model"),
        ("microsoft/Phi-2", "microsoft/Phi-2 (839M) - Local model"), 
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (369M) - Local model")
    ]
    
    questions = [
        inquirer.List('base_model',
                     message="Choose base model: (Use arrow keys)",
                     choices=[display_name for _, display_name in models],
                     default=models[0][1])
    ]
    
    answers = inquirer.prompt(questions)
    selected_display = answers['base_model']
    
    # Find the actual model name (without the extra text)
    for model_name, display_name in models:
        if display_name == selected_display:
            return model_name
    
    return models[0][0]  # fallback

def choose_training_recipe():
    """Choose training recipe"""
    recipes = [
        "QLoRA (4-bit NF4)",
        "QLoRA (4-bit FP4)", 
        "LoRA (FP16)",
        "LoRA (BF16)"
    ]
    
    questions = [
        inquirer.List('training_recipe',
                     message="? Choose training recipe: (Use arrow keys)",
                     choices=recipes,
                     default=recipes[0])
    ]
    
    answers = inquirer.prompt(questions)
    return answers['training_recipe']

# Removed choose_training_mode() - GPU selection now handled in choose_gpu_configuration()

def choose_evaluation_method():
    """Choose evaluation method"""
    console.print("\n[bold red]Evaluation Method[/bold red]")
    console.print("[dim]Choose how to evaluate your model after training:[/dim]")
    
    questions = [
        inquirer.List('evaluation_method',
                     message="? Choose Evaluation Method: (Use arrow keys)",
                     choices=[
                         ("Standard Evaluation - Fast, may have device issues", "standard"),
                         ("Atomic Evaluation - True process isolation, guaranteed success 🚀", "atomic")
                     ])
    ]
    
    answers = inquirer.prompt(questions)
    return answers['evaluation_method']

def choose_dataset():
    """Choose dataset from available local datasets"""
    # Get available datasets by scanning ~/humigence_data/
    available_datasets = list_local_datasets()
    
    if not available_datasets:
        console.print("[yellow]⚠️ No datasets found in ~/humigence_data[/yellow]")
        console.print("[cyan]You can create a dataset by placing a .jsonl file in ~/humigence_data/[/cyan]")
        console.print("[cyan]Each line should be a JSON object with 'instruction' and 'output' fields.[/cyan]")
        console.print("\n[blue]Example format:[/blue]")
        console.print('{"instruction": "What is machine learning?", "output": "Machine learning is..."}')
        
        # Still offer custom path and dummy data options
        console.print("\n[yellow]Available fallback options:[/yellow]")
        choices = ["Enter custom path", "Use dummy data for testing"]
        
        questions = [
            inquirer.List('dataset',
                         message="Choose fallback option: (Use arrow keys)",
                         choices=choices,
                         default=choices[0])
        ]
        
        answers = inquirer.prompt(questions)
        selected_display = answers['dataset']
        
        if selected_display == "Enter custom path":
            custom_path = Prompt.ask("Enter custom dataset path")
            return custom_path
        else:  # Use dummy data
            # Create a minimal dummy dataset
            dummy_data = [
                {"instruction": "Hello", "output": "Hi there!"},
                {"instruction": "How are you?", "output": "I'm doing well, thank you!"}
            ]
            dummy_path = Path.home() / "humigence_data" / "dummy.jsonl"
            dummy_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(dummy_path, 'w') as f:
                for item in dummy_data:
                    f.write(json.dumps(item) + '\n')
            
            console.print(f"[green]✅ Created dummy dataset at {dummy_path}[/green]")
            return str(dummy_path)
    
    # Create choices for inquirer with sample counts
    choices = []
    dataset_map = {}
    
    for name, path, count in available_datasets:
        if count == "?":
            display_name = f"{name} (unknown samples)"
        else:
            # Format count with commas for readability
            formatted_count = f"{count:,}" if isinstance(count, int) else str(count)
            display_name = f"{name} ({formatted_count} samples)"
        
        choices.append(display_name)
        dataset_map[display_name] = path
    
    # Add custom path option
    choices.append("Enter custom path")
    dataset_map["Enter custom path"] = "custom"
    
    questions = [
        inquirer.List('dataset',
                     message="Choose dataset: (Use arrow keys)",
                     choices=choices,
                     default=choices[0])
    ]
    
    answers = inquirer.prompt(questions)
    selected_display = answers['dataset']
    
    if selected_display == "Enter custom path":
        custom_path = Prompt.ask("Enter custom dataset path")
        return custom_path
    else:
        return dataset_map[selected_display]

def show_config_summary(config):
    """Show configuration summary and ask for confirmation"""
    console.print("\n[bold cyan]Configuration Summary[/bold cyan]")
    
    # Display key configuration parameters
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", width=25)
    table.add_column(style="white")
    
    table.add_row("use_flash_attn:", "True")
    table.add_row("lora:", "True")
    table.add_row("qlora:", "True" if "QLoRA" in config["training_recipe"] else "False")
    table.add_row("quant_bits:", "None")
    table.add_row("lora_r:", "16")
    table.add_row("lora_alpha:", "32")
    table.add_row("lora_dropout:", "0.05")
    table.add_row("per_device_train_batch_size:", "2")
    table.add_row("gradient_accumulation_steps:", "4")
    table.add_row("learning_rate:", "0.0002")
    table.add_row("num_train_epochs:", "1.0")
    table.add_row("fp16:", "True")
    table.add_row("bf16:", "False")
    gpu_count = config.get("gpu_info", {}).get("gpu_count", 0)
    if gpu_count == 0:
        table.add_row("gpu_mode:", "CPU Training (No GPUs detected)")
        table.add_row("gpu_ids:", "N/A")
    elif gpu_count == 1:
        gpu_name = config.get("gpu_info", {}).get("gpus", [{}])[0].get("name", "Unknown")
        table.add_row("gpu_mode:", f"Single GPU (will auto-select GPU 0: {gpu_name})")
        table.add_row("gpu_ids:", "Will be [0]")
    else:
        table.add_row("gpu_mode:", f"Multi-GPU or Single GPU (will prompt from {gpu_count} available GPUs)")
        table.add_row("gpu_ids:", "Will be selected at training launch")
    table.add_row("dataset_path:", config["dataset_path"])
    table.add_row("data_schema:", "instruction_output")
    table.add_row("demo_mode:", str(config.get("demo_mode", False)))
    table.add_row("max_samples:", str(config.get("max_samples", "None (full dataset)")))
    table.add_row("train_val_test_split:", str(config.get("train_val_test_split", [0.8, 0.1, 0.1])))
    table.add_row("split_seed:", str(config.get("split_seed", 42)))
    table.add_row("eval_single_gpu:", str(config.get("eval_single_gpu", True)))
    table.add_row("eval_gpu_index:", str(config.get("eval_gpu_index", 0)))
    table.add_row("eval_batch_size:", str(config.get("eval_batch_size", 8)))
    table.add_row("num_workers:", str(config.get("num_workers", 4)))
    table.add_row("pin_memory:", str(config.get("pin_memory", True)))
    table.add_row("max_seq_len:", "1024")
    table.add_row("processed_dir:", "data/processed")
    
    console.print(table)
    
    console.print(f"\n[bold green]✓ Configuration saved to: ./runs/humigence/config.snapshot.json[/bold green]")

def collect_training_config():
    """Main function to collect all training configuration interactively"""
    try:
        # Step 1: Setup Mode
        setup_mode = choose_setup_mode()
        
        # Step 2: System Information
        show_system_info()
        
        # Step 3: GPU Information (no selection yet)
        gpu_info = show_gpu_info()
        
        # Step 4: Base Model Selection
        base_model = choose_base_model()
        
        # Step 5: Training Recipe
        training_recipe = choose_training_recipe()
        
        # Step 6: Dataset Selection
        dataset_path = choose_dataset()
        
        # Build configuration with new invariants
        config = {
            "setup_mode": setup_mode,
            "gpu_info": gpu_info,  # Store GPU info for later use
            "base_model": base_model,
            "training_recipe": training_recipe,
            "dataset_path": dataset_path,
            # A) Config & UX invariants
            "demo_mode": False,  # No silent limits unless demo mode
            "max_samples": None,  # null => use 100% of dataset
            "train_val_test_split": [0.8, 0.1, 0.1],  # deterministic split ratios
            "split_seed": 42,  # deterministic splitting
            "eval_single_gpu": True,  # hard-isolate eval to single GPU
            "eval_gpu_index": 0,  # which GPU to use for evaluation
            "eval_batch_size": 8,  # evaluation batch size
            "num_workers": 4,  # DataLoader workers
            "pin_memory": True,  # DataLoader pin_memory
            # Legacy compatibility
            "split_ratios": [0.8, 0.1, 0.1],  # alias for train_val_test_split
            "min_train_samples": 1000,
            "min_val_samples": 100,
            "min_test_samples": 100,
            "min_tokens_per_sample": 50,
            "random_seed": 42,  # alias for split_seed
            "learning_rate": "2e-4",
            "num_train_epochs": "1",
            "gradient_accumulation_steps": "4",
            "logging_steps": "10",
            "save_steps": "100",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Step 7: Configuration Summary
        show_config_summary(config)
        
        # Step 8: Confirmation
        questions = [
            inquirer.Confirm('proceed',
                           message="? Proceed with training? (Y/n)",
                           default=True)
        ]
        
        answers = inquirer.prompt(questions)
        proceed = answers['proceed']
        
        if not proceed:
            console.print("[bold red]❌ Training aborted — configuration not confirmed.[/bold red]")
            return None
        
        # Save configuration with migration and validation
        from config_migration import save_config_snapshot
        
        config_path = "runs/humigence/config.snapshot.json"
        validated_config = save_config_snapshot(config, config_path)
        
        console.print(f"[bold green]✅ Configuration saved to {config_path}[/bold green]")
        return str(config_path)
        
    except KeyboardInterrupt:
        console.print("\n[bold red]❌ Training aborted — configuration not confirmed.[/bold red]")
        return None
    except Exception as e:
        console.print(f"[bold red]❌ Error during configuration: {e}[/bold red]")
        return None
