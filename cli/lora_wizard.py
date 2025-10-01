from InquirerPy import prompt
from rich.console import Console
from rich.table import Table
from utils.device import get_system_info
from utils.validators import detect_datasets
import os
import json
from pathlib import Path
import datetime

console = Console()

def display_system_summary():
    info = get_system_info()

    table = Table(title="🖥️ System Detection Summary", show_lines=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    for key, val in info.items():
        if key == "GPUs":
            for i, gpu in enumerate(val):
                table.add_row(f"GPU {i} Name", gpu['name'])
                table.add_row(f"GPU {i} Memory", gpu['memory'])
        else:
            table.add_row(key, str(val))

    console.print("\n")
    console.print(table)

def get_available_models():
    """Get available models for LoRA training with auto-detection."""
    # Default Hugging Face cache path
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--")
    model_choices = []

    if os.path.exists(hf_cache):
        for root, dirs, files in os.walk(hf_cache):
            for d in dirs:
                if d.startswith("snapshots"):
                    model_dir = os.path.basename(os.path.dirname(root))
                    model_choices.append(model_dir.replace("models--", "").replace("--", "/"))
    
    # Add popular models for LoRA training
    model_choices += [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct", 
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "microsoft/Phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen1.5-0.5B",
        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen1.5-7B",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "manual-entry (custom path/repo)"
    ]

    # De-dupe and sort
    return sorted(list(set(model_choices)))

def get_available_datasets():
    """Get available datasets for LoRA training."""
    # Detect local datasets
    local_datasets = detect_datasets()
    
    # Add popular Hugging Face datasets
    hf_datasets = [
        ("wikitext-2-raw-v1", "Hugging Face - WikiText-2 (Raw)"),
        ("wikitext-103-raw-v1", "Hugging Face - WikiText-103 (Raw)"),
        ("openwebtext", "Hugging Face - OpenWebText"),
        ("c4", "Hugging Face - C4 (Common Crawl)"),
        ("bookcorpus", "Hugging Face - BookCorpus"),
    ]
    
    # Combine local and HF datasets
    all_datasets = []
    
    # Add local datasets first
    for name, path in local_datasets:
        all_datasets.append((f"Local - {name}", f"local:{path}"))
    
    # Add HF datasets
    for dataset_id, display_name in hf_datasets:
        all_datasets.append((display_name, f"hf:{dataset_id}"))
    
    return all_datasets

def generate_output_directory(model_name, dataset_name):
    """Generate a meaningful output directory name."""
    # Clean model name for directory
    model_clean = model_name.replace("/", "_").replace(":", "_")
    dataset_clean = dataset_name.replace("/", "_").replace(":", "_")
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"out_lora_{model_clean}_{dataset_clean}_{timestamp}"

def get_lora_presets():
    """Get LoRA configuration presets."""
    return [
        {
            "name": "Efficient (r=8, α=16)",
            "description": "Fast training, lower memory usage",
            "r": 8,
            "alpha": 16,
            "dropout": 0.05
        },
        {
            "name": "Balanced (r=16, α=32)", 
            "description": "Good balance of performance and speed",
            "r": 16,
            "alpha": 32,
            "dropout": 0.05
        },
        {
            "name": "High Quality (r=32, α=64)",
            "description": "Better performance, more parameters",
            "r": 32,
            "alpha": 64,
            "dropout": 0.1
        },
        {
            "name": "Custom Configuration",
            "description": "Set your own LoRA parameters",
            "r": 16,
            "alpha": 32,
            "dropout": 0.05
        }
    ]

def run():
    """Run the LoRA training wizard."""
    console.print("\n[bold magenta]🔧 Single-GPU LoRA Training Setup[/bold magenta]")

    # Setup mode selection
    questions = [
        {
            "type": "list",
            "name": "setup_mode",
            "message": "Choose Setup Mode:",
            "choices": [
                "Quick Start – Recommended settings for most users",
                "Custom Setup – Full control over all parameters"
            ],
        }
    ]

    answers = prompt(questions)
    setup_mode = answers.get("setup_mode").split(" ")[0].lower()  # 'quick' or 'custom'
    
    console.print(f"\n[green]✅ You selected:[/green] [yellow]{answers.get('setup_mode')}[/yellow]")

    # Display system summary
    display_system_summary()

    # Model selection
    console.print("\n[bold blue]🧠 Model Selection[/bold blue]")
    available_models = get_available_models()
    
    model_question = [
        {
            "type": "list",
            "name": "base_model",
            "message": "Choose Base Model for LoRA Training:",
            "choices": available_models
        }
    ]

    model_answer = prompt(model_question)
    selected_model = model_answer.get("base_model")

    # Handle manual entry
    if selected_model == "manual-entry (custom path/repo)":
        manual_input = prompt([
            {
                "type": "input",
                "name": "custom_model",
                "message": "Enter Hugging Face repo or local model path:",
                "validate": lambda x: len(x.strip()) > 0
            }
        ])
        selected_model = manual_input.get("custom_model")

    console.print(f"\n[green]✅ Selected model:[/green] [yellow]{selected_model}[/yellow]")

    # Dataset selection
    console.print("\n[bold blue]📚 Dataset Selection[/bold blue]")
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        console.print("[bold red]⚠️ No datasets found! Please ensure you have datasets available.[/bold red]")
        return None

    dataset_question = [
        {
            "type": "list",
            "name": "dataset",
            "message": "Choose Dataset for Training:",
            "choices": [name for name, _ in available_datasets]
        }
    ]

    dataset_answer = prompt(dataset_question)
    selected_dataset_display = dataset_answer.get("dataset")
    
    # Find the actual dataset path
    selected_dataset = None
    for name, path in available_datasets:
        if name == selected_dataset_display:
            selected_dataset = path
            break

    console.print(f"\n[green]✅ Selected dataset:[/green] [yellow]{selected_dataset_display}[/yellow]")

    # Generate output directory
    output_dir = generate_output_directory(selected_model, selected_dataset_display)
    console.print(f"\n[green]📁 Output directory:[/green] [yellow]{output_dir}[/yellow]")

    # LoRA configuration
    console.print("\n[bold blue]⚙️ LoRA Configuration[/bold blue]")
    lora_presets = get_lora_presets()
    
    lora_question = [
        {
            "type": "list",
            "name": "lora_preset",
            "message": "Choose LoRA Configuration:",
            "choices": [f"{preset['name']} - {preset['description']}" for preset in lora_presets]
        }
    ]

    lora_answer = prompt(lora_question)
    selected_preset = lora_answer.get("lora_preset").split(" - ")[0]
    
    # Find the preset
    selected_lora_config = None
    for preset in lora_presets:
        if preset['name'] == selected_preset:
            selected_lora_config = preset
            break

    console.print(f"\n[green]✅ LoRA config:[/green] [yellow]{selected_preset}[/yellow]")

    # Training parameters
    if setup_mode == "custom":
        console.print("\n[bold blue]🎯 Training Parameters[/bold blue]")
        
        param_questions = [
            {
                "type": "input",
                "name": "max_steps",
                "message": "Maximum training steps:",
                "default": "1000",
                "validate": lambda x: x.isdigit() and int(x) > 0
            },
            {
                "type": "input", 
                "name": "batch_size",
                "message": "Per-device batch size:",
                "default": "4",
                "validate": lambda x: x.isdigit() and int(x) > 0
            },
            {
                "type": "input",
                "name": "grad_accum",
                "message": "Gradient accumulation steps:",
                "default": "4", 
                "validate": lambda x: x.isdigit() and int(x) > 0
            },
            {
                "type": "input",
                "name": "learning_rate",
                "message": "Learning rate:",
                "default": "2e-4",
                "validate": lambda x: float(x) > 0
            },
            {
                "type": "input",
                "name": "block_size",
                "message": "Block size for text grouping:",
                "default": "512",
                "validate": lambda x: x.isdigit() and int(x) > 0
            }
        ]
        
        param_answers = prompt(param_questions)
    else:
        # Quick start defaults
        param_answers = {
            "max_steps": "1000",
            "batch_size": "4", 
            "grad_accum": "4",
            "learning_rate": "2e-4",
            "block_size": "512"
        }

    # Custom LoRA parameters if needed
    if selected_preset == "Custom Configuration":
        console.print("\n[bold blue]🔧 Custom LoRA Parameters[/bold blue]")
        
        custom_lora_questions = [
            {
                "type": "input",
                "name": "lora_r",
                "message": "LoRA rank (r):",
                "default": "16",
                "validate": lambda x: x.isdigit() and int(x) > 0
            },
            {
                "type": "input",
                "name": "lora_alpha", 
                "message": "LoRA alpha:",
                "default": "32",
                "validate": lambda x: x.isdigit() and int(x) > 0
            },
            {
                "type": "input",
                "name": "lora_dropout",
                "message": "LoRA dropout:",
                "default": "0.05",
                "validate": lambda x: 0 <= float(x) <= 1
            }
        ]
        
        custom_lora_answers = prompt(custom_lora_questions)
        selected_lora_config.update({
            "r": int(custom_lora_answers["lora_r"]),
            "alpha": int(custom_lora_answers["lora_alpha"]),
            "dropout": float(custom_lora_answers["lora_dropout"])
        })

    # Parse dataset type
    if selected_dataset.startswith("local:"):
        dataset_name = "jsonl"
        dataset_config = selected_dataset[6:]  # Remove "local:" prefix
    elif selected_dataset.startswith("hf:"):
        dataset_name = "wikitext"
        dataset_config = selected_dataset[3:]  # Remove "hf:" prefix
    else:
        dataset_name = "wikitext"
        dataset_config = selected_dataset

    # Combine all configuration
    final_config = {
        "setup_mode": setup_mode,
        "base_model": selected_model,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_display": selected_dataset_display,
        "output_dir": output_dir,
        "lora_config": selected_lora_config,
        "max_steps": int(param_answers["max_steps"]),
        "batch_size": int(param_answers["batch_size"]),
        "grad_accum": int(param_answers["grad_accum"]),
        "learning_rate": float(param_answers["learning_rate"]),
        "block_size": int(param_answers["block_size"]),
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Display configuration summary
    console.print("\n[bold cyan]📋 Configuration Summary[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Parameter", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Model", selected_model)
    summary_table.add_row("Dataset", selected_dataset_display)
    summary_table.add_row("Output Directory", output_dir)
    summary_table.add_row("LoRA Rank (r)", str(selected_lora_config["r"]))
    summary_table.add_row("LoRA Alpha", str(selected_lora_config["alpha"]))
    summary_table.add_row("LoRA Dropout", str(selected_lora_config["dropout"]))
    summary_table.add_row("Max Steps", str(final_config["max_steps"]))
    summary_table.add_row("Batch Size", str(final_config["batch_size"]))
    summary_table.add_row("Grad Accumulation", str(final_config["grad_accum"]))
    summary_table.add_row("Learning Rate", str(final_config["learning_rate"]))
    summary_table.add_row("Block Size", str(final_config["block_size"]))
    
    console.print(summary_table)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = Path(output_dir) / "lora_config.json"
    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2)

    console.print(f"\n[bold green]✅ Configuration saved to:[/bold green] [cyan]{config_path}[/cyan]")

    # Generate reproduction script
    reproduce_script = f"""#!/bin/bash
# Re-run this exact LoRA training config
cd {Path.cwd()}
python3 cli/train_lora_single.py \\
    --model {selected_model} \\
    --output-dir {output_dir} \\
    --max-steps {final_config["max_steps"]} \\
    --batch-size {final_config["batch_size"]} \\
    --grad-accum {final_config["grad_accum"]} \\
    --learning-rate {final_config["learning_rate"]} \\
    --block-size {final_config["block_size"]} \\
    --lora-r {selected_lora_config["r"]} \\
    --lora-alpha {selected_lora_config["alpha"]} \\
    --lora-dropout {selected_lora_config["dropout"]} \\
    --dataset {dataset_name} \\
    --dataset-config {dataset_config}
"""

    reproduce_path = Path(output_dir) / "reproduce.sh"
    with open(reproduce_path, "w") as f:
        f.write(reproduce_script)
    reproduce_path.chmod(0o755)

    console.print(f"[bold green]✅ Reproduction script saved to:[/bold green] [cyan]{reproduce_path}[/cyan]")

    # Final confirmation
    final_prompt = prompt([
        {
            "type": "confirm",
            "name": "confirm_training",
            "message": "🚀 Start LoRA training now?",
            "default": True
        }
    ])

    if not final_prompt["confirm_training"]:
        console.print("[bold yellow]❌ Training cancelled.[/bold yellow]")
        return None
    else:
        console.print("[bold green]🚀 Starting LoRA training...[/bold green]")
        return final_config

if __name__ == "__main__":
    run()

