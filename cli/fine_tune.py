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
    # Default Hugging Face cache path
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--")
    model_choices = []

    if os.path.exists(hf_cache):
        for root, dirs, files in os.walk(hf_cache):
            for d in dirs:
                if d.startswith("snapshots"):
                    model_dir = os.path.basename(os.path.dirname(root))
                    model_choices.append(model_dir.replace("models--", "").replace("--", "/"))
    
    # Add manually defined models
    model_choices += [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/Phi-2",
        "Qwen/Qwen1.5-0.5B",
        "manual-entry (custom path/repo)"
    ]

    # De-dupe and sort
    return sorted(list(set(model_choices)))

def run():
    console.print("\n[bold magenta]🧪 Supervised Fine-Tuning Setup[/bold magenta]")

    questions = [
        {
            "type": "list",
            "name": "setup_mode",
            "message": "Choose Setup Mode:",
            "choices": ["Basic Setup – Essential configuration only", "Advanced Setup – Full control over all parameters"],
        }
    ]

    answers = prompt(questions)
    setup_mode = answers.get("setup_mode").split(" ")[0].lower()  # 'basic' or 'advanced'
    
    console.print(f"\n[green]✅ You selected:[/green] [yellow]{answers.get('setup_mode')}[/yellow]")

    # Display system summary
    display_system_summary()

    # GPU selection
    gpu_options = []
    info = get_system_info()
    for idx, gpu in enumerate(info['GPUs']):
        gpu_options.append(f"Single GPU – GPU {idx}: {gpu['name']}")

    if len(gpu_options) > 1:
        gpu_options.append("Multi-GPU – All")
        gpu_options.append("Multi-GPU – Custom")

    gpu_question = [
        {
            "type": "list",
            "name": "gpu_choice",
            "message": "�� Choose Training Configuration:",
            "choices": gpu_options,
        }
    ]
    gpu_answer = prompt(gpu_question)
    selected_gpu = gpu_answer.get("gpu_choice")

    console.print(f"\n[green]✅ You selected GPU config:[/green] [yellow]{selected_gpu}[/yellow]")

    # Model selection
    model_question = [
        {
            "type": "list",
            "name": "base_model",
            "message": "🧠 Choose Base Model:",
            "choices": get_available_models()
        }
    ]

    model_answer = prompt(model_question)
    selected_model = model_answer.get("base_model")

    # If manual-entry selected
    if selected_model == "manual-entry (custom path/repo)":
        manual_input = prompt([
            {
                "type": "input",
                "name": "custom_model",
                "message": "Enter Hugging Face repo or local model path:"
            }
        ])
        selected_model = manual_input.get("custom_model")

    console.print(f"\n[green]✅ You selected model:[/green] [yellow]{selected_model}[/yellow]")

    # Dataset selection
    dataset_options = detect_datasets()
    if not dataset_options:
        console.print("[bold red]⚠️ No datasets found in ~/humigence_data[/bold red]")
        return

    dataset_question = [
        {
            "type": "list",
            "name": "dataset_path",
            "message": "📚 Choose Dataset to Train On:",
            "choices": [opt[0] for opt in dataset_options]
        }
    ]

    dataset_answer = prompt(dataset_question)
    selected_dataset = [
        path for name, path in dataset_options if name == dataset_answer["dataset_path"]
    ][0]

    console.print(f"\n[green]✅ You selected dataset:[/green] [yellow]{selected_dataset}[/yellow]")

    # Training recipe selection
    recipe_question = [
        {
            "type": "list",
            "name": "recipe",
            "message": "🧪 Choose Training Recipe:",
            "choices": [
                "QLoRA (4-bit NF4)",
                "LoRA (FP16)",
                "LoRA (BF16)",
                "Full Fine-tuning (FP32)"
            ],
        }
    ]

    recipe_answer = prompt(recipe_question)
    selected_recipe = recipe_answer.get("recipe")

    console.print(f"\n[green]✅ Training recipe:[/green] [yellow]{selected_recipe}[/yellow]")

    # Parameter branching - Basic vs Advanced
    if setup_mode == "advanced":
        param_questions = [
            {
                "type": "input",
                "name": "learning_rate",
                "message": "Enter Learning Rate:",
                "default": "2e-5"
            },
            {
                "type": "input",
                "name": "num_train_epochs",
                "message": "Enter Number of Epochs:",
                "default": "3"
            },
            {
                "type": "input",
                "name": "gradient_accumulation_steps",
                "message": "Enter Gradient Accumulation Steps:",
                "default": "4"
            },
            {
                "type": "input",
                "name": "logging_steps",
                "message": "Enter Logging Steps:",
                "default": "10"
            },
            {
                "type": "input",
                "name": "save_steps",
                "message": "Enter Save Steps:",
                "default": "100"
            }
        ]

        param_answers = prompt(param_questions)
    else:
        # Basic mode defaults
        param_answers = {
            "learning_rate": "2e-5",
            "num_train_epochs": "3",
            "gradient_accumulation_steps": "4",
            "logging_steps": "10",
            "save_steps": "100"
        }

    console.print(f"\n[cyan]📦 Hyperparameters Loaded:[/cyan]")
    for k, v in param_answers.items():
        console.print(f"[bold]{k}[/bold]: {v}")

    # Combine config
    final_config = {
        "setup_mode": setup_mode,
        "gpu_config": selected_gpu,
        "base_model": selected_model,
        "dataset_path": selected_dataset,
        "training_recipe": selected_recipe,
        **param_answers,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Create directory and write config snapshot
    run_dir = Path("runs/humigence")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "config.snapshot.json"

    with open(snapshot_path, "w") as f:
        json.dump(final_config, f, indent=2)

    console.print(f"\n[bold green]✅ Configuration saved to:[/bold green] [cyan]{snapshot_path}[/cyan]")

    # Generate reproduce.sh script
    reproduce_script = f"""#!/bin/bash
# Re-run this exact training config
python3 -m pipelines.lora_trainer --config {snapshot_path}
"""

    reproduce_path = run_dir / "reproduce.sh"
    with open(reproduce_path, "w") as f:
        f.write(reproduce_script)

    # Make executable
    reproduce_path.chmod(0o755)

    console.print(f"[bold green]✅ Reproduction script saved to:[/bold green] [cyan]{reproduce_path}[/cyan]")

    # Final confirmation prompt
    final_prompt = prompt([
        {
            "type": "confirm",
            "name": "confirm_training",
            "message": "🚀 Proceed with training now?",
            "default": True
        }
    ])

    if not final_prompt["confirm_training"]:
        console.print("[bold yellow]❌ Training cancelled.[/bold yellow]")
        return
    else:
        console.print("[bold green]�� Starting training...[/bold green]")
        # Call training engine next (Step 13)

if __name__ == "__main__":
    run()
