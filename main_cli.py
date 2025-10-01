#!/usr/bin/env python3
"""
Humigence CLI - Main entry point for all Humigence commands
"""

import typer
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path
import sys
import os
from datetime import datetime

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from training.train_wikitext import run_training, run_training_from_config
from training.autodetect import detect_family, suggested_lora_targets
from validation.matrix import (
    get_gpu_info, precision_supported, estimate_model_params,
    estimate_memory_bytes, tokenizer_ok, PRECISIONS,
)
from validation.dryrun import dry_run
from validation.fallback import FallbackSimulator, ConfigCandidate
from config.schema import ValidationConfig, TrainingConfig, ConfigMetadata, save_config, validation_to_training_config

app = typer.Typer(
    name="humigence",
    help="Your AI. Your pipeline. Zero code.",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def train_wikitext(
    model: str = typer.Option(
        "", 
        "--model", 
        "-m", 
        help="Path or Hugging Face model name (e.g., 'gpt2' or 'microsoft/DialoGPT-small')"
    ),
    output_dir: str = typer.Option(
        ..., 
        "--output-dir", 
        "-o", 
        help="Directory where checkpoints will be saved"
    ),
    epochs: int = typer.Option(
        1, 
        "--epochs", 
        "-e", 
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        2, 
        "--batch-size", 
        "-b", 
        help="Per-device batch size"
    ),
    learning_rate: float = typer.Option(
        5e-5, 
        "--learning-rate", 
        "-lr", 
        help="Learning rate for training"
    ),
    dataset: str = typer.Option(
        "wikitext", 
        "--dataset", 
        help="Dataset name (default: wikitext)"
    ),
    dataset_config: str = typer.Option(
        "wikitext-2-raw-v1", 
        "--dataset-config", 
        help="Dataset configuration (default: wikitext-2-raw-v1)"
    ),
    max_steps: Optional[int] = typer.Option(
        None, 
        "--max-steps", 
        help="Maximum training steps (overrides epochs if set)"
    ),
    block_size: int = typer.Option(
        1024, 
        "--block-size", 
        help="Maximum sequence length"
    ),
    grad_accum: int = typer.Option(
        4, 
        "--grad-accum", 
        help="Gradient accumulation steps"
    ),
    warmup_steps: int = typer.Option(
        100, 
        "--warmup-steps", 
        help="Number of warmup steps"
    ),
    logging_steps: int = typer.Option(
        10, 
        "--logging-steps", 
        help="Logging frequency in steps"
    ),
    save_steps: int = typer.Option(
        200, 
        "--save-steps", 
        help="Model saving frequency in steps"
    ),
    eval_steps: int = typer.Option(
        200, 
        "--eval-steps", 
        help="Evaluation frequency in steps"
    ),
    lora_r: int = typer.Option(
        8, 
        "--lora-r", 
        help="LoRA rank"
    ),
    lora_alpha: int = typer.Option(
        32, 
        "--lora-alpha", 
        help="LoRA alpha parameter"
    ),
    lora_dropout: float = typer.Option(
        0.05, 
        "--lora-dropout", 
        help="LoRA dropout rate"
    ),
    config: Optional[str] = typer.Option(
        None, 
        "--config", 
        help="Load configuration from YAML file"
    ),
):
    """
    Train a model on Wikitext dataset using LoRA fine-tuning.
    
    This command fine-tunes a language model on the Wikitext dataset using LoRA (Low-Rank Adaptation)
    for efficient parameter updates. The training runs on a single GPU by default.
    
    Examples:
        # Basic training with GPT-2
        humigence train-wikitext --model gpt2 --output-dir ./out
        
        # Training with custom parameters
        humigence train-wikitext --model microsoft/DialoGPT-small --output-dir ./out --epochs 2 --batch-size 4 --learning-rate 1e-4
        
        # Training with specific steps instead of epochs
        humigence train-wikitext --model gpt2 --output-dir ./out --max-steps 1000 --batch-size 2
        
        # Training with config file
        humigence train-wikitext --config ./myconfig.yaml --output-dir ./out
    """
    
    # Validate that either model or config is provided
    if not config and not model:
        console.print("[bold red]❌ Error: Either --model or --config must be provided[/bold red]")
        raise typer.Exit(1)
    
    # Load config from file if provided
    if config:
        try:
            from config.schema import load_config, validation_to_training_config
            # Try to load as TrainingConfig first, then ValidationConfig
            try:
                loaded_config, metadata = load_config(config, TrainingConfig)
            except Exception:
                # If it fails, try loading as ValidationConfig and convert
                validation_config, metadata = load_config(config, ValidationConfig)
                loaded_config = validation_to_training_config(validation_config, output_dir)
            
            # Override with CLI arguments (CLI takes precedence)
            config_dict = loaded_config.dict()
            
            # Update with CLI values (only if they're not default values)
            if model != "":  # If model was provided via CLI
                config_dict["model"] = model
            if output_dir != "":  # If output_dir was provided via CLI
                config_dict["output_dir"] = output_dir
            if epochs != 1:
                config_dict["epochs"] = epochs
            if batch_size != 2:
                config_dict["batch_size"] = batch_size
            if learning_rate != 5e-5:
                config_dict["learning_rate"] = learning_rate
            if dataset != "wikitext":
                config_dict["dataset"] = dataset
            if dataset_config != "wikitext-2-raw-v1":
                config_dict["dataset_config"] = dataset_config
            if max_steps is not None:
                config_dict["max_steps"] = max_steps
            if block_size != 1024:
                config_dict["block_size"] = block_size
            if grad_accum != 4:
                config_dict["grad_accum"] = grad_accum
            if warmup_steps != 100:
                config_dict["warmup_steps"] = warmup_steps
            if logging_steps != 10:
                config_dict["logging_steps"] = logging_steps
            if save_steps != 200:
                config_dict["save_steps"] = save_steps
            if eval_steps != 200:
                config_dict["eval_steps"] = eval_steps
            if lora_r != 8:
                config_dict["lora_r"] = lora_r
            if lora_alpha != 32:
                config_dict["lora_alpha"] = lora_alpha
            if lora_dropout != 0.05:
                config_dict["lora_dropout"] = lora_dropout
            
            # Create new config with merged values
            final_config = TrainingConfig(**config_dict)
            
            # Extract values for display and function call
            model = final_config.model
            output_dir = final_config.output_dir
            dataset = final_config.dataset
            dataset_config = final_config.dataset_config
            epochs = final_config.epochs
            batch_size = final_config.batch_size
            learning_rate = final_config.learning_rate
            max_steps = final_config.max_steps
            block_size = final_config.block_size
            grad_accum = final_config.grad_accum
            warmup_steps = final_config.warmup_steps
            logging_steps = final_config.logging_steps
            save_steps = final_config.save_steps
            eval_steps = final_config.eval_steps
            lora_r = final_config.lora_r
            lora_alpha = final_config.lora_alpha
            lora_dropout = final_config.lora_dropout
            
            console.print(f"[bold blue]📁 Loaded configuration from {config}[/bold blue]")
            
            # Display provenance information if metadata is available
            if metadata:
                provenance_info = f"Created: {metadata.created}"
                if metadata.gpu:
                    provenance_info += f" | GPU: {metadata.gpu}"
                if metadata.auto_heal and metadata.fallback_chain:
                    provenance_info += f" | Auto-healed: {' → '.join(metadata.fallback_chain)}"
                elif metadata.auto_heal:
                    provenance_info += " | Auto-healed: (no fallbacks needed)"
                else:
                    provenance_info += " | Direct validation (no auto-healing)"
                
                console.print(f"[dim]📋 {provenance_info}[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]❌ Failed to load config from {config}: {e}[/bold red]")
            raise typer.Exit(1)
    
    # Display training configuration
    config_panel = Panel(
        f"""[bold blue]Training Configuration[/bold blue]
        
[cyan]Model:[/cyan] {model}
[cyan]Output Directory:[/cyan] {output_dir}
[cyan]Epochs:[/cyan] {epochs}
[cyan]Batch Size:[/cyan] {batch_size}
[cyan]Learning Rate:[/cyan] {learning_rate}
[cyan]Dataset:[/cyan] {dataset}/{dataset_config}
[cyan]Max Steps:[/cyan] {max_steps if max_steps else 'Auto-calculated'}
[cyan]Block Size:[/cyan] {block_size}
[cyan]Gradient Accumulation:[/cyan] {grad_accum}
[cyan]LoRA Rank:[/cyan] {lora_r}
[cyan]LoRA Alpha:[/cyan] {lora_alpha}
[cyan]LoRA Dropout:[/cyan] {lora_dropout}""",
        title="🚀 Starting Wikitext Training",
        border_style="green"
    )
    
    console.print(config_panel)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        if config:
            # Use config-based training with launcher
            from training.launcher import launch_training
            result = launch_training(final_config)
        else:
            # Use individual parameters - convert to TrainingConfig and use launcher
            from config.schema import TrainingConfig
            from training.launcher import launch_training
            
            training_config = TrainingConfig(
                model=model,
                output_dir=output_dir,
                dataset=dataset,
                dataset_config=dataset_config,
                precision="fp16",
                seq_len=block_size,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                max_steps=max_steps,
                block_size=block_size,
                grad_accum=grad_accum,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                lora=True,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                gradient_checkpointing=True,
                text_field="text",
                schema="plain",
                gpu_mode="single",
                gpu_ids=[0]
            )
            
            result = launch_training(training_config)
        
        if result["status"] == "success":
            console.print(Panel(
                f"""[bold green]✅ Training Completed Successfully![/bold green]
                
[cyan]Output Directory:[/cyan] {result['output_dir']}
[cyan]Model Path:[/cyan] {result['model_path']}

[bold blue]Final Metrics:[/bold blue]
[cyan]Train Loss:[/cyan] {result['metrics'].get('train_loss', 'N/A')}
[cyan]Eval Loss:[/cyan] {result['metrics'].get('eval_loss', 'N/A')}
[cyan]Total Steps:[/cyan] {result['metrics'].get('total_steps', 'N/A')}
[cyan]Epochs:[/cyan] {result['metrics'].get('epochs', 'N/A')}
[cyan]Train Runtime:[/cyan] {result['metrics'].get('train_runtime', 'N/A')}s
[cyan]Samples/Second:[/cyan] {result['metrics'].get('train_samples_per_second', 'N/A')}""",
                title="🎉 Training Results",
                border_style="green"
            ))
            return
        else:
            console.print(Panel(
                f"""[bold red]❌ Training Failed[/bold red]
                
[red]Error:[/red] {result.get('error', 'Unknown error')}
[cyan]Output Directory:[/cyan] {result.get('output_dir', 'N/A')}""",
                title="💥 Training Error",
                border_style="red"
            ))
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(Panel(
            f"""[bold red]❌ Unexpected Error[/bold red]
            
[red]Error:[/red] {str(e)}""",
            title="💥 Unexpected Error",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML configuration file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Override output directory"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Override number of epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Override batch size"),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", "-lr", help="Override learning rate"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Override maximum training steps"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Override dataset specification"),
    text_field: Optional[str] = typer.Option(None, "--text-field", help="Override text field for HF datasets"),
    schema: Optional[str] = typer.Option(None, "--schema", help="Override schema for JSONL datasets"),
    gradient_checkpointing: Optional[bool] = typer.Option(None, "--gradient-checkpointing/--no-gradient-checkpointing", help="Override gradient checkpointing"),
    flash_attn: Optional[bool] = typer.Option(None, "--flash-attn/--no-flash-attn", help="Override flash attention"),
    dtype: Optional[str] = typer.Option(None, "--dtype", help="Override data type: fp32|fp16|bf16"),
    gpu_mode: Optional[str] = typer.Option(None, "--gpu-mode", help="Override GPU mode: single|multi"),
    gpu_ids: Optional[str] = typer.Option(None, "--gpu-ids", help="Override GPU IDs (comma-separated, e.g., '0,1,2')"),
):
    """
    Train a model using a configuration file with dataset-agnostic support.
    
    This command supports training on:
    - Wikitext datasets (wikitext)
    - JSONL SFT datasets (jsonl:path/to/file.jsonl)
    - Hugging Face datasets (hf:dataset_name or dataset_name)
    
    Examples:
        # Train with Wikitext
        humigence train --config gpt2_wikitext.yaml
        
        # Train with JSONL SFT dataset
        humigence train --config my_sft_config.yaml
        
        # Train with Hugging Face dataset
        humigence train --config imdb_config.yaml
        
        # Override specific parameters
        humigence train --config my_config.yaml --epochs 3 --batch-size 4
    """
    
    # Load configuration
    try:
        from config.schema import load_config, validation_to_training_config
        # Try to load as TrainingConfig first, then ValidationConfig
        try:
            loaded_config, metadata = load_config(config, TrainingConfig)
        except Exception:
            # If it fails, try loading as ValidationConfig and convert
            validation_config, metadata = load_config(config, ValidationConfig)
            if not output_dir:
                console.print("[bold red]❌ Error: --output-dir is required when using ValidationConfig[/bold red]")
                raise typer.Exit(1)
            loaded_config = validation_to_training_config(validation_config, output_dir)
        
        # Override with CLI arguments (CLI takes precedence)
        config_dict = loaded_config.dict()
        
        if output_dir:
            config_dict["output_dir"] = output_dir
        if epochs is not None:
            config_dict["epochs"] = epochs
        if batch_size is not None:
            config_dict["batch_size"] = batch_size
        if learning_rate is not None:
            config_dict["learning_rate"] = learning_rate
        if max_steps is not None:
            config_dict["max_steps"] = max_steps
        if dataset:
            config_dict["dataset"] = dataset
        if text_field:
            config_dict["text_field"] = text_field
        if schema:
            config_dict["schema"] = schema
        if gradient_checkpointing is not None:
            config_dict["gradient_checkpointing"] = gradient_checkpointing
        if flash_attn is not None:
            config_dict["flash_attn"] = flash_attn
        if dtype:
            config_dict["dtype"] = dtype
        if gpu_mode:
            config_dict["gpu_mode"] = gpu_mode
        if gpu_ids:
            # Parse comma-separated GPU IDs
            try:
                gpu_ids_list = [int(x.strip()) for x in gpu_ids.split(",")]
                config_dict["gpu_ids"] = gpu_ids_list
            except ValueError:
                console.print(f"[red]❌ Invalid GPU IDs format: {gpu_ids}. Use comma-separated integers (e.g., '0,1,2')[/red]")
                raise typer.Exit(1)
        
        # Create final config
        final_config = TrainingConfig(**config_dict)
        
        console.print(f"[bold blue]📁 Loaded configuration from {config}[/bold blue]")
        
        # Display provenance information if metadata is available
        if metadata:
            provenance_info = f"Created: {metadata.created}"
            if metadata.gpu:
                provenance_info += f" | GPU: {metadata.gpu}"
            if metadata.auto_heal and metadata.fallback_chain:
                provenance_info += f" | Auto-healed: {' → '.join(metadata.fallback_chain)}"
            elif metadata.auto_heal:
                provenance_info += " | Auto-healed: (no fallbacks needed)"
            else:
                provenance_info += " | Direct validation (no auto-healing)"
            
            console.print(f"[dim]📋 {provenance_info}[/dim]")
            
            # Display dataset provenance if available
            if metadata.dataset:
                dataset_info = f"📁 Dataset: {metadata.dataset.get('file_path', metadata.dataset.get('dataset_name', 'N/A'))}"
                if metadata.dataset.get('schema'):
                    dataset_info += f" ({metadata.dataset['schema']})"
                console.print(f"[dim]{dataset_info}[/dim]")
                
                if 'train_size' in metadata.dataset and 'eval_size' in metadata.dataset:
                    size_info = f"🔢 Train size: {metadata.dataset['train_size']} | Eval size: {metadata.dataset['eval_size']}"
                    console.print(f"[dim]{size_info}[/dim]")
                
                if 'sha256' in metadata.dataset:
                    sha256 = metadata.dataset['sha256']
                    if len(sha256) > 12:
                        sha256 = sha256[:12] + "..."
                    console.print(f"[dim]🔑 SHA256: {sha256}[/dim]")
            else:
                console.print("[yellow]⚠️  Config missing dataset metadata. Consider re-running validate to persist provenance.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load config from {config}: {e}[/bold red]")
        raise typer.Exit(1)
    
    # Display training configuration
    dataset_info = f"{final_config.dataset.type}"
    if final_config.dataset.path:
        dataset_info += f" ({final_config.dataset.path})"
    elif final_config.dataset.name:
        dataset_info += f" ({final_config.dataset.name})"
    
    config_panel = Panel(
        f"""[bold blue]Training Configuration[/bold blue]
        
[cyan]Model:[/cyan] {final_config.model}
[cyan]Output Directory:[/cyan] {final_config.output_dir}
[cyan]Dataset:[/cyan] {dataset_info}
[cyan]Schema:[/cyan] {final_config.dataset.schema_type or 'auto'}
[cyan]Text Field:[/cyan] {final_config.dataset.text_field or 'auto'}
[cyan]Epochs:[/cyan] {final_config.epochs}
[cyan]Batch Size:[/cyan] {final_config.batch_size}
[cyan]Learning Rate:[/cyan] {final_config.learning_rate}
[cyan]Max Steps:[/cyan] {final_config.max_steps if final_config.max_steps else 'Auto-calculated'}
[cyan]Block Size:[/cyan] {final_config.block_size}
[cyan]Gradient Accumulation:[/cyan] {final_config.grad_accum}
[cyan]LoRA Rank:[/cyan] {final_config.lora_r}
[cyan]LoRA Alpha:[/cyan] {final_config.lora_alpha}
[cyan]LoRA Dropout:[/cyan] {final_config.lora_dropout}
[cyan]Gradient Checkpointing:[/cyan] {final_config.gradient_checkpointing}
[cyan]Flash Attention:[/cyan] {final_config.flash_attn}
[cyan]Data Type:[/cyan] {final_config.dtype}""",
        title="🚀 Starting Dataset-Agnostic Training",
        border_style="green"
    )
    
    console.print(config_panel)
    
    # Create output directory if it doesn't exist
    Path(final_config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        from training.launcher import launch_training
        result = launch_training(final_config)
        
        if result["status"] == "success":
            console.print(Panel(
                f"""[bold green]✅ Training Completed Successfully![/bold green]
                
[cyan]Output Directory:[/cyan] {result['output_dir']}
[cyan]Model Path:[/cyan] {result['model_path']}

[bold blue]Final Metrics:[/bold blue]
[cyan]Train Loss:[/cyan] {result['metrics'].get('train_loss', 'N/A')}
[cyan]Eval Loss:[/cyan] {result['metrics'].get('eval_loss', 'N/A')}
[cyan]Total Steps:[/cyan] {result['metrics'].get('total_steps', 'N/A')}
[cyan]Epochs:[/cyan] {result['metrics'].get('epochs', 'N/A')}
[cyan]Train Runtime:[/cyan] {result['metrics'].get('train_runtime', 'N/A')}s
[cyan]Samples/Second:[/cyan] {result['metrics'].get('train_samples_per_second', 'N/A')}""",
                title="🎉 Training Results",
                border_style="green"
            ))
            return
        else:
            console.print(Panel(
                f"""[bold red]❌ Training Failed[/bold red]
                
[red]Error:[/red] {result.get('error', 'Unknown error')}
[cyan]Output Directory:[/cyan] {result.get('output_dir', 'N/A')}""",
                title="💥 Training Error",
                border_style="red"
            ))
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(Panel(
            f"""[bold red]❌ Unexpected Error[/bold red]
            
[red]Error:[/red] {str(e)}""",
            title="💥 Unexpected Error",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def validate(
    model: str = typer.Option(..., help="HF model id or local path"),
    dataset: str = typer.Option("wikitext", help="Dataset specification: wikitext | jsonl:<path> | hf:<name>"),
    precision: str = typer.Option("fp16", help="fp32|fp16|bf16|qlora4bit"),
    seq_len: int = typer.Option(1024, help="Sequence length"),
    batch_size: int = typer.Option(2, help="Batch size"),
    lora: bool = typer.Option(True, help="Enable LoRA"),
    max_samples: int = typer.Option(128, help="Max samples for schema sniff"),
    text_field: Optional[str] = typer.Option(None, help="Text field for generic HF datasets"),
    schema: Optional[str] = typer.Option(None, help="Schema for JSONL datasets: sft | dialogue | plain | auto"),
    role_markers: bool = typer.Option(True, "--role-markers/--no-role-markers", help="Use role markers for dialogue datasets"),
    user_marker: str = typer.Option("<user>", help="User role marker"),
    assistant_marker: str = typer.Option("<assistant>", help="Assistant role marker"),
    eval_split: Optional[float] = typer.Option(None, help="Fraction of data to use for evaluation (0.0-1.0)"),
    eval_file: Optional[str] = typer.Option(None, help="Path to separate evaluation file (for JSONL)"),
    gradient_checkpointing: bool = typer.Option(False, "--gradient-checkpointing/--no-gradient-checkpointing", help="Enable gradient checkpointing"),
    flash_attn: bool = typer.Option(False, "--flash-attn/--no-flash-attn", help="Enable flash attention"),
    dtype: str = typer.Option("fp16", help="Data type: fp32|fp16|bf16"),
    dry_run_flag: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Do the 1-batch fwd+bwd"),
    auto_heal: bool = typer.Option(True, "--auto-heal/--no-auto-heal", help="Enable auto-healing fallback simulation"),
    max_attempts: int = typer.Option(10, help="Maximum fallback attempts for auto-healing"),
    save_config_path: Optional[str] = typer.Option(None, "--save-config", help="Save auto-healed config to YAML file"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing config file instead of versioning"),
):
    """
    Validate model, dataset, and training configuration before training.
    
    This command performs comprehensive validation including:
    - Model family detection and LoRA target module validation
    - GPU capability and precision support checks
    - Memory estimation and OOM prevention
    - Tokenizer validation
    - Optional 1-batch dry-run to test actual training setup
    
    Examples:
        # Basic validation with GPT-2
        humigence validate --model gpt2 --dataset wikitext --precision fp16
        
        # Validate with BF16 (will fail on non-BF16 GPUs)
        humigence validate --model gpt2 --precision bf16
        
        # Validate with 4-bit quantization
        humigence validate --model gpt2 --precision qlora4bit
        
        # Validate without dry-run
        humigence validate --model gpt2 --no-dry-run
    """
    if precision not in PRECISIONS:
        typer.secho(f"Unsupported precision: {precision}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Detect model family and get config
    family, cfg = detect_family(model)
    gpu = get_gpu_info()
    tok_ok, tok_msg = tokenizer_ok(model)
    prec_ok, prec_msg = precision_supported(precision, gpu)
    
    # Detect dataset type and validate
    dataset_type = _detect_dataset_type(dataset)
    dataset_ok, dataset_msg = _validate_dataset(dataset, dataset_type, text_field, schema)
    
    # Create dataset configuration with eval split support
    dataset_config = _create_dataset_config(dataset, text_field, schema, role_markers, user_marker, assistant_marker, eval_split, eval_file)
    
    # GPU-aware defaults and warnings
    _apply_gpu_aware_defaults(gpu, precision, batch_size, seq_len, gradient_checkpointing, flash_attn, dtype)
    
    # Load dataset to capture metadata
    dataset_metadata = None
    if dataset_ok:
        try:
            from training.data_loader import create_dataset_loader
            loader = create_dataset_loader(
                dataset, 
                text_field=text_field, 
                schema=schema or "auto",
                role_markers=role_markers,
                user_marker=user_marker,
                assistant_marker=assistant_marker,
                eval_split=eval_split,
                eval_file=eval_file
            )
            # Load dataset to get metadata
            train_dataset, eval_dataset = loader.load()
            dataset_metadata = loader.get_metadata()
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load dataset metadata: {e}[/yellow]")
            dataset_metadata = None

    # Estimate parameters and memory
    params = estimate_model_params(cfg)
    mem_est = estimate_memory_bytes(params, precision, adam=True, lora=lora)
    mem_info = f"est ~{mem_est/1e9:.2f} GB" if mem_est else "n/a"

    # Collect warnings
    warns = []
    if not tok_ok:
        warns.append(f"Tokenizer: {tok_msg}")
    if not prec_ok:
        warns.append(f"Precision: {prec_msg}")
    if not dataset_ok:
        warns.append(f"Dataset: {dataset_msg}")
    
    # Check sequence length against model limits
    max_pos = getattr(cfg, "max_position_embeddings", None)
    if max_pos and seq_len > max_pos:
        warns.append(f"seq_len {seq_len} > model limit {max_pos}. Suggest <= {max_pos}.")

    # Create summary table
    tbl = Table(title="Humigence Validation Summary")
    tbl.add_column("Item", style="cyan")
    tbl.add_column("Value", style="white")
    tbl.add_row("Model", model)
    tbl.add_row("Family", family)
    tbl.add_row("Dataset Type", dataset_config.type)
    tbl.add_row("Dataset Path/Name", dataset_config.path or dataset_config.name or "N/A")
    tbl.add_row("Schema", dataset_config.schema_type or "auto")
    tbl.add_row("Text Field", dataset_config.text_field or "auto")
    if dataset_config.type == "jsonl" and dataset_config.schema_type == "dialogue":
        tbl.add_row("Role Markers", f"{dataset_config.user_marker} / {dataset_config.assistant_marker}")
    
    # Add dataset metadata if available
    if dataset_metadata:
        tbl.add_row("Train Size", str(dataset_metadata.get("train_size", "N/A")))
        tbl.add_row("Eval Size", str(dataset_metadata.get("eval_size", "N/A")))
        if "sha256" in dataset_metadata:
            sha256 = dataset_metadata["sha256"]
            if len(sha256) > 12:
                sha256 = sha256[:12] + "..."
            tbl.add_row("SHA256", sha256)
    
    tbl.add_row("Precision", precision)
    tbl.add_row("GPU", f"{gpu.name} (bf16={gpu.bf16_supported}, cc={gpu.cc_major}.{gpu.cc_minor})" if gpu.available else "CPU")
    tbl.add_row("Params (est.)", f"{params:,}" if params else "unknown")
    tbl.add_row("Memory (est.)", mem_info)
    tbl.add_row("Seq Len", str(seq_len))
    tbl.add_row("Batch Size", str(batch_size))
    tbl.add_row("LoRA", str(lora))
    tbl.add_row("Tokenizer", "OK" if tok_ok else f"ISSUE: {tok_msg}")
    tbl.add_row("Precision Support", "OK" if prec_ok else f"ISSUE: {prec_msg}")
    tbl.add_row("Dataset", "OK" if dataset_ok else f"ISSUE: {dataset_msg}")
    console.print(tbl)

    # Display warnings
    if warns:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warns:
            console.print(f" - {w}")

    # Check precision support
    if not prec_ok:
        console.print("\n[bold red]FAIL[/bold red]: Precision not supported.")
        _print_fallback(precision, gpu, lora, seq_len, batch_size)
        raise typer.Exit(2)

    # Perform dry run if requested
    if dry_run_flag:
        console.print("\n[bold]Running 1-batch dry-run...[/bold]")
        lora_targets = suggested_lora_targets(family) if lora else None
        res = dry_run(
            model_id_or_path=model,
            precision=precision,
            seq_len=seq_len,
            batch_size=batch_size,
            lora=lora,
            lora_targets=lora_targets,
        )
        if res.ok:
            console.print(f"[green]PASS[/green]: dry-run completed. loss={res.details.get('loss'):.4f}")
            
            # Save config if requested (even without auto-healing)
            if save_config_path:
                validation_config = ValidationConfig(
                    model=model,
                    dataset=dataset_config,
                    precision=precision,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    lora=lora,
                    lora_targets=lora_targets,
                    gradient_checkpointing=gradient_checkpointing,
                    flash_attn=flash_attn,
                    dtype=dtype,
                    max_samples=max_samples
                )
                
                # Create runtime metadata
                runtime_metadata = _create_runtime_metadata(gpu)
                
                # Create metadata
                metadata = ConfigMetadata(
                    created=datetime.now().isoformat(),
                    gpu=f"{gpu.name} (bf16={gpu.bf16_supported}, cc={gpu.cc_major}.{gpu.cc_minor})" if gpu.available else "CPU",
                    precision_supported=[p for p in ["fp32", "fp16", "bf16", "qlora4bit"] if precision_supported(p, gpu)[0]],
                    validator_version="0.3",
                    auto_heal=False,
                    fallback_chain=[],
                    original_config={
                        "model": model,
                        "precision": precision,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "lora": lora,
                        "gradient_checkpointing": gradient_checkpointing,
                        "flash_attn": flash_attn,
                        "dtype": dtype
                    },
                    dataset=dataset_metadata,
                    runtime=runtime_metadata
                )
                
                saved_path = save_config(validation_config, save_config_path, metadata, overwrite)
                console.print(f"\n[bold green]✅ Config saved to {saved_path}[/bold green]")
            
            raise typer.Exit(0)
        else:
            console.print(f"[red]FAIL[/red]: dry-run error: {res.error}")
            
            # Auto-healing fallback simulation
            if auto_heal:
                console.print(f"[yellow]Auto-healing enabled. Attempting fallback simulation...[/yellow]")
                
                # Create initial config candidate
                initial_config = ConfigCandidate(
                    model=model,
                    precision=precision,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    lora=lora,
                    lora_targets=lora_targets,
                    gradient_checkpointing=False,
                    dataset=dataset,
                    text_field=text_field
                )
                
                # Run fallback simulation
                simulator = FallbackSimulator()
                success, final_config = simulator.simulate_fallbacks(initial_config, max_attempts)
                
                if success:
                    console.print(f"\n[bold green]🎉 AUTO-HEALING SUCCESSFUL![/bold green]")
                    console.print(f"[dim]Found working configuration after {len(simulator.attempts)} attempts[/dim]")
                    
                    # Generate and display YAML config
                    yaml_config = simulator.generate_yaml_config(final_config)
                    console.print(f"\n[bold blue]AUTO-HEALED CONFIG PATCH[/bold blue]")
                    console.print(f"[dim]```yaml[/dim]")
                    console.print(yaml_config)
                    console.print(f"[dim]```[/dim]")
                    
                    # Save config if requested
                    if save_config_path:
                        # Create ValidationConfig from final_config
                        validation_config = ValidationConfig(
                            model=final_config.model,
                            dataset=final_config.dataset,
                            precision=final_config.precision,
                            seq_len=final_config.seq_len,
                            batch_size=final_config.batch_size,
                            lora=final_config.lora,
                            lora_targets=final_config.lora_targets,
                            gradient_checkpointing=final_config.gradient_checkpointing,
                            text_field=final_config.text_field,
                            schema=getattr(final_config, 'schema', schema),
                            max_samples=max_samples
                        )
                        
                        # Create fallback chain from simulator attempts
                        fallback_chain = []
                        for attempt in simulator.attempts[1:]:  # Skip initial attempt
                            if attempt.notes:
                                fallback_chain.append(attempt.notes)
                            else:
                                # Generate fallback description from config changes
                                prev_config = simulator.attempts[attempt.attempt_num - 2].config
                                curr_config = attempt.config
                                
                                changes = []
                                if prev_config.precision != curr_config.precision:
                                    changes.append(f"precision {prev_config.precision} → {curr_config.precision}")
                                if prev_config.seq_len != curr_config.seq_len:
                                    changes.append(f"seq_len {prev_config.seq_len} → {curr_config.seq_len}")
                                if prev_config.batch_size != curr_config.batch_size:
                                    changes.append(f"batch_size {prev_config.batch_size} → {curr_config.batch_size}")
                                if prev_config.gradient_checkpointing != curr_config.gradient_checkpointing:
                                    changes.append(f"gradient_checkpointing {prev_config.gradient_checkpointing} → {curr_config.gradient_checkpointing}")
                                
                                if changes:
                                    fallback_chain.append(", ".join(changes))
                        
                        # Create metadata with fallback chain
                        metadata = ConfigMetadata(
                            created=datetime.now().isoformat(),
                            gpu=f"{gpu.name} (bf16={gpu.bf16_supported}, cc={gpu.cc_major}.{gpu.cc_minor})" if gpu.available else "CPU",
                            precision_supported=[p for p in ["fp32", "fp16", "bf16", "qlora4bit"] if precision_supported(p, gpu)[0]],
                            validator_version="0.3",
                            auto_heal=True,
                            fallback_chain=fallback_chain,
                            original_config={
                                "model": model,
                                "precision": precision,
                                "seq_len": seq_len,
                                "batch_size": batch_size,
                                "lora": lora
                            },
                            dataset=dataset_metadata
                        )
                        
                        saved_path = save_config(validation_config, save_config_path, metadata, overwrite)
                        console.print(f"\n[bold green]✅ Auto-healed config saved to {saved_path}[/bold green]")
                    
                    raise typer.Exit(0)
                else:
                    console.print(f"\n[bold red]❌ AUTO-HEALING FAILED[/bold red]")
                    console.print(f"[dim]Could not find working configuration after {max_attempts} attempts[/dim]")
                    _print_fallback(precision, gpu, lora, seq_len, batch_size, res.oom)
                    raise typer.Exit(3)
            else:
                # No auto-healing, just show fallback suggestions
                if res.oom:
                    console.print("[yellow]Detected OOM. Proposing fallback...[/yellow]")
                _print_fallback(precision, gpu, lora, seq_len, batch_size, res.oom)
                raise typer.Exit(3)
    else:
        # No dry-run; rely on static checks
        if warns:
            console.print("[yellow]COMPLETE WITH WARNINGS[/yellow]")
            raise typer.Exit(0)
        console.print("[green]PASS[/green]")
        raise typer.Exit(0)


def _detect_dataset_type(dataset_spec: str) -> str:
    """Detect dataset type from specification"""
    if dataset_spec == "wikitext":
        return "wikitext"
    elif dataset_spec.startswith("jsonl:"):
        return "jsonl"
    elif dataset_spec.startswith("hf:"):
        return "hf"
    else:
        # Assume it's a direct HF dataset name
        return "hf"


def _create_dataset_config(dataset_spec: str, text_field: Optional[str], schema: Optional[str], 
                          role_markers: bool, user_marker: str, assistant_marker: str,
                          eval_split: Optional[float] = None, eval_file: Optional[str] = None):
    """Create DatasetConfig from CLI parameters"""
    from config.schema import DatasetConfig
    
    dataset_type = _detect_dataset_type(dataset_spec)
    
    if dataset_type == "wikitext":
        return DatasetConfig(type="wikitext", name="wikitext")
    
    elif dataset_type == "jsonl":
        file_path = dataset_spec[6:]  # Remove "jsonl:" prefix
        return DatasetConfig(
            type="jsonl",
            path=file_path,
            schema_type=schema or "auto",
            role_markers=role_markers,
            user_marker=user_marker,
            assistant_marker=assistant_marker,
            eval_split=eval_split,
            eval_file=eval_file
        )
    
    elif dataset_type == "hf":
        dataset_name = dataset_spec[3:] if dataset_spec.startswith("hf:") else dataset_spec
        return DatasetConfig(
            type="hf",
            name=dataset_name,
            text_field=text_field or "text",
            eval_split=eval_split
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _apply_gpu_aware_defaults(gpu, precision: str, batch_size: int, seq_len: int, 
                             gradient_checkpointing: bool, flash_attn: bool, dtype: str):
    """Apply GPU-aware defaults and warnings"""
    if not gpu.available:
        console.print("[yellow]⚠️  No GPU detected - using CPU mode[/yellow]")
        return
    
    # Get GPU memory info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"[blue]🔧 GPU Memory: {gpu_memory_gb:.1f}GB[/blue]")
            
            # Warn about potential OOM issues
            if precision == "fp32" and gpu_memory_gb < 24:
                console.print(f"[yellow]⚠️  Detected {gpu_memory_gb:.1f}GB GPU — fp32 may OOM, recommend fp16 with batch_size<=4[/yellow]")
            elif precision == "bf16" and not gpu.bf16_supported:
                console.print(f"[yellow]⚠️  GPU doesn't support BF16, recommend fp16[/yellow]")
            elif batch_size > 4 and gpu_memory_gb < 16:
                console.print(f"[yellow]⚠️  Large batch size ({batch_size}) on {gpu_memory_gb:.1f}GB GPU may cause OOM[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not get GPU memory info: {e}[/yellow]")


def _create_runtime_metadata(gpu) -> Dict[str, Any]:
    """Create runtime environment metadata"""
    runtime_metadata = {}
    
    try:
        import torch
        import platform
        
        # GPU info
        if gpu.available:
            runtime_metadata["gpu"] = gpu.name
            runtime_metadata["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            runtime_metadata["cuda"] = torch.version.cuda
        else:
            runtime_metadata["gpu"] = "CPU"
            runtime_metadata["vram_gb"] = 0
            runtime_metadata["cuda"] = None
        
        # PyTorch version
        runtime_metadata["torch"] = torch.__version__
        
        # System info
        runtime_metadata["platform"] = platform.platform()
        runtime_metadata["python"] = platform.python_version()
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not collect runtime metadata: {e}[/yellow]")
        runtime_metadata["error"] = str(e)
    
    return runtime_metadata


def _validate_dataset(dataset_spec: str, dataset_type: str, text_field: Optional[str], schema: Optional[str]) -> tuple[bool, str]:
    """Validate dataset specification and accessibility"""
    try:
        if dataset_type == "wikitext":
            # Wikitext is always valid
            return True, "OK"
        
        elif dataset_type == "jsonl":
            file_path = dataset_spec[6:]  # Remove "jsonl:" prefix
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            # Try to read first line to validate JSON format
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        import json
                        json.loads(first_line)
                return True, "OK"
            except json.JSONDecodeError:
                return False, f"Invalid JSON format in {file_path}"
            except Exception as e:
                return False, f"Error reading {file_path}: {e}"
        
        elif dataset_type == "hf":
            dataset_name = dataset_spec[3:] if dataset_spec.startswith("hf:") else dataset_spec
            # Try to load dataset info (without actually downloading)
            try:
                from datasets import get_dataset_infos
                infos = get_dataset_infos(dataset_name)
                if not infos:
                    return False, f"Dataset {dataset_name} not found"
                return True, "OK"
            except Exception as e:
                return False, f"Error accessing dataset {dataset_name}: {e}"
        
        else:
            return False, f"Unknown dataset type: {dataset_type}"
    
    except Exception as e:
        return False, f"Dataset validation error: {e}"


def _print_fallback(precision: str, gpu, lora: bool, seq_len: int, batch_size: int, oom: bool = False):
    """Print fallback configuration recommendations"""
    console.print("\n[bold]RECOMMENDED CONFIG PATCH[/bold]")
    suggest = {
        "precision": precision,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lora": lora,
        "gradient_checkpointing": False,
    }
    
    # Precision fallback
    if precision == "bf16" and not gpu.bf16_supported:
        suggest["precision"] = "fp16"
    if precision == "qlora4bit" and not gpu.available:
        suggest["precision"] = "fp16"
    
    # OOM mitigations
    if oom:
        if batch_size > 1:
            suggest["batch_size"] = max(1, batch_size // 2)
        else:
            suggest["gradient_checkpointing"] = True
            if seq_len > 1024:
                suggest["seq_len"] = min(1024, seq_len // 2)
            if precision in ("bf16", "fp32"):
                suggest["precision"] = "fp16"

    for k, v in suggest.items():
        console.print(f" - {k}: {v}")


@app.command()
def gpu_info():
    """Show detailed GPU information and selection options."""
    from validation.matrix import get_all_gpu_info
    
    multi_gpu_info = get_all_gpu_info()
    
    if not multi_gpu_info.gpus:
        console.print(Panel(
            "[bold red]❌ No GPUs detected[/bold red]\n"
            "[dim]Training will run on CPU[/dim]",
            title="GPU Information",
            border_style="red"
        ))
        return
    
    # Create GPU information table
    table = Table(title="Available GPUs")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Name", style="white", width=40)
    table.add_column("VRAM", style="green", width=12)
    table.add_column("Compute Capability", style="blue", width=15)
    table.add_column("BF16 Support", style="yellow", width=12)
    
    for gpu in multi_gpu_info.gpus:
        vram_gb = gpu.total_bytes / (1024**3)
        cc = f"{gpu.cc_major}.{gpu.cc_minor}"
        bf16_support = "✅ Yes" if gpu.bf16_supported else "❌ No"
        
        table.add_row(
            str(gpu.device_index),
            gpu.name,
            f"{vram_gb:.1f} GB",
            cc,
            bf16_support
        )
    
    console.print(table)
    
    # Show selection examples
    console.print(Panel(
        f"""[bold blue]GPU Selection Examples[/bold blue]

[cyan]Single GPU Training:[/cyan]
  humigence train --config my_config.yaml --gpu-mode single --gpu-ids 0

[cyan]Multi-GPU Training (all GPUs):[/cyan]
  humigence train --config my_config.yaml --gpu-mode multi --gpu-ids 0,1

[cyan]Multi-GPU Training (specific GPUs):[/cyan]
  humigence train --config my_config.yaml --gpu-mode multi --gpu-ids 1,2

[dim]Total VRAM: {multi_gpu_info.total_vram_gb:.1f} GB across {multi_gpu_info.count} GPUs[/dim]""",
        title="Usage Examples",
        border_style="green"
    ))


@app.command()
def version():
    """Show version information."""
    console.print("[bold blue]Humigence v1.0.0[/bold blue]")
    console.print("[dim]Your AI. Your pipeline. Zero code.[/dim]")


@app.callback()
def main(
    version: bool = typer.Option(
        False, 
        "--version", 
        "-v", 
        help="Show version and exit"
    )
):
    """
    Humigence - Your AI. Your pipeline. Zero code.
    
    A complete MLOps suite built for makers, teams, and enterprises.
    """
    if version:
        console.print("[bold blue]Humigence v1.0.0[/bold blue]")
        console.print("[dim]Your AI. Your pipeline. Zero code.[/dim]")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
