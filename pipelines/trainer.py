# pipelines/trainer.py

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import numpy as np
from distributed_utils import RankZeroOnly

console = Console()

class ProductionTrainer:
    """Production-grade trainer with proper logging and checkpoint management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path("runs/humigence")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distributed training setup
        self.ddp = config.get("ddp", False)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)
        self.is_main = config.get("is_main", True)
        self.device = config.get("device", "cuda:0")
        
        # Training configuration
        self.base_model = config["base_model"]
        self.training_recipe = config["training_recipe"]
        self.learning_rate = float(config.get("learning_rate", "2e-4"))
        self.num_epochs = int(config.get("num_train_epochs", "1"))
        self.batch_size = int(config.get("per_device_train_batch_size", "2"))
        self.gradient_accumulation = int(config.get("gradient_accumulation_steps", "4"))
        self.max_samples = config.get("max_samples")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_history = []
        
    def _setup_distributed_training(self):
        """Setup distributed training if enabled"""
        if self.ddp:
            # Wrap model with DDP after moving to device
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if isinstance(self.device, torch.device) else [0],
                output_device=self.device.index if isinstance(self.device, torch.device) else 0
            )
            console.print(f"[blue]✅ Model wrapped with DDP (rank {self.rank})[/blue]")
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with proper configuration"""
        with RankZeroOnly(self.is_main) as rank_zero:
            rank_zero.print(f"[blue]🤖 Loading model: {self.base_model}[/blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine device mapping based on GPU configuration
        multi_gpu = self.config.get("multi_gpu", False)
        selected_gpus = self.config.get("selected_gpus", [])
        
        if multi_gpu and selected_gpus:
            # Multi-GPU training with specific GPU selection
            console.print(f"[blue]🔧 Multi-GPU training on GPUs: {selected_gpus}[/blue]")
            device_map = "auto"  # Let transformers handle distribution
        elif multi_gpu:
            # Multi-GPU training with all available GPUs
            console.print("[blue]🔧 Multi-GPU training on all available GPUs[/blue]")
            device_map = "auto"
        else:
            # Single GPU training
            if selected_gpus:
                gpu_id = selected_gpus[0]
                console.print(f"[blue]🔧 Single GPU training on GPU {gpu_id}[/blue]")
                device_map = f"cuda:{gpu_id}"
            else:
                console.print("[blue]🔧 Single GPU training on default GPU[/blue]")
                device_map = "auto"
        
        # Load model based on training recipe
        if "QLoRA" in self.training_recipe:
            console.print("[blue]🔧 Setting up QLoRA with quantization...[/blue]")
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
        else:
            # Regular LoRA without quantization
            console.print("[blue]🔧 Setting up LoRA without quantization...[/blue]")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if "BF16" in self.training_recipe else torch.float16
            )
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_percentage = (trainable_params / all_params) * 100
        
        console.print(f"[green]✅ Model loaded successfully[/green]")
        console.print(f"[blue]📊 Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}% of all parameters)[/blue]")
    
    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Prepare datasets for training"""
        console.print("[blue]📚 Preparing datasets for training...[/blue]")
        
        # Convert to instruction-response format and tokenize
        train_texts = self._convert_to_texts(train_data)
        val_texts = self._convert_to_texts(val_data)
        test_texts = self._convert_to_texts(test_data)
        
        # Apply max_samples limit if specified
        if self.max_samples and len(train_texts) > self.max_samples:
            console.print(f"[yellow]⚠️ Limiting training data to {self.max_samples} samples[/yellow]")
            train_texts = train_texts[:self.max_samples]
        
        # Tokenize datasets
        train_encodings = self._tokenize_texts(train_texts)
        val_encodings = self._tokenize_texts(val_texts)
        test_encodings = self._tokenize_texts(test_texts)
        
        # Create dataset objects
        self.train_dataset = self._create_dataset(train_encodings)
        self.val_dataset = self._create_dataset(val_encodings)
        self.test_dataset = self._create_dataset(test_encodings)
        
        console.print(f"[green]✅ Datasets prepared: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test[/green]")
    
    def _convert_to_texts(self, data: List[Dict]) -> List[str]:
        """Convert dataset samples to instruction-response text format"""
        texts = []
        for sample in data:
            if "instruction" in sample and "output" in sample:
                instruction = sample["instruction"]
                output = sample["output"]
                input_text = sample.get("input", "")
                
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                texts.append(text)
            else:
                # Fallback for other formats
                values = [v for v in sample.values() if isinstance(v, str)]
                if len(values) >= 2:
                    text = f"### Instruction:\n{values[0]}\n\n### Response:\n{values[1]}"
                    texts.append(text)
        
        return texts
    
    def _tokenize_texts(self, texts: List[str]) -> Dict:
        """Tokenize texts with proper padding and truncation"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    def _create_dataset(self, encodings: Dict) -> List[Dict]:
        """Create dataset from tokenized encodings"""
        dataset = []
        for i in range(len(encodings["input_ids"])):
            dataset.append({
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
                "labels": encodings["input_ids"][i].clone()
            })
        return dataset
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        console.print("[blue]⚙️ Setting up training configuration...[/blue]")
        
        # Calculate effective batch size
        effective_batch_size = self.batch_size * self.gradient_accumulation
        console.print(f"[blue]📊 Effective batch size: {effective_batch_size}[/blue]")
        
        # Get GPU configuration
        multi_gpu = self.config.get("multi_gpu", False)
        use_distributed = self.config.get("use_distributed", False)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            fp16="FP16" in self.training_recipe,
            bf16="BF16" in self.training_recipe,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False,
            # Multi-GPU configuration
            ddp_find_unused_parameters=False if multi_gpu else None,
            dataloader_pin_memory=self.config.get("pin_memory", True),
            dataloader_num_workers=self.config.get("num_workers", 4),
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        console.print("[green]✅ Training setup complete[/green]")
    
    def train(self) -> Dict:
        """Run training with progress tracking"""
        console.print("\n[bold green]🚀 Starting training...[/bold green]")
        
        # Display training configuration
        self._display_training_config()
        
        # Start training
        start_time = time.time()
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Create progress task
            task = progress.add_task("Training...", total=self.num_epochs)
            
            # Train the model
            training_result = self.trainer.train()
            
            # Update progress
            progress.update(task, completed=self.num_epochs)
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Extract training metrics
        train_loss = training_result.training_loss
        eval_loss = training_result.metrics.get("eval_loss", train_loss)
        
        # Get final metrics
        final_metrics = self.trainer.evaluate()
        
        # Store training history
        self.training_history = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "training_duration": training_duration,
            "final_metrics": final_metrics,
            "epochs_completed": self.num_epochs,
            "best_model_path": str(self.output_dir / "checkpoint-best")
        }
        
        console.print(f"[green]✅ Training completed in {training_duration:.2f} seconds[/green]")
        console.print(f"[blue]📊 Final training loss: {train_loss:.4f}[/blue]")
        console.print(f"[blue]📊 Final validation loss: {eval_loss:.4f}[/blue]")
        
        return self.training_history
    
    def _display_training_config(self):
        """Display training configuration"""
        table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Model", self.base_model)
        table.add_row("Recipe", self.training_recipe)
        table.add_row("Learning Rate", str(self.learning_rate))
        table.add_row("Epochs", str(self.num_epochs))
        table.add_row("Batch Size", str(self.batch_size))
        table.add_row("Gradient Accumulation", str(self.gradient_accumulation))
        table.add_row("Train Samples", str(len(self.train_dataset)))
        table.add_row("Val Samples", str(len(self.val_dataset)))
        table.add_row("Test Samples", str(len(self.test_dataset)))
        
        console.print(table)
    
    def save_model(self):
        """Save the trained model and tokenizer"""
        console.print("[blue]💾 Saving model and tokenizer...[/blue]")
        
        # Save model
        self.model.save_pretrained(self.output_dir / "final_model")
        self.tokenizer.save_pretrained(self.output_dir / "tokenizer")
        
        # Save training configuration
        config_save_path = self.output_dir / "training_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history
        history_save_path = self.output_dir / "training_history.json"
        with open(history_save_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        console.print(f"[green]✅ Model saved to: {self.output_dir}[/green]")
        console.print(f"[green]✅ Training history saved to: {history_save_path}[/green]")
    
    def get_model_and_tokenizer(self):
        """Get the trained model and tokenizer"""
        return self.model, self.tokenizer
