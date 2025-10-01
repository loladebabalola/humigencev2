# pipelines/distributed_trainer.py

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from distributed_utils import RankZeroOnly
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class DistributedProductionTrainer:
    """Production trainer with proper distributed training support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path("runs/humigence")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distributed training setup
        self.ddp = config.get("ddp", False)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)
        self.is_main = config.get("is_main", True)
        self.device = torch.device(config.get("device", "cuda:0"))
        
        # Training configuration
        self.base_model = config["base_model"]
        self.training_recipe = config["training_recipe"]
        self.learning_rate = float(config.get("learning_rate", "2e-4"))
        self.num_epochs = int(config.get("num_train_epochs", "1"))
        self.batch_size = int(config.get("per_device_train_batch_size", "2"))
        self.gradient_accumulation = int(config.get("gradient_accumulation_steps", "4"))
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with proper device placement"""
        with RankZeroOnly(self.is_main) as rank_zero:
            rank_zero.print(f"[blue]🤖 Loading model: {self.base_model}[/blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with proper device placement
        if self.ddp:
            # For DDP, load model to CPU first, then move to device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map=None,  # Load to CPU
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if "BF16" in self.training_recipe else torch.float16
            )
            # Move to device
            self.model = self.model.to(self.device)
        else:
            # Single GPU training
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if "BF16" in self.training_recipe else torch.float16
            )
        
        # Apply LoRA if needed
        if "LoRA" in self.training_recipe:
            self._apply_lora()
        
        # Setup DDP if needed
        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index
            )
            with RankZeroOnly(self.is_main) as rank_zero:
                rank_zero.print(f"[blue]✅ Model wrapped with DDP (rank {self.rank})[/blue]")
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        with RankZeroOnly(self.is_main) as rank_zero:
            rank_zero.print("[blue]✅ LoRA configuration applied[/blue]")
    
    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Prepare datasets with distributed sampling"""
        # Convert to datasets
        from datasets import Dataset
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Create data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create distributed samplers if needed
        if self.ddp:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            ddp_find_unused_parameters=False,  # Important for DDP
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Set samplers for distributed training
        if self.ddp:
            self.trainer.train_dataloader.sampler = self.train_sampler
            self.trainer.eval_dataloader.sampler = self.val_sampler
    
    def train(self):
        """Run training with proper distributed handling"""
        with RankZeroOnly(self.is_main) as rank_zero:
            rank_zero.print("[blue]🚀 Starting training...[/blue]")
        
        # Train the model
        self.trainer.train()
        
        # Save model (only on main process)
        if self.is_main:
            self.trainer.save_model()
            with RankZeroOnly(self.is_main) as rank_zero:
                rank_zero.print("[blue]💾 Model saved[/blue]")
        
        # Synchronize all processes
        if self.ddp:
            dist.barrier()
        
        return {"status": "success", "message": "Training completed"}
    
    def evaluate(self):
        """Run evaluation with proper distributed handling"""
        with RankZeroOnly(self.is_main) as rank_zero:
            rank_zero.print("[blue]🧪 Running evaluation...[/blue]")
        
        # Run evaluation
        eval_results = self.trainer.evaluate()
        
        # Gather results from all ranks if DDP
        if self.ddp:
            # Gather evaluation results
            gathered_results = [None] * self.world_size
            dist.all_gather_object(gathered_results, eval_results)
            
            # Average results across ranks
            if self.is_main:
                avg_results = {}
                for key in eval_results.keys():
                    if isinstance(eval_results[key], (int, float)):
                        values = [r[key] for r in gathered_results if r is not None]
                        avg_results[key] = sum(values) / len(values)
                    else:
                        avg_results[key] = eval_results[key]
                eval_results = avg_results
        
        return eval_results
