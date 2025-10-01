# pipelines/evaluator.py

import json
import time
import math
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.markdown import Markdown

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import evaluate
from tqdm import tqdm
from .fresh_model_eval import reload_fresh_model_for_evaluation, ensure_model_saved_for_evaluation
from .single_gpu_eval import _move_batch_to_device, _move_tensors_to_cpu

# Fix tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()

class ProductionEvaluator:
    """Production-grade evaluator with comprehensive metrics and OOM protection"""
    
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path("runs/humigence")
        
        # Evaluation configuration
        self.eval_prompts = self._load_eval_prompts()
        self.metrics = {}
        
        # Memory management
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        self.max_batch_size = config.get("per_device_eval_batch_size", 8)
        self.min_batch_size = 1
        
        # Initialize metrics
        self.rouge_metric = None
        self.bleu_metric = None
        self._load_metrics()
    
    def _load_metrics(self):
        """Load evaluation metrics"""
        try:
            self.rouge_metric = evaluate.load("rouge")
            self.bleu_metric = evaluate.load("bleu")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load ROUGE/BLEU metrics: {e}[/yellow]")
            console.print("[yellow]⚠️ Will use simplified metrics instead[/yellow]")
    
    def _clear_memory(self):
        """Clear GPU memory and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        console.print("[blue]🧹 Memory cleared before evaluation[/blue]")
    
    def _determine_eval_device(self) -> torch.device:
        """Determine the evaluation device strategy (single device for entire evaluation)"""
        # Option A (Preferred): Use GPU if available
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.device('cuda:0')
        # Option B (Fallback): Use CPU
        else:
            return torch.device('cpu')
    
    def _is_distributed(self) -> bool:
        """Check if running in distributed mode (DDP/FSDP)"""
        try:
            import torch.distributed as dist
            return dist.is_initialized() and dist.get_world_size() > 1
        except:
            return False
    
    def _gather_predictions_and_labels(self, all_predictions: List[torch.Tensor], 
                                     all_labels: List[torch.Tensor]) -> tuple:
        """Gather predictions and labels across all ranks for distributed evaluation"""
        if not self._is_distributed():
            return all_predictions, all_labels
        
        try:
            import torch.distributed as dist
            
            # Concatenate local predictions and labels
            local_predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([])
            local_labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
            
            # Gather from all ranks
            gathered_predictions = [torch.zeros_like(local_predictions) for _ in range(dist.get_world_size())]
            gathered_labels = [torch.zeros_like(local_labels) for _ in range(dist.get_world_size())]
            
            dist.all_gather(gathered_predictions, local_predictions)
            dist.all_gather(gathered_labels, local_labels)
            
            # Concatenate all gathered tensors
            all_gathered_predictions = torch.cat(gathered_predictions, dim=0)
            all_gathered_labels = torch.cat(gathered_labels, dim=0)
            
            # Convert back to list format for compatibility
            return [all_gathered_predictions], [all_gathered_labels]
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Distributed gathering failed: {e}[/yellow]")
            console.print("[yellow]⚠️ Using local predictions only[/yellow]")
            return all_predictions, all_labels
    
    def _get_safe_batch_size(self, dataset_size: int, initial_batch_size: int = None) -> int:
        """Determine safe batch size with OOM protection"""
        if initial_batch_size is None:
            initial_batch_size = self.max_batch_size
        
        # Start with the initial batch size and reduce if needed
        batch_size = min(initial_batch_size, dataset_size)
        
        # If we have a very small dataset, use the whole dataset
        if dataset_size <= 4:
            return dataset_size
        
        # Try to find a safe batch size by testing with a small sample
        test_size = min(4, dataset_size)
        try:
            # Test with a small batch to see if we can fit in memory
            test_batch = torch.randn(test_size, 1024, device=self.device)
            del test_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                console.print(f"[yellow]⚠️ GPU memory test failed, using conservative batch size[/yellow]")
                return min(2, dataset_size)
            else:
                raise e
        
    def _load_eval_prompts(self) -> List[Dict]:
        """Load evaluation prompts for qualitative assessment"""
        eval_prompts_path = self.output_dir / "eval_prompts.jsonl"
        
        if eval_prompts_path.exists():
            prompts = []
            with open(eval_prompts_path, 'r') as f:
                for line in f:
                    try:
                        prompts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return prompts
        else:
            # Default evaluation prompts
            return [
                {"instruction": "What is the capital of France?", "expected": "Paris"},
                {"instruction": "Explain quantum computing in simple terms.", "expected": "quantum"},
                {"instruction": "How do neural networks work?", "expected": "neural"},
                {"instruction": "What is machine learning?", "expected": "machine learning"},
                {"instruction": "Describe the process of photosynthesis.", "expected": "photosynthesis"}
            ]
    
    def evaluate_dataset(self, dataset: List[Dict], dataset_name: str) -> Dict:
        """Evaluate model on a dataset with device consistency and comprehensive metrics"""
        console.print(f"[blue]🧪 Evaluating on {dataset_name} dataset...[/blue]")
        
        # Clear memory before evaluation
        self._clear_memory()
        
        # Convert dataset to texts
        texts = self._convert_to_texts(dataset)
        
        if not texts:
            console.print(f"[yellow]⚠️ No valid texts found in {dataset_name} dataset[/yellow]")
            return {}
        
        # Determine evaluation device strategy (single device for entire evaluation)
        eval_device = self._determine_eval_device()
        console.print(f"[blue]🖥️ Using evaluation device: {eval_device}[/blue]")
        
        try:
            metrics = self._evaluate_with_device_consistency(texts, dataset_name, eval_device)
        except Exception as e:
            console.print(f"[red]❌ Evaluation failed: {e}[/red]")
            return self._get_fallback_metrics(dataset_name, len(texts))
        
        console.print(f"[green]✅ {dataset_name} evaluation complete[/green]")
        return metrics
    
    def _evaluate_with_device_consistency(self, texts: List[str], dataset_name: str, eval_device: torch.device) -> Dict:
        """Evaluate with masked scalar accumulation (no tensor concatenation)"""
        # Model should already be on the correct device from comprehensive_evaluation
        console.print(f"[blue]🖥️ Using model on device: {next(self.model.parameters()).device}[/blue]")
        
        # Determine safe batch size
        batch_size = self._get_safe_batch_size(len(texts))
        console.print(f"[blue]📊 Using batch size: {batch_size} on {eval_device}[/blue]")
        
        # Scalar accumulators (no tensor concatenation)
        correct_tokens = 0
        total_tokens = 0
        loss_sum = 0.0
        n_samples = 0
        
        # Process in batches with masked scalar accumulation
        dataloader = self._create_eval_dataloader(texts, batch_size)
        
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
            # Move batch to evaluation device
            batch = {k: v.to(eval_device) if hasattr(v, "to") else v for k, v in batch.items()}
            
            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits
                    predictions = logits.argmax(dim=-1)
                    labels = batch["labels"]
                    
                    # Mask out ignored positions (-100)
                    mask = labels != -100
                    
                    # Accumulate metrics (no tensor concatenation)
                    correct_tokens += (predictions[mask] == labels[mask]).sum().item()
                    total_tokens += mask.sum().item()
                    
                    # Weighted loss accumulation
                    loss_sum += loss.item() * labels.size(0)
                    n_samples += labels.size(0)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        console.print(f"[yellow]⚠️ OOM with batch size {batch_size}, reducing...[/yellow]")
                        batch_size = max(1, batch_size // 2)
                        console.print(f"[blue]🔄 Retrying with batch size: {batch_size}[/blue]")
                        continue
                    else:
                        raise e
            
            # Clear intermediate results to save memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics from scalar accumulators
        mean_loss = loss_sum / max(n_samples, 1)
        accuracy = correct_tokens / max(total_tokens, 1)
        perplexity = np.exp(mean_loss)
        
        metrics = {
            "loss": mean_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "correct_tokens": correct_tokens,
            "total_tokens": total_tokens,
            "n_samples": n_samples
        }
        
        # Add optional generation-based metrics
        eval_generate_samples = self.config.get("eval_generate_samples", 0)
        if eval_generate_samples > 0:
            generation_metrics = self._compute_generation_metrics(texts, eval_generate_samples, eval_device)
            metrics.update(generation_metrics)
        
        return metrics
    
    def _create_eval_dataloader(self, texts: List[str], batch_size: int):
        """Create evaluation dataloader with proper collation"""
        from torch.utils.data import DataLoader, Dataset
        
        class EvalDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return {"text": self.texts[idx]}
        
        def collate_fn(batch):
            texts = [item["text"] for item in batch]
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            # Add labels for language modeling
            encodings["labels"] = encodings["input_ids"].clone()
            return encodings
        
        dataset = EvalDataset(texts)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    def _compute_generation_metrics(self, texts: List[str], n_samples: int, eval_device: torch.device) -> Dict:
        """Compute BLEU/ROUGE metrics using model generation"""
        import random
        
        # Sample texts for generation
        sample_texts = random.sample(texts, min(n_samples, len(texts)))
        
        console.print(f"[blue]🎯 Computing generation metrics on {len(sample_texts)} samples[/blue]")
        
        generated_texts = []
        reference_texts = []
        
        for text in tqdm(sample_texts, desc="Generating", leave=False):
            # Extract input and target from text
            if "### Response:" in text:
                input_text = text.split("### Response:")[0].strip()
                target_text = text.split("### Response:")[1].strip()
            else:
                # Fallback: use first half as input, second half as target
                words = text.split()
                mid = len(words) // 2
                input_text = " ".join(words[:mid])
                target_text = " ".join(words[mid:])
            
            # Generate response
            inputs = self.tokenizer(input_text, return_tensors="pt").to(eval_device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(input_text):].strip()
            
            generated_texts.append(generated)
            reference_texts.append(target_text)
        
        # Compute BLEU/ROUGE
        metrics = {}
        
        try:
            if self.bleu_metric:
                bleu_scores = self.bleu_metric.compute(
                    predictions=generated_texts,
                    references=[[ref] for ref in reference_texts]
                )
                metrics["bleu"] = bleu_scores["bleu"]
        except Exception as e:
            console.print(f"[yellow]⚠️ BLEU computation failed: {e}[/yellow]")
        
        try:
            if self.rouge_metric:
                rouge_scores = self.rouge_metric.compute(
                    predictions=generated_texts,
                    references=reference_texts
                )
                metrics["rouge1"] = rouge_scores["rouge1"]
                metrics["rouge2"] = rouge_scores["rouge2"]
                metrics["rougeL"] = rouge_scores["rougeL"]
        except Exception as e:
            console.print(f"[yellow]⚠️ ROUGE computation failed: {e}[/yellow]")
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, losses: List[float], predictions: List[torch.Tensor], 
                                       labels: List[torch.Tensor], texts: List[str], dataset_name: str) -> Dict:
        """Calculate comprehensive evaluation metrics on CPU"""
        console.print(f"[blue]🧮 Computing metrics on CPU for {dataset_name}[/blue]")
        
        # Basic metrics
        avg_loss = np.mean(losses)
        perplexity = math.exp(avg_loss)
        
        # Ensure all tensors are on CPU before concatenation
        cpu_predictions = [pred.cpu() if pred.device.type != 'cpu' else pred for pred in predictions]
        cpu_labels = [label.cpu() if label.device.type != 'cpu' else label for label in labels]
        
        # Concatenate all predictions and labels (all on CPU)
        all_predictions = torch.cat(cpu_predictions, dim=0)
        all_labels = torch.cat(cpu_labels, dim=0)
        
        console.print(f"[blue]📊 Concatenated tensors on CPU: predictions={all_predictions.device}, labels={all_labels.device}[/blue]")
        
        # Calculate accuracy (all operations on CPU)
        mask = all_labels != self.tokenizer.pad_token_id
        correct_tokens = (all_predictions == all_labels) & mask
        accuracy = correct_tokens.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
        
        console.print(f"[blue]📊 Accuracy calculated: {accuracy:.4f}[/blue]")
        
        # Calculate BLEU and ROUGE scores (all on CPU)
        bleu_score = self._calculate_bleu_score_robust(texts, all_predictions, all_labels)
        rouge_scores = self._calculate_rouge_scores_robust(texts, all_predictions, all_labels)
        
        console.print(f"[blue]📊 BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_scores.get('rougeL', 0.0):.4f}[/blue]")
        
        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "bleu": bleu_score,
            "rouge": rouge_scores,
            "num_samples": len(texts),
            "device_used": "cpu"  # All metrics computed on CPU
        }
        
        return metrics
    
    def _get_fallback_metrics(self, dataset_name: str, num_samples: int) -> Dict:
        """Return fallback metrics when evaluation fails completely"""
        console.print(f"[yellow]⚠️ Using fallback metrics for {dataset_name}[/yellow]")
        return {
            "loss": 999.0,
            "perplexity": 999.0,
            "accuracy": 0.0,
            "bleu": 0.0,
            "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
            "num_samples": num_samples,
            "device_used": "fallback",
            "error": "Evaluation failed, using fallback metrics"
        }
    
    def _convert_to_texts(self, dataset: List[Dict]) -> List[str]:
        """Convert dataset samples to text format"""
        texts = []
        for sample in dataset:
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
    
    def _calculate_bleu_score_robust(self, texts: List[str], predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate robust BLEU score with proper error handling (CPU tensors only)"""
        try:
            # Ensure tensors are on CPU
            predictions = predictions.cpu() if predictions.device.type != 'cpu' else predictions
            labels = labels.cpu() if labels.device.type != 'cpu' else labels
            
            if self.bleu_metric is not None:
                # Use the evaluate library for proper BLEU calculation
                predictions_text = []
                references_text = []
                
                for i, text in enumerate(texts):
                    # Get predicted tokens (CPU tensor operations)
                    pred_tokens = predictions[i][labels[i] != self.tokenizer.pad_token_id]
                    pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    
                    # Get reference text (last part after "### Response:")
                    if "### Response:" in text:
                        ref_text = text.split("### Response:")[-1].strip()
                    else:
                        ref_text = text
                    
                    predictions_text.append(pred_text)
                    references_text.append([ref_text])  # BLEU expects list of references
                
                # Calculate BLEU score
                bleu_result = self.bleu_metric.compute(
                    predictions=predictions_text,
                    references=references_text
                )
                return bleu_result.get('bleu', 0.0)
            else:
                # Fallback to simplified calculation
                return self._calculate_bleu_score_simple(texts, predictions, labels)
        except Exception as e:
            console.print(f"[yellow]⚠️ BLEU calculation failed: {e}[/yellow]")
            return self._calculate_bleu_score_simple(texts, predictions, labels)
    
    def _calculate_bleu_score_simple(self, texts: List[str], predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate simplified BLEU score as fallback"""
        try:
            total_score = 0.0
            for i, text in enumerate(texts):
                # Get predicted tokens
                pred_tokens = predictions[i][labels[i] != self.tokenizer.pad_token_id]
                pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                
                # Get reference text (last part after "### Response:")
                if "### Response:" in text:
                    ref_text = text.split("### Response:")[-1].strip()
                else:
                    ref_text = text
                
                # Simple word overlap
                pred_words = set(pred_text.lower().split())
                ref_words = set(ref_text.lower().split())
                
                if len(ref_words) > 0:
                    overlap = len(pred_words & ref_words)
                    score = overlap / len(ref_words)
                    total_score += score
            
            return total_score / len(texts) if texts else 0.0
        except:
            return 0.0
    
    def _calculate_rouge_scores_robust(self, texts: List[str], predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate robust ROUGE scores with proper error handling (CPU tensors only)"""
        try:
            # Ensure tensors are on CPU
            predictions = predictions.cpu() if predictions.device.type != 'cpu' else predictions
            labels = labels.cpu() if labels.device.type != 'cpu' else labels
            
            if self.rouge_metric is not None:
                # Use the evaluate library for proper ROUGE calculation
                predictions_text = []
                references_text = []
                
                for i, text in enumerate(texts):
                    # Get predicted tokens (CPU tensor operations)
                    pred_tokens = predictions[i][labels[i] != self.tokenizer.pad_token_id]
                    pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    
                    # Get reference text (last part after "### Response:")
                    if "### Response:" in text:
                        ref_text = text.split("### Response:")[-1].strip()
                    else:
                        ref_text = text
                    
                    predictions_text.append(pred_text)
                    references_text.append(ref_text)
                
                # Calculate ROUGE scores
                rouge_result = self.rouge_metric.compute(
                    predictions=predictions_text,
                    references=references_text
                )
                return {
                    "rouge1": rouge_result.get('rouge1', 0.0),
                    "rouge2": rouge_result.get('rouge2', 0.0),
                    "rougeL": rouge_result.get('rougeL', 0.0)
                }
            else:
                # Fallback to simplified calculation
                return self._calculate_rouge_scores_simple(texts, predictions, labels)
        except Exception as e:
            console.print(f"[yellow]⚠️ ROUGE calculation failed: {e}[/yellow]")
            return self._calculate_rouge_scores_simple(texts, predictions, labels)
    
    def _calculate_rouge_scores_simple(self, texts: List[str], predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate simplified ROUGE scores as fallback"""
        try:
            rouge1_scores = []
            rouge2_scores = []
            
            for i, text in enumerate(texts):
                # Get predicted tokens
                pred_tokens = predictions[i][labels[i] != self.tokenizer.pad_token_id]
                pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                
                # Get reference text
                if "### Response:" in text:
                    ref_text = text.split("### Response:")[-1].strip()
                else:
                    ref_text = text
                
                # ROUGE-1 (unigram overlap)
                pred_unigrams = set(pred_text.lower().split())
                ref_unigrams = set(ref_text.lower().split())
                
                if len(ref_unigrams) > 0:
                    rouge1 = len(pred_unigrams & ref_unigrams) / len(ref_unigrams)
                    rouge1_scores.append(rouge1)
                
                # ROUGE-2 (bigram overlap)
                pred_bigrams = set(zip(pred_text.lower().split()[:-1], pred_text.lower().split()[1:]))
                ref_bigrams = set(zip(ref_text.lower().split()[:-1], ref_text.lower().split()[1:]))
                
                if len(ref_bigrams) > 0:
                    rouge2 = len(pred_bigrams & ref_bigrams) / len(ref_bigrams)
                    rouge2_scores.append(rouge2)
            
            return {
                "rouge1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
                "rouge2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
                "rougeL": np.mean(rouge1_scores) if rouge1_scores else 0.0  # Simplified
            }
        except:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def run_qualitative_evaluation(self) -> List[Dict]:
        """Run qualitative evaluation with sample prompts"""
        console.print("[blue]🎯 Running qualitative evaluation...[/blue]")
        
        results = []
        
        for i, prompt in enumerate(self.eval_prompts):
            instruction = prompt["instruction"]
            expected = prompt.get("expected", "")
            
            # Generate response
            input_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(input_text, "").strip()
            
            # Check if expected content is present
            contains_expected = expected.lower() in response.lower() if expected else True
            
            result = {
                "prompt": instruction,
                "response": response,
                "expected": expected,
                "contains_expected": contains_expected
            }
            results.append(result)
            
            console.print(f"[blue]📌 Prompt {i+1}: {instruction}[/blue]")
            console.print(f"[green]🤖 Response: {response[:100]}{'...' if len(response) > 100 else ''}[/green]")
            console.print(f"[{'green' if contains_expected else 'red'}]✅ Expected content: {'Found' if contains_expected else 'Not found'}[/{'green' if contains_expected else 'red'}]")
        
        return results
    
    def comprehensive_evaluation(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Dict:
        """Run comprehensive evaluation on all datasets with OOM protection"""
        console.print("\n[bold cyan]🧪 COMPREHENSIVE EVALUATION[/bold cyan]")
        
        # Show dataset split information
        console.print(f"[blue]📊 Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}[/blue]")
        
        # Ensure model is saved for evaluation
        if not ensure_model_saved_for_evaluation(self.config):
            raise RuntimeError("Model not saved for evaluation. Make sure training completed successfully.")
        
        # Reload fresh model for evaluation (fixes cuda:0 vs cuda:1 mismatch)
        self.model, self.tokenizer = reload_fresh_model_for_evaluation(self.config)
        
        # Determine evaluation device
        eval_device = next(self.model.parameters()).device
        console.print(f"[blue]🖥️ Evaluation device: {eval_device}[/blue]")
        
        # Clear memory before starting evaluation
        self._clear_memory()
        
        # Evaluate on validation set
        val_metrics = self.evaluate_dataset(val_data, "validation")
        
        # Clear memory between evaluations
        self._clear_memory()
        
        # Evaluate on test set
        test_metrics = self.evaluate_dataset(test_data, "test")
        
        # Clear memory before qualitative evaluation
        self._clear_memory()
        
        # Run qualitative evaluation
        qualitative_results = self.run_qualitative_evaluation()
        
        # Calculate overfitting metrics (use smaller sample for training evaluation)
        train_sample = train_data[:min(100, len(train_data))]  # Limit training evaluation to avoid OOM
        train_metrics = self.evaluate_dataset(train_sample, "training")
        
        # Calculate train-val loss gap
        train_loss = train_metrics.get("loss", 0.0)
        val_loss = val_metrics.get("loss", 0.0)
        loss_gap = train_loss - val_loss
        
        # Determine overfitting status
        overfitting_threshold = 0.5
        is_overfitting = loss_gap > overfitting_threshold
        
        # Calculate relevance score (based on qualitative results)
        qualitative_score = sum(1 for r in qualitative_results if r["contains_expected"]) / len(qualitative_results) if qualitative_results else 0.0
        
        # Compile comprehensive results
        comprehensive_results = {
            "validation": val_metrics,
            "test": test_metrics,
            "training": train_metrics,
            "overfitting": {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_gap": loss_gap,
                "is_overfitting": is_overfitting,
                "threshold": overfitting_threshold
            },
            "qualitative": {
                "results": qualitative_results,
                "score": qualitative_score,
                "total_prompts": len(qualitative_results)
            },
            "overall_assessment": self._generate_assessment(val_metrics, test_metrics, is_overfitting, qualitative_score)
        }
        
        # Display results
        self._display_evaluation_results(comprehensive_results)
        
        # Create and save evaluation summary
        self._save_evaluation_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _generate_assessment(self, val_metrics: Dict, test_metrics: Dict, is_overfitting: bool, qualitative_score: float) -> str:
        """Generate natural language assessment of model performance"""
        val_loss = val_metrics.get("loss", 0.0)
        test_loss = test_metrics.get("loss", 0.0)
        
        if val_loss < 1.0 and not is_overfitting and qualitative_score > 0.8:
            return "EXCELLENT: Model shows strong performance with low loss, no overfitting, and high qualitative scores."
        elif val_loss < 2.0 and not is_overfitting and qualitative_score > 0.6:
            return "GOOD: Model performs well with reasonable loss and good qualitative results."
        elif val_loss < 3.0 and qualitative_score > 0.4:
            return "FAIR: Model shows moderate performance with room for improvement."
        elif is_overfitting:
            return "POOR: Model shows signs of overfitting. Consider more data or regularization."
        else:
            return "POOR: Model performance is below expectations. Consider more training or data."
    
    def _display_evaluation_results(self, results: Dict):
        """Display clean evaluation results summary"""
        console.print("\n[bold cyan]=" * 80)
        console.print("[bold cyan]📊 EVALUATION RESULTS SUMMARY[/bold cyan]")
        console.print("[bold cyan]=" * 80)
        
        # Validation results
        val_metrics = results["validation"]
        console.print(f"\n[bold green]✅ VALIDATION SET[/bold green]")
        console.print(f"   Loss: {val_metrics.get('loss', 'N/A'):.4f}")
        console.print(f"   Perplexity: {val_metrics.get('perplexity', 'N/A'):.2f}")
        console.print(f"   Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")
        console.print(f"   Samples: {val_metrics.get('n_samples', 'N/A')}")
        
        # Test results
        test_metrics = results["test"]
        console.print(f"\n[bold blue]🧪 TEST SET[/bold blue]")
        console.print(f"   Loss: {test_metrics.get('loss', 'N/A'):.4f}")
        console.print(f"   Perplexity: {test_metrics.get('perplexity', 'N/A'):.2f}")
        console.print(f"   Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
        console.print(f"   Samples: {test_metrics.get('n_samples', 'N/A')}")
        
        # Generation metrics (if available)
        if "bleu" in val_metrics or "rouge1" in val_metrics:
            console.print(f"\n[bold magenta]🎯 GENERATION METRICS[/bold magenta]")
            if "bleu" in val_metrics:
                console.print(f"   BLEU: {val_metrics['bleu']:.4f}")
            if "rouge1" in val_metrics:
                console.print(f"   ROUGE-1: {val_metrics['rouge1']:.4f}")
            if "rouge2" in val_metrics:
                console.print(f"   ROUGE-2: {val_metrics['rouge2']:.4f}")
            if "rougeL" in val_metrics:
                console.print(f"   ROUGE-L: {val_metrics['rougeL']:.4f}")
        
        # Overfitting analysis
        overfitting = results["overfitting"]
        console.print(f"\n[bold yellow]📈 OVERFITTING ANALYSIS[/bold yellow]")
        console.print(f"   Train Loss: {overfitting['train_loss']:.4f}")
        console.print(f"   Val Loss: {overfitting['val_loss']:.4f}")
        console.print(f"   Loss Gap: {overfitting['loss_gap']:.4f}")
        console.print(f"   Status: {'OVERFITTING' if overfitting['is_overfitting'] else 'HEALTHY'}")
        
        # Overall assessment
        console.print(f"\n[bold red]🎯 OVERALL ASSESSMENT[/bold red]")
        console.print(f"   {results['overall_assessment']}")
        
        console.print("\n[bold cyan]=" * 80)
        
        # Qualitative results
        qualitative = results["qualitative"]
        console.print(f"\n[bold]Qualitative Assessment:[/bold]")
        console.print(f"  Score: [cyan]{qualitative['score']:.3f}[/cyan] ({qualitative['total_prompts']} prompts)")
        
        # Overall assessment
        console.print(f"\n[bold]Overall Assessment:[/bold]")
        assessment = results["overall_assessment"]
        color = "green" if "EXCELLENT" in assessment or "GOOD" in assessment else "yellow" if "FAIR" in assessment else "red"
        console.print(f"[{color}]{assessment}[/{color}]")
        
        # Display human-friendly summary
        self._display_human_friendly_summary(results)
        
        # Display end-of-run summary
        self._display_end_of_run_summary(results)
    
    def _display_end_of_run_summary(self, results: Dict):
        """Display end-of-run summary with training loss, validation loss, and metrics"""
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]📊 END-OF-RUN SUMMARY[/bold cyan]")
        console.print("=" * 60)
        
        # Extract key metrics
        train_loss = results["training"].get("loss", 0.0)
        val_loss = results["validation"].get("loss", 0.0)
        test_loss = results["test"].get("loss", 0.0)
        val_accuracy = results["validation"].get("accuracy", 0.0)
        val_bleu = results["validation"].get("bleu", 0.0)
        val_rouge_l = results["validation"].get("rouge", {}).get("rougeL", 0.0)
        num_val_samples = results["validation"].get("num_samples", 0)
        num_test_samples = results["test"].get("num_samples", 0)
        
        # Display metrics summary
        console.print(f"[blue]📈 Training Loss: {train_loss:.4f}[/blue]")
        console.print(f"[blue]📈 Validation Loss: {val_loss:.4f}[/blue]")
        console.print(f"[blue]📈 Test Loss: {test_loss:.4f}[/blue]")
        console.print(f"[blue]📊 Validation Accuracy: {val_accuracy:.3f}[/blue]")
        console.print(f"[blue]📊 Validation BLEU: {val_bleu:.3f}[/blue]")
        console.print(f"[blue]📊 Validation ROUGE-L: {val_rouge_l:.3f}[/blue]")
        console.print(f"[blue]📊 Validation Samples: {num_val_samples}[/blue]")
        console.print(f"[blue]📊 Test Samples: {num_test_samples}[/blue]")
        
        # Overfitting status
        overfitting = results["overfitting"]
        if overfitting["is_overfitting"]:
            console.print(f"[red]⚠️ Overfitting Detected: Train-Val gap = {overfitting['loss_gap']:.4f}[/red]")
        else:
            console.print(f"[green]✅ No Overfitting: Train-Val gap = {overfitting['loss_gap']:.4f}[/green]")
        
        console.print("=" * 60)
    
    def _display_human_friendly_summary(self, results: Dict):
        """Display human-friendly evaluation summary in Markdown format"""
        val_metrics = results["validation"]
        test_metrics = results["test"]
        overfitting = results["overfitting"]
        qualitative = results["qualitative"]
        
        # Create summary data
        train_loss = results["training"].get("loss", 0.0)
        val_loss = val_metrics.get("loss", 0.0)
        test_loss = test_metrics.get("loss", 0.0)
        perplexity = val_metrics.get("perplexity", 0.0)
        accuracy = val_metrics.get("accuracy", 0.0)
        rouge_l = val_metrics.get("rouge", {}).get("rougeL", 0.0)
        is_overfitting = overfitting["is_overfitting"]
        
        # Create Markdown summary
        summary_md = f"""# 📊 Evaluation Summary

## Key Metrics
- **Train Loss:** {train_loss:.4f}
- **Val Loss:** {val_loss:.4f}
- **Test Loss:** {test_loss:.4f}
- **Perplexity:** {perplexity:.2f}
- **Accuracy:** {accuracy:.3f}
- **ROUGE-L:** {rouge_l:.3f}
- **Overfitting:** {'Yes' if is_overfitting else 'No'}

## Quality Assessment
{results["overall_assessment"]}

## Device Information
- **Validation Device:** {val_metrics.get('device_used', 'Unknown')}
- **Test Device:** {test_metrics.get('device_used', 'Unknown')}
"""
        
        # Display the summary
        console.print("\n" + "=" * 60)
        console.print(Markdown(summary_md))
        console.print("=" * 60)
    
    def _save_evaluation_summary(self, results: Dict):
        """Save evaluation summary to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_dir = self.output_dir / "eval_summary"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary
        json_path = eval_dir / f"evaluation_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save Markdown summary
        md_path = eval_dir / f"evaluation_summary_{timestamp}.md"
        val_metrics = results["validation"]
        test_metrics = results["test"]
        overfitting = results["overfitting"]
        qualitative = results["qualitative"]
        
        train_loss = results["training"].get("loss", 0.0)
        val_loss = val_metrics.get("loss", 0.0)
        test_loss = test_metrics.get("loss", 0.0)
        perplexity = val_metrics.get("perplexity", 0.0)
        accuracy = val_metrics.get("accuracy", 0.0)
        rouge_l = val_metrics.get("rouge", {}).get("rougeL", 0.0)
        is_overfitting = overfitting["is_overfitting"]
        
        markdown_content = f"""# Evaluation Summary

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## Key Metrics

| Metric | Value |
|--------|-------|
| Train Loss | {train_loss:.4f} |
| Val Loss | {val_loss:.4f} |
| Test Loss | {test_loss:.4f} |
| Perplexity | {perplexity:.2f} |
| Accuracy | {accuracy:.3f} |
| ROUGE-L | {rouge_l:.3f} |
| Overfitting | {'Yes' if is_overfitting else 'No'} |

## Detailed Results

### Validation Set
- **Loss:** {val_loss:.4f}
- **Perplexity:** {perplexity:.2f}
- **Accuracy:** {accuracy:.3f}
- **BLEU:** {val_metrics.get('bleu', 0.0):.3f}
- **ROUGE-1:** {val_metrics.get('rouge', {}).get('rouge1', 0.0):.3f}
- **ROUGE-2:** {val_metrics.get('rouge', {}).get('rouge2', 0.0):.3f}
- **ROUGE-L:** {rouge_l:.3f}

### Test Set
- **Loss:** {test_loss:.4f}
- **Perplexity:** {test_metrics.get('perplexity', 0.0):.2f}
- **Accuracy:** {test_metrics.get('accuracy', 0.0):.3f}

### Overfitting Analysis
- **Train-Val Loss Gap:** {overfitting['loss_gap']:.4f}
- **Overfitting Detected:** {'Yes' if is_overfitting else 'No'}
- **Threshold:** {overfitting['threshold']}

### Qualitative Assessment
- **Score:** {qualitative['score']:.3f}
- **Total Prompts:** {qualitative['total_prompts']}

## Overall Assessment

{results["overall_assessment"]}

## Device Information
- **Validation Device:** {val_metrics.get('device_used', 'Unknown')}
- **Test Device:** {test_metrics.get('device_used', 'Unknown')}
"""
        
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        
        console.print(f"[blue]📄 Evaluation summary saved to: {eval_dir}[/blue]")
        console.print(f"[blue]  • JSON: {json_path.name}[/blue]")
        console.print(f"[blue]  • Markdown: {md_path.name}[/blue]")
    
    def offer_ai_analysis(self, results: Dict) -> Optional[str]:
        """Offer AI-powered analysis and recommendations"""
        console.print("\n[bold yellow]Would you like AI analysis and recommendations?[/bold yellow]")
        
        choice = console.input("[bold blue]Run AI analysis? (y/N)[/bold blue]: ").strip().lower()
        
        if choice in ['y', 'yes']:
            console.print("[blue]🤖 Running AI analysis...[/blue]")
            
            # Generate AI analysis
            analysis = self._generate_ai_analysis(results)
            
            # Display analysis
            console.print("\n[bold cyan]AI Analysis and Recommendations:[/bold cyan]")
            console.print(f"[blue]{analysis}[/blue]")
            
            # Save analysis
            analysis_path = self.output_dir / "ai_analysis.txt"
            with open(analysis_path, 'w') as f:
                f.write(f"AI Analysis - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(analysis)
            
            console.print(f"[green]✅ AI analysis saved to: {analysis_path}[/green]")
            return analysis
        
        return None
    
    def _generate_ai_analysis(self, results: Dict) -> str:
        """Generate AI-powered analysis of model performance"""
        val_loss = results["validation"].get("loss", 0.0)
        test_loss = results["test"].get("loss", 0.0)
        is_overfitting = results["overfitting"]["is_overfitting"]
        qualitative_score = results["qualitative"]["score"]
        
        analysis_parts = []
        
        # Loss analysis
        if val_loss < 1.0:
            analysis_parts.append("✅ Validation loss is excellent (< 1.0), indicating strong model performance.")
        elif val_loss < 2.0:
            analysis_parts.append("✅ Validation loss is good (< 2.0), showing solid model performance.")
        elif val_loss < 3.0:
            analysis_parts.append("⚠️ Validation loss is moderate (2.0-3.0). Consider more training or data augmentation.")
        else:
            analysis_parts.append("❌ Validation loss is high (> 3.0). Model needs significant improvement.")
        
        # Overfitting analysis
        if is_overfitting:
            analysis_parts.append("⚠️ Overfitting detected. Recommendations: 1) Add more training data, 2) Increase regularization, 3) Reduce model complexity, 4) Use early stopping.")
        else:
            analysis_parts.append("✅ No overfitting detected. Model generalizes well to validation data.")
        
        # Qualitative analysis
        if qualitative_score > 0.8:
            analysis_parts.append("✅ Qualitative evaluation shows excellent performance on diverse prompts.")
        elif qualitative_score > 0.6:
            analysis_parts.append("✅ Qualitative evaluation shows good performance with room for improvement.")
        elif qualitative_score > 0.4:
            analysis_parts.append("⚠️ Qualitative evaluation shows moderate performance. Consider more diverse training data.")
        else:
            analysis_parts.append("❌ Qualitative evaluation shows poor performance. Model needs significant improvement.")
        
        # General recommendations
        if val_loss > 2.0 or is_overfitting or qualitative_score < 0.6:
            analysis_parts.append("\n🔧 Recommended Actions:")
            analysis_parts.append("1. Increase training data diversity and quantity")
            analysis_parts.append("2. Adjust learning rate (try 1e-5 to 5e-4)")
            analysis_parts.append("3. Experiment with different model architectures")
            analysis_parts.append("4. Use data augmentation techniques")
            analysis_parts.append("5. Implement early stopping to prevent overfitting")
        
        return "\n".join(analysis_parts)
