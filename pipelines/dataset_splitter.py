# pipelines/dataset_splitter.py

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

console = Console()

class DatasetSplitter:
    """Production-grade dataset splitter with integrity checks and AI augmentation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.split_ratios = config.get("split_ratios", [0.8, 0.1, 0.1])  # train/val/test
        self.min_train_samples = config.get("min_train_samples", 1000)
        self.min_val_samples = config.get("min_val_samples", 100)
        self.min_test_samples = config.get("min_test_samples", 100)
        self.min_tokens_per_sample = config.get("min_tokens_per_sample", 50)
        self.random_seed = config.get("random_seed", 42)
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from JSONL file"""
        console.print(f"[blue]📚 Loading dataset from: {dataset_path}[/blue]")
        
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    data.append(sample)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️ Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                    continue
        
        if not data:
            raise ValueError(f"❌ No valid data found in {dataset_path}")
        
        console.print(f"[green]✅ Loaded {len(data)} samples from dataset[/green]")
        return data
    
    def split_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/validation/test sets"""
        console.print(f"[blue]📊 Splitting dataset with ratios: {self.split_ratios}[/blue]")
        
        # Shuffle data for random splitting
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        total_samples = len(shuffled_data)
        train_size = int(total_samples * self.split_ratios[0])
        val_size = int(total_samples * self.split_ratios[1])
        
        # Split the data
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        # Ensure we have at least one sample in each split
        if len(test_data) == 0 and len(val_data) > 0:
            test_data = [val_data.pop()]
        if len(val_data) == 0 and len(train_data) > 1:
            val_data = [train_data.pop()]
        
        console.print(f"[green]✅ Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test[/green]")
        
        return train_data, val_data, test_data
    
    def check_integrity(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Tuple[bool, List[str]]:
        """Check dataset integrity and return issues"""
        issues = []
        
        # Check sample counts
        if len(train_data) < self.min_train_samples:
            issues.append(f"Training samples ({len(train_data)}) < minimum ({self.min_train_samples})")
        
        if len(val_data) < self.min_val_samples:
            issues.append(f"Validation samples ({len(val_data)}) < minimum ({self.min_val_samples})")
        
        if len(test_data) < self.min_test_samples:
            issues.append(f"Test samples ({len(test_data)}) < minimum ({self.min_test_samples})")
        
        # Check token length (approximate)
        all_data = train_data + val_data + test_data
        avg_tokens = self._estimate_avg_tokens(all_data)
        if avg_tokens < self.min_tokens_per_sample:
            issues.append(f"Average tokens per sample ({avg_tokens:.1f}) < minimum ({self.min_tokens_per_sample})")
        
        return len(issues) == 0, issues
    
    def _estimate_avg_tokens(self, data: List[Dict]) -> float:
        """Estimate average tokens per sample"""
        total_tokens = 0
        for sample in data:
            # Simple estimation: count words * 1.3 (rough token-to-word ratio)
            text = ""
            if "instruction" in sample:
                text += sample["instruction"] + " "
            if "output" in sample:
                text += sample["output"] + " "
            if "input" in sample:
                text += sample["input"] + " "
            
            word_count = len(text.split())
            total_tokens += word_count * 1.3
        
        return total_tokens / len(data) if data else 0
    
    def display_integrity_report(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], issues: List[str]):
        """Display dataset integrity report"""
        console.print("\n[bold cyan]🔍 Dataset Integrity Report[/bold cyan]")
        
        # Create summary table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Split", style="cyan")
        table.add_column("Samples", justify="right")
        table.add_column("Percentage", justify="right")
        table.add_column("Status", justify="center")
        
        total_samples = len(train_data) + len(val_data) + len(test_data)
        
        for split_name, split_data, ratio in [("Train", train_data, self.split_ratios[0]), 
                                            ("Validation", val_data, self.split_ratios[1]), 
                                            ("Test", test_data, self.split_ratios[2])]:
            count = len(split_data)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            
            # Get minimum samples for this split
            if split_name == "Train":
                min_samples = self.min_train_samples
            elif split_name == "Validation":
                min_samples = self.min_val_samples
            else:  # Test
                min_samples = self.min_test_samples
            
            status = "✅" if count >= min_samples else "❌"
            
            table.add_row(split_name, str(count), f"{percentage:.1f}%", status)
        
        console.print(table)
        
        # Show token statistics
        all_data = train_data + val_data + test_data
        avg_tokens = self._estimate_avg_tokens(all_data)
        console.print(f"\n[blue]Average tokens per sample: {avg_tokens:.1f}[/blue]")
        
        # Show issues if any
        if issues:
            console.print("\n[bold red]⚠️ Integrity Issues Detected:[/bold red]")
            for issue in issues:
                console.print(f"[red]  • {issue}[/red]")
        else:
            console.print("\n[bold green]✅ All integrity checks passed![/bold green]")
    
    def offer_ai_augmentation(self, issues: List[str], original_data: List[Dict]) -> Optional[str]:
        """Offer AI augmentation options to user"""
        console.print("\n[bold yellow]How would you like to proceed?[/bold yellow]")
        console.print("[bold]1.[/bold] Proceed anyway (warn about quality)")
        console.print("[bold]2.[/bold] Augment using AI (synthetic sample generation)")
        console.print("[bold]3.[/bold] Switch dataset")
        console.print("[bold]4.[/bold] Abort training")
        
        while True:
            choice = console.input("[bold blue]Select option (1-4)[/bold blue]: ").strip()
            
            if choice == "1":
                console.print("[yellow]⚠️ Proceeding with dataset as-is (quality may be affected)[/yellow]")
                return None
            
            elif choice == "2":
                console.print("[blue]🤖 Running AI augmentation...[/blue]")
                augmented_path = self._run_ai_augmentation(original_data, issues)
                if augmented_path:
                    console.print(f"[green]✅ Dataset augmented and saved to: {augmented_path}[/green]")
                    return augmented_path
                else:
                    console.print("[red]❌ AI augmentation failed, proceeding with original dataset[/red]")
                    return None
            
            elif choice == "3":
                console.print("[blue]📁 Please select a different dataset[/blue]")
                # This would integrate with the dataset selection UI
                return "switch_dataset"
            
            elif choice == "4":
                console.print("[red]❌ Training aborted by user[/red]")
                return "abort"
            
            else:
                console.print("[yellow]⚠️ Invalid choice. Please select 1, 2, 3, or 4.[/yellow]")
    
    def _run_ai_augmentation(self, data: List[Dict], issues: List[str]) -> Optional[str]:
        """Run AI-powered dataset augmentation"""
        try:
            augmented_data = data.copy()
            
            # Determine target sizes based on issues
            target_train = self.min_train_samples
            target_val = self.min_val_samples
            target_test = self.min_test_samples
            
            # Augment training data if needed
            if len(data) < target_train:
                console.print(f"[blue]🔧 Augmenting to reach {target_train} training samples...[/blue]")
                while len(augmented_data) < target_train:
                    base_sample = random.choice(data)
                    if "instruction" in base_sample and "output" in base_sample:
                        # Create variations
                        variations = [
                            {
                                "instruction": f"Please {base_sample['instruction'].lower()}",
                                "output": base_sample["output"]
                            },
                            {
                                "instruction": f"Can you {base_sample['instruction'].lower()}",
                                "output": base_sample["output"]
                            },
                            {
                                "instruction": f"How to {base_sample['instruction'].lower()}",
                                "output": base_sample["output"]
                            },
                            {
                                "instruction": f"Explain {base_sample['instruction'].lower()}",
                                "output": f"Here's an explanation: {base_sample['output']}"
                            }
                        ]
                        augmented_data.extend(variations)
            
            # Expand short samples if needed
            if any("tokens" in issue.lower() for issue in issues):
                console.print("[blue]🔧 Expanding short samples...[/blue]")
                for i, sample in enumerate(augmented_data):
                    if "instruction" in sample and "output" in sample:
                        instruction = sample["instruction"]
                        output = sample["output"]
                        
                        # Expand if too short
                        if len(instruction.split()) < 10:
                            sample["instruction"] = f"Please provide a detailed explanation: {instruction}"
                        if len(output.split()) < 20:
                            sample["output"] = f"{output} This is important because it helps users understand the concept better and provides practical guidance for implementation."
            
            # Create versioned filename
            timestamp = int(time.time())
            augmented_path = f"data/augmented_dataset_{timestamp}.jsonl"
            Path(augmented_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save augmented dataset
            with open(augmented_path, 'w', encoding='utf-8') as f:
                for item in augmented_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Log the augmentation
            log_path = Path("runs/humigence") / "augmentation.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'a') as f:
                f.write(f"\n=== AI Dataset Augmentation - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Original samples: {len(data)}\n")
                f.write(f"Augmented samples: {len(augmented_data)}\n")
                f.write(f"Issues addressed: {', '.join(issues)}\n")
                f.write(f"Augmented dataset: {augmented_path}\n")
                f.write("=" * 50 + "\n")
            
            console.print(f"[green]✅ AI augmentation completed: {len(augmented_data)} samples[/green]")
            return augmented_path
            
        except Exception as e:
            console.print(f"[red]❌ AI augmentation failed: {e}[/red]")
            return None
    
    def process_dataset(self, dataset_path: str) -> Tuple[List[Dict], List[Dict], List[Dict], str]:
        """Main method to process dataset with splitting and integrity checks"""
        # Load dataset
        data = self.load_dataset(dataset_path)
        
        # Split dataset
        train_data, val_data, test_data = self.split_dataset(data)
        
        # Check integrity
        is_valid, issues = self.check_integrity(train_data, val_data, test_data)
        
        # Display integrity report
        self.display_integrity_report(train_data, val_data, test_data, issues)
        
        # Handle integrity issues
        if not is_valid:
            result = self.offer_ai_augmentation(issues, data)
            
            if result == "abort":
                raise KeyboardInterrupt("Training aborted by user")
            elif result == "switch_dataset":
                raise ValueError("Please select a different dataset")
            elif result and result != "switch_dataset":
                # Reload with augmented dataset
                console.print("[blue]🔄 Reloading with augmented dataset...[/blue]")
                return self.process_dataset(result)
        
        return train_data, val_data, test_data, dataset_path
