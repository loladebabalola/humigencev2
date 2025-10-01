# pipelines/production_pipeline.py

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel

# Import production-grade components
from .dataset_splitter import DatasetSplitter
from .trainer import ProductionTrainer
from .distributed_trainer import DistributedProductionTrainer
from .evaluator import ProductionEvaluator
from .reporter import ProductionReporter
from .memory_monitor import MemoryMonitor
from distributed_utils import RankZeroOnly

console = Console()

class ProductionPipeline:
    """Production-grade supervised fine-tuning pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path("runs/humigence")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distributed training setup
        self.distributed = config.get("distributed", False)
        self.ddp = config.get("ddp", False)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)
        self.local_rank = config.get("local_rank", 0)
        self.device = config.get("device", "cuda:0")
        self.is_main = config.get("is_main", True)
        
        # Initialize components
        self.dataset_splitter = DatasetSplitter(config)
        
        # Use distributed trainer if DDP is enabled
        if self.ddp:
            self.trainer = DistributedProductionTrainer(config)
        else:
            self.trainer = ProductionTrainer(config)
            
        self.evaluator = None
        self.reporter = ProductionReporter(config)
        self.memory_monitor = MemoryMonitor(config)
        
        # Pipeline state
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.dataset_path = None
        
    def run(self) -> Dict:
        """Run the complete production pipeline"""
        console.print("\n[bold cyan]🚀 Starting Production Fine-Tuning Pipeline[/bold cyan]")
        console.print("[bold cyan]=" * 60)
        
        # Print training mode and device information with rank-aware logging
        with RankZeroOnly(self.is_main) as rank_zero:
            if self.ddp:
                rank_zero.print(f"[green]✅ Multi-GPU Training: {self.world_size} GPUs[/green]")
                rank_zero.print(f"[blue]   Rank: {self.rank}/{self.world_size-1}[/blue]")
                rank_zero.print(f"[blue]   Local Rank: {self.local_rank}[/blue]")
                rank_zero.print(f"[blue]   Device: {self.device}[/blue]")
            else:
                rank_zero.print("[blue]✅ Single GPU Training: cuda:0[/blue]")
                rank_zero.print(f"[blue]   Device: {self.device}[/blue]")
        
        try:
            # Step 1: Dataset Processing and Splitting
            self._process_dataset()
            
            # Step 2: Model Loading and Training
            self._train_model()
            
            # Step 3: Comprehensive Evaluation
            self._evaluate_model()
            
            # Step 4: Generate Reports
            self._generate_reports()
            
            console.print("\n[bold green]✅ Production pipeline completed successfully![/bold green]")
            
            return {
                "status": "success",
                "dataset_info": self._get_dataset_info(),
                "training_results": self.trainer.training_history,
                "evaluation_results": self.evaluation_results,
                "report_path": str(self.reporter.summary_dir)
            }
            
        except KeyboardInterrupt:
            console.print("\n[bold red]❌ Pipeline aborted by user[/bold red]")
            return {"status": "aborted"}
        except Exception as e:
            console.print(f"\n[bold red]❌ Pipeline failed: {e}[/bold red]")
            return {"status": "failed", "error": str(e)}
    
    def _process_dataset(self):
        """Process dataset with splitting and integrity checks"""
        console.print("\n[bold blue]📊 Step 1: Dataset Processing and Splitting[/bold blue]")
        
        dataset_path = self.config["dataset_path"]
        console.print(f"[blue]📁 Processing dataset: {dataset_path}[/blue]")
        
        # Process dataset with splitting and integrity checks
        self.train_data, self.val_data, self.test_data, self.dataset_path = self.dataset_splitter.process_dataset(dataset_path)
        
        console.print(f"[green]✅ Dataset processed successfully[/green]")
        console.print(f"[blue]📊 Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}[/blue]")
    
    def _train_model(self):
        """Load model and run training with memory monitoring"""
        console.print("\n[bold blue]🤖 Step 2: Model Loading and Training[/bold blue]")
        
        # Load model and tokenizer
        self.trainer.load_model_and_tokenizer()
        
        # Prepare datasets
        self.trainer.prepare_datasets(self.train_data, self.val_data, self.test_data)
        
        # Setup training
        self.trainer.setup_training()
        
        # Add memory monitoring if enabled
        if self.config.get("memory_monitoring", False):
            console.print("[blue]📊 Memory monitoring enabled[/blue]")
            self.trainer.set_memory_monitor(self.memory_monitor)
        
        try:
            # Run training
            self.trainer.train()
            
        except RuntimeError as e:
            # Handle NCCL/CUDA errors
            if self.memory_monitor.handle_nccl_error(e):
                console.print("[yellow]🔄 Attempting to recover from error...[/yellow]")
                # Try to continue training
                self.trainer.train()
                console.print("[green]✅ Training recovered and completed[/green]")
            else:
                console.print(f"[red]❌ Training failed: {e}[/red]")
                raise e
        
        # Save model for evaluation
        self.trainer.save_model()
        console.print("[blue]💾 Model saved for evaluation[/blue]")
        
        console.print("[green]✅ Training completed successfully[/green]")
    
    def _evaluate_model(self):
        """Run comprehensive evaluation with proper device cleanup"""
        console.print("\n[bold blue]🧪 Step 3: Comprehensive Evaluation[/bold blue]")
        
        # Get trained model and tokenizer
        model, tokenizer = self.trainer.get_model_and_tokenizer()
        
        # Initialize evaluator
        self.evaluator = ProductionEvaluator(model, tokenizer, self.config)
        
        # Run comprehensive evaluation with device cleanup
        self.evaluation_results = self.evaluator.comprehensive_evaluation(
            self.train_data, self.val_data, self.test_data
        )
        
        # Offer AI analysis
        self.ai_analysis = self.evaluator.offer_ai_analysis(self.evaluation_results)
        
        console.print("[green]✅ Evaluation completed successfully[/green]")
    
    def _generate_reports(self):
        """Generate comprehensive reports"""
        console.print("\n[bold blue]📊 Step 4: Generate Reports[/bold blue]")
        
        # Generate comprehensive report
        self.report = self.reporter.generate_comprehensive_report(
            dataset_info=self._get_dataset_info(),
            training_history=self.trainer.training_history,
            evaluation_results=self.evaluation_results,
            ai_analysis=self.ai_analysis
        )
        
        # Save artifacts info
        self.reporter.save_artifacts(
            model_path=str(self.output_dir / "final_model"),
            tokenizer_path=str(self.output_dir / "tokenizer")
        )
        
        console.print("[green]✅ Reports generated successfully[/green]")
    
    def _get_dataset_info(self) -> Dict:
        """Get dataset information for reporting"""
        total_samples = len(self.train_data) + len(self.val_data) + len(self.test_data)
        
        return {
            "total_samples": total_samples,
            "train_samples": len(self.train_data),
            "val_samples": len(self.val_data),
            "test_samples": len(self.test_data),
            "train_percentage": (len(self.train_data) / total_samples * 100) if total_samples > 0 else 0,
            "val_percentage": (len(self.val_data) / total_samples * 100) if total_samples > 0 else 0,
            "test_percentage": (len(self.test_data) / total_samples * 100) if total_samples > 0 else 0,
            "avg_tokens": self.dataset_splitter._estimate_avg_tokens(self.train_data + self.val_data + self.test_data),
            "dataset_path": self.dataset_path
        }

def main(config_path: Path) -> Dict:
    """Main entry point for production pipeline"""
    console.print(Panel.fit(
        "[bold cyan]Humigence Production Pipeline[/bold cyan]\n"
        "[dim]Production-grade supervised fine-tuning with comprehensive evaluation[/dim]",
        border_style="cyan"
    ))
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create and run pipeline
    pipeline = ProductionPipeline(config)
    results = pipeline.run()
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python production_pipeline.py <config_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    results = main(config_path)
    print(f"Pipeline completed with status: {results['status']}")
