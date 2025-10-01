# pipelines/reporter.py

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class ProductionReporter:
    """Production-grade reporter for comprehensive training summaries"""
    
    def __init__(self, config: Dict, output_dir: str = "runs/humigence"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.summary_dir = self.output_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    dataset_info: Dict,
                                    training_history: Dict,
                                    evaluation_results: Dict,
                                    ai_analysis: Optional[str] = None) -> Dict:
        """Generate comprehensive training report"""
        
        console.print("\n[bold cyan]📊 Generating Comprehensive Report...[/bold cyan]")
        
        # Compile all information
        report = {
            "metadata": self._generate_metadata(),
            "dataset_info": dataset_info,
            "training_config": self._extract_training_config(),
            "training_results": training_history,
            "evaluation_results": evaluation_results,
            "ai_analysis": ai_analysis,
            "summary": self._generate_summary(dataset_info, training_history, evaluation_results)
        }
        
        # Save reports
        self._save_json_report(report)
        self._save_markdown_report(report)
        self._display_summary_table(report)
        
        console.print(f"[green]✅ Comprehensive report generated in: {self.summary_dir}[/green]")
        
        return report
    
    def _generate_metadata(self) -> Dict:
        """Generate report metadata"""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report_id": f"humigence_report_{self.timestamp}",
            "version": "1.0",
            "generator": "Humigence Production Pipeline"
        }
    
    def _extract_training_config(self) -> Dict:
        """Extract training configuration from config"""
        return {
            "base_model": self.config.get("base_model", "unknown"),
            "training_recipe": self.config.get("training_recipe", "unknown"),
            "learning_rate": float(self.config.get("learning_rate", "2e-4")),
            "num_epochs": int(self.config.get("num_train_epochs", "1")),
            "batch_size": int(self.config.get("per_device_train_batch_size", "2")),
            "gradient_accumulation": int(self.config.get("gradient_accumulation_steps", "4")),
            "max_samples": self.config.get("max_samples"),
            "split_ratios": self.config.get("split_ratios", [0.8, 0.1, 0.1])
        }
    
    def _generate_summary(self, dataset_info: Dict, training_history: Dict, evaluation_results: Dict) -> Dict:
        """Generate executive summary"""
        val_metrics = evaluation_results.get("validation", {})
        test_metrics = evaluation_results.get("test", {})
        overfitting = evaluation_results.get("overfitting", {})
        qualitative = evaluation_results.get("qualitative", {})
        
        # Calculate key metrics
        val_loss = val_metrics.get("loss", 0.0)
        test_loss = test_metrics.get("loss", 0.0)
        is_overfitting = overfitting.get("is_overfitting", False)
        qualitative_score = qualitative.get("score", 0.0)
        
        # Determine overall status
        if val_loss < 1.0 and not is_overfitting and qualitative_score > 0.8:
            status = "EXCELLENT"
            status_color = "green"
        elif val_loss < 2.0 and not is_overfitting and qualitative_score > 0.6:
            status = "GOOD"
            status_color = "green"
        elif val_loss < 3.0 and qualitative_score > 0.4:
            status = "FAIR"
            status_color = "yellow"
        else:
            status = "POOR"
            status_color = "red"
        
        return {
            "overall_status": status,
            "status_color": status_color,
            "key_metrics": {
                "validation_loss": val_loss,
                "test_loss": test_loss,
                "perplexity": val_metrics.get("perplexity", 0.0),
                "accuracy": val_metrics.get("accuracy", 0.0),
                "bleu_score": val_metrics.get("bleu", 0.0),
                "qualitative_score": qualitative_score
            },
            "overfitting_detected": is_overfitting,
            "training_duration": training_history.get("training_duration", 0.0),
            "recommendations": self._generate_recommendations(val_loss, is_overfitting, qualitative_score)
        }
    
    def _generate_recommendations(self, val_loss: float, is_overfitting: bool, qualitative_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if val_loss > 2.0:
            recommendations.append("Consider increasing training data or adjusting learning rate")
        
        if is_overfitting:
            recommendations.append("Add regularization or more diverse training data")
        
        if qualitative_score < 0.6:
            recommendations.append("Improve training data quality and diversity")
        
        if val_loss < 1.0 and qualitative_score > 0.8:
            recommendations.append("Model performance is excellent - ready for production")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory - monitor in production")
        
        return recommendations
    
    def _save_json_report(self, report: Dict):
        """Save comprehensive JSON report"""
        json_path = self.summary_dir / f"training_report_{self.timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"[blue]📄 JSON report saved: {json_path}[/blue]")
    
    def _save_markdown_report(self, report: Dict):
        """Save human-readable Markdown report"""
        md_path = self.summary_dir / f"training_report_{self.timestamp}.md"
        
        markdown_content = self._generate_markdown_content(report)
        
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        
        console.print(f"[blue]📄 Markdown report saved: {md_path}[/blue]")
    
    def _generate_markdown_content(self, report: Dict) -> str:
        """Generate Markdown content for the report"""
        metadata = report["metadata"]
        dataset_info = report["dataset_info"]
        training_config = report["training_config"]
        training_results = report["training_results"]
        evaluation_results = report["evaluation_results"]
        summary = report["summary"]
        
        content = f"""# Humigence Training Report

**Generated:** {metadata['timestamp']}  
**Report ID:** {metadata['report_id']}  
**Status:** <span style="color: {summary['status_color']}">{summary['overall_status']}</span>

## Executive Summary

This report summarizes the supervised fine-tuning of {training_config['base_model']} using the {training_config['training_recipe']} approach.

### Key Metrics
- **Validation Loss:** {summary['key_metrics']['validation_loss']:.4f}
- **Test Loss:** {summary['key_metrics']['test_loss']:.4f}
- **Perplexity:** {summary['key_metrics']['perplexity']:.2f}
- **Accuracy:** {summary['key_metrics']['accuracy']:.3f}
- **BLEU Score:** {summary['key_metrics']['bleu_score']:.3f}
- **Qualitative Score:** {summary['key_metrics']['qualitative_score']:.3f}

### Overfitting Analysis
- **Overfitting Detected:** {'Yes' if summary['overfitting_detected'] else 'No'}
- **Training Duration:** {summary['training_duration']:.2f} seconds

## Dataset Information

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | {dataset_info.get('train_samples', 'N/A')} | {dataset_info.get('train_percentage', 'N/A')}% |
| Validation | {dataset_info.get('val_samples', 'N/A')} | {dataset_info.get('val_percentage', 'N/A')}% |
| Test | {dataset_info.get('test_samples', 'N/A')} | {dataset_info.get('test_percentage', 'N/A')}% |

**Total Samples:** {dataset_info.get('total_samples', 'N/A')}  
**Average Tokens per Sample:** {dataset_info.get('avg_tokens', 'N/A')}

## Training Configuration

- **Model:** {training_config['base_model']}
- **Recipe:** {training_config['training_recipe']}
- **Learning Rate:** {training_config['learning_rate']}
- **Epochs:** {training_config['num_epochs']}
- **Batch Size:** {training_config['batch_size']}
- **Gradient Accumulation:** {training_config['gradient_accumulation']}
- **Max Samples:** {training_config['max_samples'] or 'Full dataset'}

## Training Results

- **Final Training Loss:** {training_results.get('train_loss', 'N/A')}
- **Final Validation Loss:** {training_results.get('eval_loss', 'N/A')}
- **Training Duration:** {training_results.get('training_duration', 'N/A')} seconds
- **Epochs Completed:** {training_results.get('epochs_completed', 'N/A')}

## Evaluation Results

### Validation Set
- **Loss:** {evaluation_results.get('validation', {}).get('loss', 'N/A')}
- **Perplexity:** {evaluation_results.get('validation', {}).get('perplexity', 'N/A')}
- **Accuracy:** {evaluation_results.get('validation', {}).get('accuracy', 'N/A')}
- **BLEU:** {evaluation_results.get('validation', {}).get('bleu', 'N/A')}

### Test Set
- **Loss:** {evaluation_results.get('test', {}).get('loss', 'N/A')}
- **Perplexity:** {evaluation_results.get('test', {}).get('perplexity', 'N/A')}
- **Accuracy:** {evaluation_results.get('test', {}).get('accuracy', 'N/A')}

### ROUGE Scores
- **ROUGE-1:** {evaluation_results.get('validation', {}).get('rouge', {}).get('rouge1', 'N/A')}
- **ROUGE-2:** {evaluation_results.get('validation', {}).get('rouge', {}).get('rouge2', 'N/A')}
- **ROUGE-L:** {evaluation_results.get('validation', {}).get('rouge', {}).get('rougeL', 'N/A')}

## Recommendations

{chr(10).join(f"- {rec}" for rec in summary['recommendations'])}

## AI Analysis

{report.get('ai_analysis', 'No AI analysis available')}

---

*Report generated by Humigence Production Pipeline v1.0*
"""
        
        return content
    
    def _display_summary_table(self, report: Dict):
        """Display summary table in console"""
        summary = report["summary"]
        key_metrics = summary["key_metrics"]
        
        console.print("\n[bold cyan]📊 TRAINING SUMMARY[/bold cyan]")
        
        # Status table
        status_table = Table(title="Overall Status", show_header=True, header_style="bold cyan")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")
        
        status_table.add_row("Overall Status", f"[{summary['status_color']}]{summary['overall_status']}[/{summary['status_color']}]")
        status_table.add_row("Validation Loss", f"{key_metrics['validation_loss']:.4f}")
        status_table.add_row("Test Loss", f"{key_metrics['test_loss']:.4f}")
        status_table.add_row("Perplexity", f"{key_metrics['perplexity']:.2f}")
        status_table.add_row("Accuracy", f"{key_metrics['accuracy']:.3f}")
        status_table.add_row("BLEU Score", f"{key_metrics['bleu_score']:.3f}")
        status_table.add_row("Qualitative Score", f"{key_metrics['qualitative_score']:.3f}")
        status_table.add_row("Overfitting", "Yes" if summary['overfitting_detected'] else "No")
        status_table.add_row("Training Duration", f"{summary['training_duration']:.2f}s")
        
        console.print(status_table)
        
        # Recommendations
        if summary['recommendations']:
            console.print("\n[bold yellow]💡 Recommendations:[/bold yellow]")
            for i, rec in enumerate(summary['recommendations'], 1):
                console.print(f"[yellow]  {i}. {rec}[/yellow]")
        
        # File locations
        console.print(f"\n[bold blue]📁 Reports saved to: {self.summary_dir}[/bold blue]")
        console.print(f"[blue]  • JSON: training_report_{self.timestamp}.json[/blue]")
        console.print(f"[blue]  • Markdown: training_report_{self.timestamp}.md[/blue]")
    
    def save_artifacts(self, model_path: str, tokenizer_path: str):
        """Save model artifacts information"""
        artifacts_info = {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifacts_dir": str(self.output_dir)
        }
        
        artifacts_path = self.summary_dir / "artifacts.json"
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts_info, f, indent=2)
        
        console.print(f"[blue]📦 Artifacts info saved: {artifacts_path}[/blue]")
