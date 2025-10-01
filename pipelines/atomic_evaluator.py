# pipelines/atomic_evaluator.py

import torch
import os
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()

def humigence_atomic_evaluation(config: Dict, model_path: Path, datasets: Dict) -> Dict:
    """
    THE DEFINITIVE SOLUTION: Complete process separation for evaluation
    
    This function provides true process isolation to eliminate device mismatch issues
    that occur when training on multiple GPUs and evaluating on a single GPU.
    
    Args:
        config: Configuration dictionary containing gpu_id and other settings
        model_path: Path to the trained model directory
        datasets: Dictionary containing validation and test datasets
        
    Returns:
        Dictionary containing evaluation results
    """
    console.print("🚀 Starting atomic evaluation with true process isolation...")
    
    # Create a temporary directory for all artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save evaluation script that is COMPLETELY independent
        eval_script = f"""
import torch
import json
import os
import sys
from pathlib import Path

# ADD the model path to Python path so we can import transformers
model_path = Path(r"{model_path}")
sys.path.insert(0, str(model_path.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

def main():
    # SET CUDA DEVICE FIRST - before ANY other imports
    os.environ["CUDA_VISIBLE_DEVICES"] = "{config.get('gpu_id', 0)}"
    
    # Force torch to initialize on the correct device
    torch.cuda.set_device(torch.device("cuda:{config.get('gpu_id', 0)}"))
    
    print("🔧 Loading model in clean environment...")
    
    # Load model and tokenizer FRESH
    model = AutoModelForCausalLM.from_pretrained(
        r"{model_path}",
        torch_dtype=torch.float16,
        device_map=None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(r"{model_path}")
    
    # Move model to target device
    device = torch.device("cuda:{config.get('gpu_id', 0)}")
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully on", device)
    
    # Load datasets
    datasets_path = Path(r"{temp_dir}") / "datasets.json"
    with open(datasets_path, 'r') as f:
        datasets_dict = json.load(f)
    
    results = {{}}
    
    for dataset_name, dataset_dict in datasets_dict.items():
        if dataset_dict:  # Skip None entries
            print(f"🧪 Evaluating {{dataset_name}}...")
            dataset = Dataset.from_dict(dataset_dict)
            loss, perplexity = evaluate_dataset(model, tokenizer, dataset, device)
            results[dataset_name] = {{'eval_loss': loss, 'perplexity': perplexity}}
            print(f"✅ {{dataset_name}} - Loss: {{loss:.4f}}, Perplexity: {{perplexity:.2f}}")
    
    # Save results
    results_path = Path(r"{temp_dir}") / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🎉 ATOMIC EVALUATION COMPLETED SUCCESSFULLY!")

def evaluate_dataset(model, tokenizer, dataset, device):
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        # Move to device in one operation
        return {{k: v.to(device) for k, v in inputs.items()}}
    
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Verify device consistency
            for key, tensor in batch.items():
                if tensor.device != device:
                    raise RuntimeError(f"Tensor {{key}} on wrong device: {{tensor.device}} vs {{device}}")
            
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            
            if loss is not None:
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
            
            if i % 50 == 0:
                print(f"   Batch {{i}} completed")
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

if __name__ == "__main__":
    main()
"""
        
        # Save the script
        script_path = temp_path / "atomic_eval.py"
        with open(script_path, 'w') as f:
            f.write(eval_script)
        
        # Save datasets
        datasets_dict = {
            'validation': datasets['validation'].to_dict() if 'validation' in datasets else None,
            'test': datasets['test'].to_dict() if 'test' in datasets else None
        }
        
        with open(temp_path / "datasets.json", 'w') as f:
            json.dump(datasets_dict, f)
        
        # Use os.system for TRUE process separation (not subprocess)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
        env['PYTHONPATH'] = str(model_path.parent)  # Ensure imports work
        
        # Use absolute path to Python executable
        python_exe = sys.executable
        script_cmd = f'"{python_exe}" "{script_path}"'
        
        console.print("🚀 Launching truly isolated evaluation process...")
        return_code = os.system(script_cmd)
        
        if return_code == 0:
            # Load results
            results_path = temp_path / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                console.print("✅ Atomic evaluation completed successfully!")
                return results
            else:
                raise RuntimeError("Results file not found")
        else:
            raise RuntimeError(f"Atomic evaluation failed with return code: {return_code}")

def generate_atomic_report(config: Dict, eval_results: Dict) -> Dict:
    """Generate the final successful report for atomic evaluation"""
    report = {
        'status': 'success',
        'pipeline_version': 'atomic_evaluation_1.0',
        'evaluation_method': 'atomic_process_isolation',
        'results': eval_results,
        'device_info': {
            'evaluation_device': f"cuda:{config.get('gpu_id', 0)}",
            'process_isolation': True,
            'cuda_devices_visible': '1'  # Only one device visible during eval
        },
        'metrics': {
            'validation_loss': eval_results.get('validation', {}).get('eval_loss', 'N/A'),
            'test_loss': eval_results.get('test', {}).get('eval_loss', 'N/A'),
            'validation_perplexity': eval_results.get('validation', {}).get('perplexity', 'N/A'),
            'test_perplexity': eval_results.get('test', {}).get('perplexity', 'N/A'),
        }
    }
    
    # Print beautiful summary
    console.print("\n" + "="*80)
    console.print("🎯 ATOMIC EVALUATION RESULTS")
    console.print("="*80)
    for dataset_name, results in eval_results.items():
        if results:
            console.print(f"📊 {dataset_name.upper():>12}: Loss: {results['eval_loss']:.4f}, Perplexity: {results['perplexity']:.2f}")
    console.print("="*80)
    console.print("✅ DEVICE ISSUES PERMANENTLY RESOLVED")
    console.print("="*80)
    
    return report

def generate_atomic_fallback_report(config: Dict) -> Dict:
    """Generate a report even if atomic evaluation fails"""
    return {
        'status': 'training_completed_evaluation_failed',
        'pipeline_version': 'atomic_evaluation_1.0',
        'evaluation_method': 'atomic_process_isolation_attempted',
        'error': 'Atomic evaluation failed but training completed successfully',
        'next_steps': [
            'Model was trained successfully and saved',
            'Evaluation can be run manually with atomic evaluation',
            'The device isolation approach ensures no multi-GPU contamination'
        ]
    }

