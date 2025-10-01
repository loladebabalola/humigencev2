# datasets/download_datasets.py

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import print
import requests
import zipfile
import tempfile

console = Console()

# Base directory for datasets
HUMIGENCE_DATA_DIR = Path.home() / "humigence_data"

def ensure_data_dir():
    """Ensure the humigence_data directory exists"""
    HUMIGENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return HUMIGENCE_DATA_DIR

def save_jsonl(data: List[Dict[str, Any]], filepath: Path) -> int:
    """Save data as JSONL file and return number of samples"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(data)

def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB"""
    return filepath.stat().st_size / (1024 * 1024)

def create_demo_dataset() -> List[Dict[str, Any]]:
    """Create a small demo dataset with 5 samples"""
    return [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
        },
        {
            "instruction": "Explain the difference between supervised and unsupervised learning.",
            "input": "",
            "output": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds hidden patterns in data without labeled examples."
        },
        {
            "instruction": "What is a neural network?",
            "input": "",
            "output": "A neural network is a computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information through weighted connections."
        },
        {
            "instruction": "What is overfitting in machine learning?",
            "input": "",
            "output": "Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor performance on new, unseen data."
        },
        {
            "instruction": "What is the purpose of a validation set?",
            "input": "",
            "output": "A validation set is used to tune hyperparameters and evaluate model performance during training, helping to prevent overfitting and select the best model configuration."
        }
    ]

# Removed all Hugging Face dataset download functions
# The system now only works with locally available datasets

def create_finance_quiz() -> Optional[Path]:
    """Create a custom finance quiz dataset"""
    console.print("[cyan]📥 Creating Finance Quiz dataset...[/cyan]")
    
    finance_questions = [
        {
            "instruction": "What is compound interest?",
            "input": "",
            "output": "Compound interest is interest calculated on the initial principal and the accumulated interest of previous periods. It's often called 'interest on interest' and can significantly increase investment returns over time."
        },
        {
            "instruction": "What is the difference between a stock and a bond?",
            "input": "",
            "output": "A stock represents ownership in a company and gives you a claim on its assets and earnings. A bond is a loan you make to a company or government, which pays you back with interest over a fixed period."
        },
        {
            "instruction": "What is diversification in investing?",
            "input": "",
            "output": "Diversification is the practice of spreading investments across different assets, sectors, or geographic regions to reduce risk. The idea is that if one investment performs poorly, others may perform well, balancing overall portfolio risk."
        },
        {
            "instruction": "What is a 401(k) retirement plan?",
            "input": "",
            "output": "A 401(k) is a tax-advantaged retirement savings plan offered by employers. Employees can contribute pre-tax money, which grows tax-deferred until withdrawal in retirement, often with employer matching contributions."
        },
        {
            "instruction": "What is the rule of 72?",
            "input": "",
            "output": "The rule of 72 is a simple formula to estimate how long it takes for an investment to double in value. You divide 72 by the annual interest rate to get the approximate number of years needed for doubling."
        },
        {
            "instruction": "What is inflation and how does it affect savings?",
            "input": "",
            "output": "Inflation is the rate at which prices for goods and services increase over time. It reduces the purchasing power of money, meaning your savings will buy less in the future. This is why investing is important to outpace inflation."
        },
        {
            "instruction": "What is a mutual fund?",
            "input": "",
            "output": "A mutual fund is an investment vehicle that pools money from many investors to buy a diversified portfolio of stocks, bonds, or other securities. It's managed by professional fund managers and offers diversification with lower minimum investments."
        },
        {
            "instruction": "What is dollar-cost averaging?",
            "input": "",
            "output": "Dollar-cost averaging is an investment strategy where you invest a fixed amount of money at regular intervals, regardless of market conditions. This helps reduce the impact of market volatility and can lower the average cost per share over time."
        },
        {
            "instruction": "What is an emergency fund?",
            "input": "",
            "output": "An emergency fund is money set aside to cover unexpected expenses like medical bills, job loss, or major repairs. Financial experts typically recommend having 3-6 months of living expenses in an easily accessible, low-risk account."
        },
        {
            "instruction": "What is the difference between gross and net income?",
            "input": "",
            "output": "Gross income is your total earnings before any deductions like taxes, insurance, or retirement contributions. Net income (take-home pay) is what you actually receive after all deductions have been subtracted from your gross income."
        }
    ]
    
    filepath = HUMIGENCE_DATA_DIR / "finance_quiz" / "finance_quiz.jsonl"
    samples = save_jsonl(finance_questions, filepath)
    size_mb = get_file_size_mb(filepath)
    
    console.print(f"[green]✅ Finance Quiz: {samples} samples, {size_mb:.1f} MB[/green]")
    return filepath

def create_ai_cohost_dataset() -> Optional[Path]:
    """Create AI Co-Host conversation dataset"""
    console.print("[cyan]📥 Creating AI Co-Host dataset...[/cyan]")
    
    conversations = [
        {
            "instruction": "Help me write a professional email to my manager about working remotely.",
            "input": "",
            "output": "I'd be happy to help you draft a professional email about remote work. Here's a template:\n\nSubject: Request for Remote Work Arrangement\n\nDear [Manager's Name],\n\nI hope this email finds you well. I would like to discuss the possibility of transitioning to a remote work arrangement. Given the current circumstances and my role's compatibility with remote work, I believe this arrangement would benefit both my productivity and work-life balance.\n\nI'm committed to maintaining the same level of communication and deliverables as in the office. I'm available to discuss this further and address any concerns you might have.\n\nThank you for your consideration.\n\nBest regards,\n[Your Name]"
        },
        {
            "instruction": "I need help brainstorming ideas for a team building activity.",
            "input": "",
            "output": "Here are some engaging team building activity ideas:\n\n1. **Virtual Escape Room** - Solve puzzles together online\n2. **Cooking Class** - Learn to make a dish together via video call\n3. **Trivia Night** - Test knowledge on various topics\n4. **Scavenger Hunt** - Find items around the house/office\n5. **Pictionary/Charades** - Classic games adapted for video calls\n6. **Show and Tell** - Share personal items or talents\n7. **Two Truths and a Lie** - Get to know each other better\n8. **Collaborative Playlist** - Create a team playlist together\n\nWhich type of activity interests your team most?"
        },
        {
            "instruction": "Can you help me prepare for a job interview?",
            "input": "",
            "output": "Absolutely! Here's a comprehensive interview preparation guide:\n\n**Before the Interview:**\n- Research the company, role, and interviewer\n- Prepare STAR method examples for behavioral questions\n- Practice common questions out loud\n- Prepare thoughtful questions to ask them\n- Plan your outfit and test technology (if virtual)\n\n**Common Questions to Practice:**\n- Tell me about yourself\n- Why do you want this job?\n- What are your strengths/weaknesses?\n- Where do you see yourself in 5 years?\n- Why should we hire you?\n\n**During the Interview:**\n- Use the STAR method (Situation, Task, Action, Result)\n- Ask thoughtful questions about the role/company\n- Show enthusiasm and cultural fit\n- Take notes if appropriate\n\nWould you like me to help you practice specific questions?"
        }
    ]
    
    filepath = HUMIGENCE_DATA_DIR / "ai_cohost_conversations" / "ai_cohost_conversations.jsonl"
    samples = save_jsonl(conversations, filepath)
    size_mb = get_file_size_mb(filepath)
    
    console.print(f"[green]✅ AI Co-Host: {samples} samples, {size_mb:.1f} MB[/green]")
    return filepath

def create_demo_dataset_file() -> Optional[Path]:
    """Create the demo dataset file"""
    console.print("[cyan]📥 Creating Demo dataset...[/cyan]")
    
    data = create_demo_dataset()
    filepath = HUMIGENCE_DATA_DIR / "demo" / "demo.jsonl"
    samples = save_jsonl(data, filepath)
    size_mb = get_file_size_mb(filepath)
    
    console.print(f"[green]✅ Demo: {samples} samples, {size_mb:.1f} MB[/green]")
    return filepath

# Dataset definitions - only local datasets
DATASET_DEFINITIONS = {
    "demo": {
        "name": "Demo (5 samples)",
        "description": "Small demo dataset for testing",
        "size": "micro",
        "download_func": create_demo_dataset_file
    },
    "finance_quiz": {
        "name": "Finance Quiz (10 samples)",
        "description": "Custom finance knowledge questions",
        "size": "small",
        "download_func": create_finance_quiz
    },
    "ai_cohost_conversations": {
        "name": "AI Co-Host (3 samples)",
        "description": "Professional conversation examples",
        "size": "small",
        "download_func": create_ai_cohost_dataset
    }
}

def get_available_datasets() -> List[Dict[str, Any]]:
    """Get list of available datasets by scanning ~/humigence_data/ for .jsonl files"""
    available = []
    
    if not HUMIGENCE_DATA_DIR.exists():
        return available
    
    # Scan for all .jsonl files in the humigence_data directory
    for jsonl_file in HUMIGENCE_DATA_DIR.rglob("*.jsonl"):
        try:
            # Get file info
            size_mb = get_file_size_mb(jsonl_file)
            
            # Count samples
            with open(jsonl_file, 'r') as f:
                sample_count = sum(1 for _ in f)
            
            # Extract dataset name from file path
            dataset_name = jsonl_file.stem
            relative_path = jsonl_file.relative_to(HUMIGENCE_DATA_DIR)
            
            # Create a user-friendly name
            display_name = dataset_name.replace('_', ' ').title()
            if sample_count <= 10:
                size_category = "micro"
            elif sample_count <= 1000:
                size_category = "small"
            elif sample_count <= 10000:
                size_category = "medium"
            else:
                size_category = "large"
            
            available.append({
                "id": dataset_name,
                "name": f"{display_name} ({sample_count} samples)",
                "description": f"Local dataset with {sample_count} samples",
                "size": size_category,
                "samples": sample_count,
                "file_size_mb": size_mb,
                "path": str(jsonl_file)
            })
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not process {jsonl_file}: {e}[/yellow]")
            continue
    
    # Sort by sample count (smallest first)
    available.sort(key=lambda x: x['samples'])
    
    return available

def download_datasets(dataset_ids: List[str]) -> Dict[str, bool]:
    """Download specified datasets and return success status"""
    ensure_data_dir()
    results = {}
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        total_tasks = len(dataset_ids)
        main_task = progress.add_task("Downloading datasets...", total=total_tasks)
        
        for i, dataset_id in enumerate(dataset_ids):
            if dataset_id in DATASET_DEFINITIONS:
                definition = DATASET_DEFINITIONS[dataset_id]
                progress.update(main_task, description=f"Downloading {definition['name']}...")
                
                try:
                    result = definition["download_func"]()
                    results[dataset_id] = result is not None
                except Exception as e:
                    console.print(f"[red]❌ Failed to download {dataset_id}: {e}[/red]")
                    results[dataset_id] = False
                
                progress.advance(main_task)
            else:
                console.print(f"[red]❌ Unknown dataset: {dataset_id}[/red]")
                results[dataset_id] = False
    
    return results

def ensure_demo_dataset() -> bool:
    """Ensure at least the demo dataset is available"""
    demo_path = HUMIGENCE_DATA_DIR / "demo" / "demo.jsonl"
    if not demo_path.exists():
        console.print("[yellow]📦 Creating demo dataset...[/yellow]")
        result = create_demo_dataset_file()
        return result is not None
    return True

if __name__ == "__main__":
    # Test the download functionality
    ensure_data_dir()
    available = get_available_datasets()
    print(f"Available datasets: {len(available)}")
    for dataset in available:
        print(f"  - {dataset['name']}: {dataset['samples']} samples, {dataset['file_size_mb']:.1f} MB")
