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
from datasets import load_dataset
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

def download_oasst1() -> Optional[Path]:
    """Download OpenAssistant OASST1 dataset"""
    console.print("[cyan]📥 Downloading OASST1 (OpenAssistant)...[/cyan]")
    
    try:
        # Load the dataset
        dataset = load_dataset("OpenAssistant/oasst1")
        
        # Convert to our schema
        data = []
        for split in ['train', 'validation']:
            if split in dataset:
                for item in dataset[split]:
                    if item.get('text') and item.get('role') == 'assistant':
                        # Find the corresponding human message
                        parent_id = item.get('parent_id')
                        if parent_id:
                            # Look for the human message
                            for human_item in dataset[split]:
                                if (human_item.get('message_id') == parent_id and 
                                    human_item.get('role') == 'prompter'):
                                    data.append({
                                        "instruction": human_item['text'],
                                        "input": "",
                                        "output": item['text']
                                    })
                                    break
        
        # Limit to reasonable size for demo
        data = data[:10000]  # Limit to 10k samples
        
        filepath = HUMIGENCE_DATA_DIR / "oasst1" / "oasst1.jsonl"
        samples = save_jsonl(data, filepath)
        size_mb = get_file_size_mb(filepath)
        
        console.print(f"[green]✅ OASST1: {samples} samples, {size_mb:.1f} MB[/green]")
        return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download OASST1: {e}[/red]")
        return None

def download_squad_v2() -> Optional[Path]:
    """Download SQuAD v2 dataset"""
    console.print("[cyan]📥 Downloading SQuAD v2...[/cyan]")
    
    try:
        dataset = load_dataset("squad_v2")
        data = []
        
        for split in ['train', 'validation']:
            if split in dataset:
                for item in dataset[split]:
                    if item['answers']['text']:  # Only include questions with answers
                        data.append({
                            "instruction": f"Answer this question: {item['question']}",
                            "input": f"Context: {item['context']}",
                            "output": item['answers']['text'][0] if item['answers']['text'] else "No answer available"
                        })
        
        filepath = HUMIGENCE_DATA_DIR / "squad_v2" / "squad_v2.jsonl"
        samples = save_jsonl(data, filepath)
        size_mb = get_file_size_mb(filepath)
        
        console.print(f"[green]✅ SQuAD v2: {samples} samples, {size_mb:.1f} MB[/green]")
        return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download SQuAD v2: {e}[/red]")
        return None

def download_dailydialog() -> Optional[Path]:
    """Download DailyDialog dataset"""
    console.print("[cyan]📥 Downloading DailyDialog...[/cyan]")
    
    try:
        dataset = load_dataset("daily_dialog")
        data = []
        
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                for conversation in dataset[split]['dialog']:
                    # Convert multi-turn conversations to instruction-response pairs
                    for i in range(0, len(conversation) - 1, 2):
                        if i + 1 < len(conversation):
                            data.append({
                                "instruction": conversation[i],
                                "input": "",
                                "output": conversation[i + 1]
                            })
        
        filepath = HUMIGENCE_DATA_DIR / "dailydialog" / "dailydialog.jsonl"
        samples = save_jsonl(data, filepath)
        size_mb = get_file_size_mb(filepath)
        
        console.print(f"[green]✅ DailyDialog: {samples} samples, {size_mb:.1f} MB[/green]")
        return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download DailyDialog: {e}[/red]")
        return None

def download_spider() -> Optional[Path]:
    """Download Spider dataset"""
    console.print("[cyan]📥 Downloading Spider...[/cyan]")
    
    try:
        dataset = load_dataset("spider")
        data = []
        
        for split in ['train', 'validation']:
            if split in dataset:
                for item in dataset[split]:
                    data.append({
                        "instruction": f"Generate SQL query: {item['question']}",
                        "input": f"Database schema: {item.get('db_id', '')}",
                        "output": item['query']
                    })
        
        filepath = HUMIGENCE_DATA_DIR / "spider" / "spider.jsonl"
        samples = save_jsonl(data, filepath)
        size_mb = get_file_size_mb(filepath)
        
        console.print(f"[green]✅ Spider: {samples} samples, {size_mb:.1f} MB[/green]")
        return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download Spider: {e}[/red]")
        return None

def download_personachat() -> Optional[Path]:
    """Download PersonaChat dataset"""
    console.print("[cyan]📥 Downloading PersonaChat...[/cyan]")
    
    try:
        dataset = load_dataset("personachat")
        data = []
        
        for split in ['train', 'validation']:
            if split in dataset:
                for conversation in dataset[split]:
                    # Convert persona-based conversations
                    persona = conversation.get('personas', [])
                    persona_text = " ".join(persona[0]) if persona else ""
                    
                    for i in range(0, len(conversation['utterances']) - 1, 2):
                        if i + 1 < len(conversation['utterances']):
                            data.append({
                                "instruction": f"{persona_text}\n\n{conversation['utterances'][i]}",
                                "input": "",
                                "output": conversation['utterances'][i + 1]
                            })
        
        filepath = HUMIGENCE_DATA_DIR / "personachat" / "personachat.jsonl"
        samples = save_jsonl(data, filepath)
        size_mb = get_file_size_mb(filepath)
        
        console.print(f"[green]✅ PersonaChat: {samples} samples, {size_mb:.1f} MB[/green]")
        return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download PersonaChat: {e}[/red]")
        return None

def download_ag_news() -> Optional[Path]:
    """Download AG News dataset"""
    console.print("[cyan]📥 Downloading AG News...[/cyan]")
    
    try:
        dataset = load_dataset("ag_news")
        data = []
        
        for split in ['train', 'test']:
            if split in dataset:
                for item in dataset[split]:
                    # Convert news classification to instruction-response
                    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}
                    label = label_map.get(item['label'], "Unknown")
                    
                    data.append({
                        "instruction": f"Classify this news article: {item['text']}",
                        "input": "",
                        "output": f"This article belongs to the {label} category."
                    })
        
        filepath = HUMIGENCE_DATA_DIR / "ag_news" / "ag_news.jsonl"
        samples = save_jsonl(data, filepath)
        size_mb = get_file_size_mb(filepath)
        
        console.print(f"[green]✅ AG News: {samples} samples, {size_mb:.1f} MB[/green]")
        return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download AG News: {e}[/red]")
        return None

def download_cornell_movie_dialogs() -> Optional[Path]:
    """Download Cornell Movie Dialogs dataset"""
    console.print("[cyan]📥 Downloading Cornell Movie Dialogs...[/cyan]")
    
    try:
        # Download from the original source
        url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "cornell.zip"
            
            # Download the zip file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract and process
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process the movie_lines.txt file
            lines_file = Path(temp_dir) / "cornell movie-dialogs-corpus" / "movie_lines.txt"
            if lines_file.exists():
                data = []
                with open(lines_file, 'r', encoding='iso-8859-1') as f:
                    for line in f:
                        parts = line.strip().split(' +++$+++ ')
                        if len(parts) >= 5:
                            text = parts[4]
                            if len(text) > 10:  # Filter out very short lines
                                data.append({
                                    "instruction": f"Continue this movie dialogue: {text[:50]}...",
                                    "input": "",
                                    "output": text
                                })
                
                # Limit size
                data = data[:5000]
                
                filepath = HUMIGENCE_DATA_DIR / "cornell_movie_dialogs" / "cornell_movie_dialogs.jsonl"
                samples = save_jsonl(data, filepath)
                size_mb = get_file_size_mb(filepath)
                
                console.print(f"[green]✅ Cornell Movie Dialogs: {samples} samples, {size_mb:.1f} MB[/green]")
                return filepath
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download Cornell Movie Dialogs: {e}[/red]")
        return None

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

# Dataset definitions
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
    },
    "cornell_movie_dialogs": {
        "name": "Cornell Movie Dialogs (5K samples)",
        "description": "Movie conversation dataset",
        "size": "small",
        "download_func": download_cornell_movie_dialogs
    },
    "spider": {
        "name": "Spider (7K samples)",
        "description": "Text-to-SQL dataset",
        "size": "medium",
        "download_func": download_spider
    },
    "personachat": {
        "name": "PersonaChat (8K samples)",
        "description": "Persona-based conversations",
        "size": "medium",
        "download_func": download_personachat
    },
    "ag_news": {
        "name": "AG News (120K samples)",
        "description": "News classification dataset",
        "size": "medium",
        "download_func": download_ag_news
    },
    "squad_v2": {
        "name": "SQuAD v2 (150K samples)",
        "description": "Question answering dataset",
        "size": "large",
        "download_func": download_squad_v2
    },
    "dailydialog": {
        "name": "DailyDialog (13K samples)",
        "description": "Daily conversation dataset",
        "size": "large",
        "download_func": download_dailydialog
    },
    "oasst1": {
        "name": "OASST1 (10K samples)",
        "description": "OpenAssistant conversations",
        "size": "large",
        "download_func": download_oasst1
    }
}

def get_available_datasets() -> List[Dict[str, Any]]:
    """Get list of available datasets with their status"""
    available = []
    
    for dataset_id, definition in DATASET_DEFINITIONS.items():
        dataset_dir = HUMIGENCE_DATA_DIR / dataset_id
        dataset_file = dataset_dir / f"{dataset_id}.jsonl"
        
        if dataset_file.exists():
            # Get file info
            size_mb = get_file_size_mb(dataset_file)
            with open(dataset_file, 'r') as f:
                sample_count = sum(1 for _ in f)
            
            available.append({
                "id": dataset_id,
                "name": definition["name"],
                "description": definition["description"],
                "size": definition["size"],
                "samples": sample_count,
                "file_size_mb": size_mb,
                "path": str(dataset_file)
            })
    
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
