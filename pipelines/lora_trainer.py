# lora_trainer.py

import json
import typer
from pathlib import Path
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TextStreamer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import os
import zipfile
import time

app = typer.Typer()
console = Console()

def estimate_micro_batch_size():
    import torch

    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_vram > 40:
        return 8
    elif total_vram > 20:
        return 4
    elif total_vram > 10:
        return 2
    else:
        return 1

def load_tokenizer_and_model(cfg):
    base_model = cfg["base_model"]
    recipe = cfg["training_recipe"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # For now, load model without quantization due to RTX 5090 compatibility issues
    # TODO: Re-enable quantization once PyTorch/bitsandbytes supports RTX 5090
    console.print("[yellow]⚠️ Loading model without quantization (RTX 5090 compatibility)[/yellow]")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="bfloat16" if "BF16" in recipe else "float16"
    )

    return tokenizer, model

def apply_lora(model, cfg):
    return get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ))

def load_small_dataset(dataset_path, tokenizer):
    import json

    # Load first 100 samples max
    data = []
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            data.append(json.loads(line))

    # Handle OpenAssistant format - group by message_tree_id
    conversations = {}
    for sample in data:
        tree_id = sample.get("message_tree_id")
        if tree_id not in conversations:
            conversations[tree_id] = []
        conversations[tree_id].append(sample)

    # Create instruction-response pairs
    texts = []
    for tree_id, messages in conversations.items():
        if len(messages) >= 2:
            # Find prompter and assistant messages
            prompter_msg = None
            assistant_msg = None
            for msg in messages:
                if msg.get("role") == "prompter" and prompter_msg is None:
                    prompter_msg = msg
                elif msg.get("role") == "assistant" and assistant_msg is None:
                    assistant_msg = msg
            
            if prompter_msg and assistant_msg:
                text = f"### Instruction:\n{prompter_msg['text']}\n\n### Response:\n{assistant_msg['text']}"
                texts.append(text)

    # Tokenize
    if texts:
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    else:
        # Fallback if no conversations found
        tokenized = tokenizer(
            ["### Instruction:\nHello\n\n### Response:\nHi there!"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    return tokenized

def get_training_args(cfg, output_dir="runs/humigence"):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # fixed for now
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=1,
        bf16="BF16" in cfg["training_recipe"],
        fp16="FP16" in cfg["training_recipe"],
        evaluation_strategy="no",
        save_strategy="steps",
        report_to="none"
    )

def run_evaluation(model, tokenizer, eval_path="runs/humigence/eval_prompts.jsonl"):
    if not Path(eval_path).exists():
        console.print("[yellow]⚠️ No evaluation prompts found — skipping eval[/yellow]")
        return []

    with open(eval_path, "r") as f:
        prompts = [json.loads(line)["instruction"] for line in f]

    results = []
    streamer = TextStreamer(tokenizer)

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        output = model.generate(
            input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        console.print(f"\n[bold cyan]📌 Prompt {i+1}[/bold cyan]: {prompt}")
        console.print(f"[bold green]🧠 Model Output[/bold green]: {decoded}")

        results.append({"prompt": prompt, "output": decoded})

    return results

def passed_acceptance_criteria(eval_results, trainer):
    loss = trainer.state.log_history[-1].get("loss", 999)
    return loss < 0.8 and len(eval_results) >= 1

def zip_artifacts(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for path in Path(folder_path).rglob('*'):
            if path.is_file():
                zipf.write(path, path.relative_to(folder_path))

@app.command()
def main(config: Path = typer.Argument(help="Path to config.snapshot.json")):
    console.print("[bold cyan]🚀 Humigence Trainer Starting...[/bold cyan]")

    # Load config file
    if not config.exists():
        console.print(f"[bold red]❌ Config file not found:[/bold red] {config}")
        raise typer.Exit(code=1)

    with open(config, "r") as f:
        cfg = json.load(f)

    # Echo key config values for debugging
    console.print("[bold green]✅ Configuration Loaded:[/bold green]")
    for k, v in cfg.items():
        console.print(f"[bold]{k}[/bold]: {v}")

    # Auto micro-batch size estimation
    micro_batch = estimate_micro_batch_size()
    console.print(f"[bold blue]📦 Estimated Micro-batch Size:[/bold blue] {micro_batch}")

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(cfg)
    console.print(f"[bold green]✅ Model + Tokenizer Loaded:[/bold green] [yellow]{cfg['base_model']}[/yellow]")

    # Apply LoRA if needed
    if "LoRA" in cfg["training_recipe"] or "QLoRA" in cfg["training_recipe"]:
        model = apply_lora(model, cfg)
        console.print("[bold green]✅ LoRA adapters applied[/bold green]")

    # Load dataset
    console.print("[bold blue]📚 Loading dataset...[/bold blue]")
    dataset = load_small_dataset(cfg["dataset_path"], tokenizer)
    console.print(f"[bold green]✅ Dataset loaded: {len(dataset['input_ids'])} samples[/bold green]")

    # Build dataset format
    train_dataset = [{"input_ids": x, "attention_mask": y} for x, y in zip(dataset["input_ids"], dataset["attention_mask"])]

    # Setup training
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = get_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )

    # Start training
    console.print("[bold green]🚀 Starting training...[/bold green]")
    trainer.train()

    # Save adapters
    model.save_pretrained("runs/humigence/adapters")
    tokenizer.save_pretrained("runs/humigence/tokenizer")
    console.print("[bold green]✅ Training complete — adapters saved.[/bold green]")

    # Run evaluation
    console.print("\n[bold magenta]🧪 Running Evaluation Prompts...[/bold magenta]")
    eval_results = run_evaluation(model, tokenizer)

    # Check acceptance criteria
    if passed_acceptance_criteria(eval_results, trainer):
        console.print("[bold green]✅ Run accepted: metrics meet thresholds.[/bold green]")
        with open("runs/humigence/ACCEPTED.txt", "w") as f:
            f.write("Training run accepted based on loss and eval criteria.\n")
    else:
        console.print("[bold red]❌ Run failed acceptance criteria.[/bold red]")
        with open("runs/humigence/REJECTED.txt", "w") as f:
            f.write("Training run rejected. Loss too high or missing eval outputs.\n")

    # Save evaluation results (if any)
    if eval_results:
        with open("runs/humigence/eval_results.jsonl", "w") as f:
            for item in eval_results:
                f.write(json.dumps(item) + "\n")

    # Export full run
    zip_artifacts("runs/humigence", "runs/humigence/artifacts.zip")
    console.print("[bold green]📦 All artifacts exported to [cyan]artifacts.zip[/cyan][/bold green]")

    # Create structured run summary
    summary = {
        "run_id": cfg.get("timestamp", time.time()),
        "status": "accepted" if Path("runs/humigence/ACCEPTED.txt").exists() else "rejected",
        "model": cfg["base_model"],
        "dataset": cfg["dataset_path"],
        "recipe": cfg["training_recipe"],
        "epochs": cfg["num_train_epochs"],
        "learning_rate": cfg["learning_rate"],
        "final_loss": trainer.state.log_history[-1].get("loss", None),
        "eval_prompt_count": len(eval_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("runs/humigence/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print("[bold green]✅ Run summary saved to run_summary.json[/bold green]")

if __name__ == "__main__":
    app()