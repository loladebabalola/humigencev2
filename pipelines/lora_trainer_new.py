# New main function implementation
def main_new(config: Path = typer.Argument(help="Path to config.snapshot.json")):
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

    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(cfg)
    console.print(f"[bold green]✅ Model + Tokenizer Loaded:[/bold green] [yellow]{cfg['base_model']}[/yellow]")

    # Apply LoRA if needed
    if "LoRA" in cfg.get("training_recipe", "") or "QLoRA" in cfg.get("training_recipe", ""):
        model = apply_lora(model, cfg)
        console.print("[bold green]✅ LoRA adapters applied[/bold green]")

    # B) Load dataset with deterministic splitting
    console.print("[bold blue]📚 Loading dataset with deterministic splitting...[/bold blue]")
    dataset_splits = load_dataset(cfg["dataset_path"], tokenizer, cfg)
    
    # Extract splits
    train_dataset = dataset_splits["train"]
    val_dataset = dataset_splits["validation"] 
    test_dataset = dataset_splits["test"]
    raw_data = dataset_splits["raw_data"]
    
    console.print(f"[bold green]✅ Dataset loaded and split:[/bold green]")
    console.print(f"  Train: {len(raw_data['train'])} samples")
    console.print(f"  Validation: {len(raw_data['validation'])} samples")
    console.print(f"  Test: {len(raw_data['test'])} samples")
    
    # C) Setup training with proper DataLoaders
    console.print("\n[bold blue]🚀 Setting up training...[/bold blue]")
    
    # Convert tokenized data to proper format
    train_data = [{"input_ids": x, "attention_mask": y, "labels": x} for x, y in zip(train_dataset["input_ids"], train_dataset["attention_mask"])]
    val_data = [{"input_ids": x, "attention_mask": y, "labels": x} for x, y in zip(val_dataset["input_ids"], val_dataset["attention_mask"])]
    
    # Setup training with proper DataLoaders
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = get_training_args(cfg, has_validation=len(val_data) > 0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data if len(val_data) > 0 else None,
        data_collator=collator
    )

    # Start training
    console.print("[bold green]🚀 Starting training...[/bold green]")
    trainer.train()

    # Get training completion info
    final_loss = trainer.state.log_history[-1].get("loss", 0.0) if trainer.state.log_history else 0.0
    val_loss = trainer.state.log_history[-1].get("eval_loss", final_loss) if trainer.state.log_history else final_loss
    total_steps = len(trainer.state.log_history)
    
    console.print(f"\n[bold green]Training completed in {total_steps} steps[/bold green]")
    console.print(f"[bold green]Final training loss: {final_loss:.4f}[/bold green]")
    console.print(f"[bold green]Final validation loss: {val_loss:.4f}[/bold green]")

    # D) Bulletproof evaluation (no device mismatch, ever)
    console.print("\n[bold blue]🧪 Running bulletproof evaluation...[/bold blue]")
    
    # Prepare model for single GPU evaluation
    eval_model, eval_device = _prepare_model_for_single_gpu_eval(cfg)
    
    # E) Device-pure evaluation loop
    console.print(f"[blue]🔄 Running device-pure evaluation on {eval_device}[/blue]")
    
    all_preds, all_labels, all_losses = [], [], []
    
    import torch
    with torch.no_grad():
        # Process validation data
        for i in range(min(len(val_data), 50)):  # Limit for testing
            sample = val_data[i]
            
            # Create batch from single sample
            batch = {}
            for key, value in sample.items():
                if hasattr(value, 'unsqueeze'):
                    batch[key] = value.unsqueeze(0)  # Add batch dimension
                else:
                    batch[key] = value
            
            # Move batch to evaluation device
            batch = {k: (v.to(eval_device) if hasattr(v, "to") else v) for k, v in batch.items()}
            
            outputs = eval_model(**batch)
            loss = outputs.loss.detach()
            
            # Get predictions and move to CPU immediately
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Move to CPU for metrics/storage
            all_losses.append(loss.float().cpu())
            all_preds.append(predictions.detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())
    
    # E) Compute metrics on CPU only
    eval_loss = torch.stack(all_losses).mean().item() if all_losses else float("nan")
    
    # Handle variable length sequences for predictions/labels
    if all_preds and all_labels:
        max_pred_len = max(pred.shape[1] for pred in all_preds)
        max_label_len = max(label.shape[1] for label in all_labels)
        max_len = max(max_pred_len, max_label_len)
        
        # Pad predictions and labels to the same length
        padded_preds = []
        padded_labels = []
        
        for pred, label in zip(all_preds, all_labels):
            # Pad predictions
            if pred.shape[1] < max_len:
                pad_size = max_len - pred.shape[1]
                pred_padded = torch.cat([pred, torch.full((pred.shape[0], pad_size), tokenizer.pad_token_id)], dim=1)
            else:
                pred_padded = pred
            padded_preds.append(pred_padded)
            
            # Pad labels
            if label.shape[1] < max_len:
                pad_size = max_len - label.shape[1]
                label_padded = torch.cat([label, torch.full((label.shape[0], pad_size), tokenizer.pad_token_id)], dim=1)
            else:
                label_padded = label
            padded_labels.append(label_padded)
        
        # Concatenate the padded tensors
        all_preds = torch.cat(padded_preds, dim=0)
        all_labels = torch.cat(padded_labels, dim=0)
    else:
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
    
    # Compute metrics on CPU only
    accuracy = (all_preds == all_labels).float().mean().item() if all_labels.numel() > 0 else 0.0
    bleu = compute_bleu(all_preds, all_labels, tokenizer)
    rouge = compute_rouge(all_preds, all_labels, tokenizer)
    
    metrics = {
        "accuracy": accuracy,
        "bleu": bleu,
        "rouge": rouge
    }
    
    # E) Print and save evaluation summary
    _print_eval_summary(final_loss, eval_loss, metrics, len(val_data), len(raw_data['test']), cfg)
    _save_eval_artifacts(final_loss, eval_loss, metrics, len(val_data), len(raw_data['test']), cfg, trainer.state.log_history)

    # Save model
    console.print("\n[bold blue]💾 Saving model...[/bold blue]")
    model.save_pretrained("runs/humigence/adapters")
    tokenizer.save_pretrained("runs/humigence/tokenizer")
    console.print("[bold green]✅ Model saved to runs/humigence[/bold green]")

    console.print(f"\n[bold green]✅ Training run completed successfully![/bold green]")
    console.print(f"[bold green]✅ All artifacts saved to: runs/humigence/[/bold green]")
