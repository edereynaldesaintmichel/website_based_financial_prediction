"""
Training script for FinancialModernBert on financial statements.

This script:
1. Loads financial statements from JSON files
2. Converts them to markdown tables with <number></number> tags
3. Masks ~10% of numbers during training
4. Trains the model to predict the masked numbers (sign + magnitude)
"""
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime

# from financial_bert import (
#     FinancialBertTokenizer,
#     FinancialModernBert,
#     build_model,
# )
# from financial_dataset import (
#     FinancialStatementDataset,
#     financial_collate_fn,
#     create_train_val_split,
#     save_split_info,
# )


def train_epoch(model, dataloader, optimizer, device, epoch, num_epochs, scaler=None, use_amp=True, accumulation_steps=1):
    """Train for one epoch with optional mixed precision and gradient accumulation.

    Args:
        accumulation_steps: Number of batches to accumulate gradients over before updating weights.
                          Effective batch size = batch_size * accumulation_steps
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Use automatic mixed precision with bfloat16
        if use_amp and device.type == "cuda":
            with autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs['loss']

            if loss is not None:
                # Scale loss by accumulation steps to get correct average
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()

                # Only update weights every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps  # Unscale for logging
                num_batches += 1

                progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
        else:
            # Fallback to FP32 if not using AMP
            outputs = model(**batch)
            loss = outputs['loss']

            if loss is not None:
                # Scale loss by accumulation steps to get correct average
                loss = loss / accumulation_steps
                loss.backward()

                # Only update weights every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps  # Unscale for logging
                num_batches += 1

                progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, dataloader, device, use_amp=True):
    """Validate the model with optional mixed precision."""
    model.eval()
    total_loss = 0
    num_batches = 0

    # Metrics for number prediction
    total_sign_correct = 0
    total_sign_count = 0
    total_magnitude_error = 0
    total_magnitude_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Use mixed precision for validation too
            if use_amp and device.type == "cuda":
                with autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs['loss']
            else:
                outputs = model(**batch)
                loss = outputs['loss']

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

            # Calculate accuracy metrics
            sign_logits = outputs['sign_logits']
            magnitude_logits = outputs['magnitude_logits']

            labels_sign = batch['labels_sign']
            labels_magnitude = batch['labels_magnitude']

            # Sign accuracy
            valid_sign_mask = (labels_sign != -100)
            if valid_sign_mask.any():
                pred_sign = torch.argmax(sign_logits, dim=-1)
                correct = (pred_sign == labels_sign) & valid_sign_mask
                total_sign_correct += correct.sum().item()
                total_sign_count += valid_sign_mask.sum().item()

            # Magnitude error (in log10 scale)
            valid_mag_mask = (labels_magnitude != -100)
            if valid_mag_mask.any():
                # Decode magnitude from logits
                mag_probs = torch.softmax(magnitude_logits, dim=-1)
                num_bins = model.config.num_magnitude_bins
                bin_indices = torch.arange(num_bins, device=device).float()

                # Expected bin index
                expected_bin = (mag_probs * bin_indices.view(1, 1, -1)).sum(dim=-1)

                # Convert to log value
                min_v = model.config.magnitude_min
                max_v = model.config.magnitude_max
                pred_log = (expected_bin / (num_bins - 1)) * (max_v - min_v) + min_v

                # Compute error
                errors = torch.abs(pred_log - labels_magnitude)
                valid_errors = errors[valid_mag_mask]
                total_magnitude_error += valid_errors.sum().item()
                total_magnitude_count += valid_mag_mask.sum().item()

    avg_loss = total_loss / max(num_batches, 1)
    sign_accuracy = total_sign_correct / max(total_sign_count, 1)
    avg_magnitude_error = total_magnitude_error / max(total_magnitude_count, 1)

    return {
        "loss": avg_loss,
        "sign_accuracy": sign_accuracy,
        "magnitude_error": avg_magnitude_error
    }


def train_financial(
    data_dir: str,
    output_dir: str = "checkpoints",
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    mask_ratio: float = 0.10,
    max_length: int = 2048,
    file_indices: list = None,
    num_magnitude_bins: int = 128,
    magnitude_min: float = -3.0,
    magnitude_max: float = 12.0,
    save_every: int = 1,
    accumulation_steps: int = 1,
    resume_from_checkpoint: str = None,
):
    """
    Main training function for Google Colab.

    Args:
        data_dir: Directory containing full_reports_*.json files
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU (use smaller values with gradient accumulation)
        lr: Learning rate
        mask_ratio: Fraction of numbers to mask (~10% recommended)
        max_length: Maximum sequence length
        file_indices: Which data files to use (default: all 0-25)
        num_magnitude_bins: Number of bins for magnitude discretization
        magnitude_min: Minimum log10 magnitude
        magnitude_max: Maximum log10 magnitude
        save_every: Save checkpoint every N epochs
        accumulation_steps: Number of batches to accumulate gradients over.
                          Effective batch size = batch_size * accumulation_steps
                          Use this to simulate larger batch sizes without OOM errors.
        resume_from_checkpoint: Path to a previous checkpoint to resume training from.
                               If provided, loads the model weights (but not optimizer state).
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = FinancialBertTokenizer(
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
    )

    # Load datasets
    print("Loading datasets...")
    if file_indices is None:
        file_indices = list(range(26))  # All files

    train_dataset, val_dataset, split_info = create_train_val_split(
        data_dir=data_dir,
        tokenizer=tokenizer,
        train_ratio=0.9,
        file_indices=file_indices,
        mask_ratio=mask_ratio,
        max_length=max_length,
    )
    
    # Save split info so validation uses the exact same companies
    split_info_path = os.path.join(output_dir, "train_val_split.json")
    save_split_info(split_info, split_info_path)

    # Create dataloaders
    collate = partial(financial_collate_fn, tokenizer=tokenizer, mask_ratio=mask_ratio, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,  # Set to 0 for debugging, increase for speed
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize model
    print("Building model...")
    model = build_model(
        num_magnitude_bins=num_magnitude_bins,
    )
    model.config.magnitude_min = magnitude_min
    model.config.magnitude_max = magnitude_max
    
    # Optionally load from a previous checkpoint
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded model weights from checkpoint")
        # Note: We intentionally do NOT load optimizer state to allow fresh training
        # with the new (corrected) data split
    
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Initialize GradScaler for mixed precision training
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    print(f"\nStarting training...")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Accumulation steps: {accumulation_steps}")
    print(f"  - Effective batch size: {batch_size * accumulation_steps}")
    print(f"  - Mask ratio: {mask_ratio:.1%}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print(f"  - Mixed Precision: {'BF16' if use_amp else 'FP32'}")
    print()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, num_epochs,
            scaler=scaler, use_amp=use_amp, accumulation_steps=accumulation_steps
        )
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, device, use_amp=use_amp)
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Sign Accuracy: {val_metrics['sign_accuracy']:.2%}")
        print(f"  Magnitude Error: {val_metrics['magnitude_error']:.4f} (in log10 scale)")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"financial_bert_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': {
                    'num_magnitude_bins': num_magnitude_bins,
                    'magnitude_min': magnitude_min,
                    'magnitude_max': magnitude_max,
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = os.path.join(output_dir, "financial_bert_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': {
                    'num_magnitude_bins': num_magnitude_bins,
                    'magnitude_min': magnitude_min,
                    'magnitude_max': magnitude_max,
                }
            }, best_path)
            print(f"  New best model saved to {best_path}")

        # Step scheduler
        scheduler.step()
        print()

    print("Training complete!")
    return model


# Example usage for Google Colab:
#
train_financial(
    data_dir="/content/drive/MyDrive/website predictor/Financials Tables",
    output_dir="/content/drive/MyDrive/website predictor/Financials Tables",
    batch_size=4,
    lr=1e-4,
    accumulation_steps=32,  # Effective batch size = 4 * 32 = 128
    num_epochs=5,
    # Resume from previous checkpoint to continue training with corrected data split:
    resume_from_checkpoint="/content/drive/MyDrive/website predictor/Financials Tables/financial_bert_best.pt",
)