"""
Validation script for FinancialModernBert.

This script shows how well the model predicts masked numbers in financial statements.
For a randomly picked statement, it outputs a markdown table showing:
- The original financial statement
- For all masked numbers: predicted vs actual values
"""
import torch
import random
import re
import math
from functools import partial

from financial_bert import (
    FinancialBertTokenizer,
    FinancialModernBert,
    build_model,
)
from financial_dataset import (
    FinancialStatementDataset,
    financial_collate_fn,
    create_train_val_split,
    load_split_info,
    load_validation_dataset,
    FIELD_NAMES,
)


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint.
    
    Supports both:
    - Best model checkpoints (model_state_dict, val_metrics, config)
    - Epoch checkpoints (model_state_dict, optimizer_state_dict, train_loss, val_metrics, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    num_magnitude_bins = config.get('num_magnitude_bins', 128)
    magnitude_min = config.get('magnitude_min', -3.0)
    magnitude_max = config.get('magnitude_max', 12.0)
    
    model = build_model(num_magnitude_bins=num_magnitude_bins)
    model.config.magnitude_min = magnitude_min
    model.config.magnitude_max = magnitude_max
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = FinancialBertTokenizer(
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
    )
    
    return model, tokenizer, config


def reconstruct_table_with_predictions(
    sample_markdown: str,
    predictions: dict,
    ground_truth: dict,
) -> str:
    """
    Reconstruct the markdown table with predictions shown inline.
    
    Args:
        sample_markdown: Original markdown table
        predictions: Dict mapping number_index (0, 1, 2, ...) -> prediction info
                     Each prediction has 'value', 'sign', 'log_magnitude'
        ground_truth: Dict mapping number_index (0, 1, 2, ...) -> ground truth info
                      Each has 'value', 'sign', 'magnitude'
    
    Returns:
        Markdown table with predictions inline where numbers were masked
    """
    import re
    
    # Parse the markdown to find all number tags
    number_pattern = re.compile(r'<number>(.*?)</number>')
    
    # Build result with replacements
    result = []
    last_end = 0
    
    for number_idx, match in enumerate(number_pattern.finditer(sample_markdown)):
        # Add text before this number
        result.append(sample_markdown[last_end:match.start()])
        
        # Check if this number was masked (by its sequential index)
        if number_idx in predictions:
            pred = predictions[number_idx]
            gt = ground_truth[number_idx]
            
            # Format values
            actual_str = f"{gt['value']:.2f}" if abs(gt['value']) >= 1 else f"{gt['value']:.6f}"
            pred_str = f"{pred['value']:.2f}" if abs(pred['value']) >= 1 else f"{pred['value']:.6f}"
            
            # Calculate relative error
            if abs(gt['value']) > 1e-10:
                rel_error = abs(pred['value'] - gt['value']) / abs(gt['value'])
            else:
                rel_error = abs(pred['value'])
            
            sign_ok = "✓" if pred['sign'] == gt['sign'] else "✗"
            
            # Create inline display
            cell = f"**[PRED: {pred_str} ; ACT: {actual_str} ; Err: {rel_error:.1%} ; {sign_ok}]**"
            result.append(cell)
        else:
            # Keep original number tag
            result.append(match.group(0))
        
        last_end = match.end()
    
    # Add remaining text
    result.append(sample_markdown[last_end:])
    
    return ''.join(result)


def validate_sample(
    model,
    tokenizer,
    sample_markdown: str,
    mask_ratio: float = 0.10,
    device: torch.device = None,
    verbose: bool = True,
):
    """
    Validate model predictions on a single financial statement.
    
    Args:
        model: Trained FinancialModernBert model
        tokenizer: FinancialBertTokenizer
        sample_markdown: The markdown table to validate
        mask_ratio: Fraction of numbers to mask
        device: Torch device
        verbose: Whether to print detailed output
    
    Returns:
        Dict with validation results
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Parse the original numbers from markdown
    number_pattern = re.compile(r'<number>(.*?)</number>')
    original_numbers = []
    for match in number_pattern.finditer(sample_markdown):
        try:
            val = float(match.group(1))
            original_numbers.append({
                'start': match.start(),
                'end': match.end(),
                'value': val,
                'text': match.group(0)
            })
        except ValueError:
            original_numbers.append({
                'start': match.start(),
                'end': match.end(),
                'value': 0.0,
                'text': match.group(0)
            })
    
    if len(original_numbers) == 0:
        print("No numbers found in sample!")
        return None
    
    # Tokenize
    encodings = tokenizer(
        [sample_markdown],
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    input_ids = encodings['input_ids'].clone()
    is_number_mask = encodings['is_number_mask'].clone()
    number_values = encodings['number_values'].clone()
    attention_mask = encodings['attention_mask']
    
    # Find number positions (token indices where numbers are)
    number_positions = (is_number_mask[0] == 1).nonzero(as_tuple=True)[0].tolist()
    
    if len(number_positions) == 0:
        print("No number tokens found after tokenization!")
        return None
    
    # Build mapping: token_position -> number_index (sequential order)
    # This assumes number tokens appear in the same order as numbers in markdown
    token_pos_to_number_idx = {pos: idx for idx, pos in enumerate(number_positions)}
    
    # Select random number indices to mask
    num_to_mask = max(1, int(len(number_positions) * mask_ratio))
    # Sample from sequential indices, not token positions
    masked_number_indices = set(random.sample(range(len(number_positions)), num_to_mask))
    
    # Build ground truth keyed by number index, and mask the selected numbers
    ground_truth = {}
    mask_token_id = tokenizer.mask_token_id
    
    for number_idx, token_pos in enumerate(number_positions):
        if number_idx in masked_number_indices:
            sign_val = int(number_values[0, token_pos, 0].item())
            mag_val = number_values[0, token_pos, 1].item()
            orig_val = tokenizer.log_to_value(sign_val, mag_val)
            
            ground_truth[number_idx] = {
                'sign': sign_val,
                'magnitude': mag_val,
                'value': orig_val
            }
            
            # Mask the number
            number_values[0, token_pos, 0] = 0.0
            number_values[0, token_pos, 1] = 0.0
            input_ids[0, token_pos] = mask_token_id
    
    # Run model
    inputs = {
        'input_ids': input_ids.to(device),
        'is_number_mask': is_number_mask.to(device),
        'number_values': number_values.to(device),
        'attention_mask': attention_mask.to(device),
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Decode predictions for masked numbers (keyed by number index)
    predictions = {}
    for number_idx in masked_number_indices:
        token_pos = number_positions[number_idx]
        
        decoded = tokenizer.decode_number(
            sign_logits=outputs['sign_logits'][0, token_pos],
            magnitude_logits=outputs['magnitude_logits'][0, token_pos],
            model_config=model.config,
            k=5,
        )
        
        predictions[number_idx] = {
            'sign': decoded['sign'],
            'magnitude': decoded['log_magnitude'],
            'value': decoded['value']
        }
    
    # Calculate errors
    results = []
    for number_idx in masked_number_indices:
        gt = ground_truth[number_idx]
        pred = predictions[number_idx]
        
        # Relative error (handle zeros)
        if abs(gt['value']) > 1e-10:
            rel_error = abs(pred['value'] - gt['value']) / abs(gt['value'])
        else:
            rel_error = abs(pred['value'])
        
        # Log magnitude error
        log_error = abs(pred['magnitude'] - gt['magnitude'])
        
        # Sign correctness
        sign_correct = (pred['sign'] == gt['sign'])
        
        results.append({
            'number_index': number_idx,
            'actual_value': gt['value'],
            'predicted_value': pred['value'],
            'actual_sign': gt['sign'],
            'predicted_sign': pred['sign'],
            'actual_log_mag': gt['magnitude'],
            'predicted_log_mag': pred['magnitude'],
            'relative_error': rel_error,
            'log_error': log_error,
            'sign_correct': sign_correct,
        })
    
    if verbose:
        print("\n" + "="*80)
        print("FINANCIAL STATEMENT VALIDATION")
        print("="*80)
        
        # Reconstruct table with predictions (now using number indices)
        reconstructed = reconstruct_table_with_predictions(
            sample_markdown, predictions, ground_truth
        )
        
        print(f"\nFinancial Statement with Predictions:")
        print(f"(Masked {len(masked_number_indices)} numbers out of {len(number_positions)} total)")
        print("\n" + reconstructed)
        
        # Summary statistics
        avg_rel_error = sum(r['relative_error'] for r in results) / len(results)
        avg_log_error = sum(r['log_error'] for r in results) / len(results)
        sign_accuracy = sum(r['sign_correct'] for r in results) / len(results)
        
        print("\n" + "="*80)
        print(f"SUMMARY:")
        print(f"  Average Relative Error: {avg_rel_error:.2%}")
        print(f"  Average Log Magnitude Error: {avg_log_error:.4f}")
        print(f"  Sign Accuracy: {sign_accuracy:.1%}")
        print("="*80)
    
    return results


def validate_random_sample(
    checkpoint_path: str,
    data_dir: str = "financial_statements_data",
    split_info_path: str = None,
    mask_ratio: float = 0.10,
    seed: int = None,
):
    """
    Load a model and validate on a randomly selected financial statement.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing financial data
        split_info_path: Path to train_val_split.json saved during training.
                         If provided, uses exactly the same validation companies.
                         If None, falls back to creating a new split (NOT RECOMMENDED).
        mask_ratio: Fraction of numbers to mask
        seed: Random seed (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer, config = load_model(checkpoint_path, device)
    
    print("Loading validation data...")
    
    if split_info_path:
        # Use the saved split from training (RECOMMENDED)
        split_info = load_split_info(split_info_path)
        val_dataset = load_validation_dataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            split_info=split_info,
            mask_ratio=mask_ratio,
        )
    else:
        # Fallback: create new split (NOT RECOMMENDED - may leak data!)
        print("WARNING: No split_info_path provided. Creating new split which may cause data leakage!")
        _, val_dataset, _ = create_train_val_split(
            data_dir=data_dir,
            tokenizer=tokenizer,
            train_ratio=0.9,
            mask_ratio=mask_ratio,
        )
    
    random.seed(None)
    # Pick random sample
    idx = random.randint(0, len(val_dataset) - 1)
    sample = val_dataset[idx]
    
    print(f"\nSelected sample index: {idx}")
    print(f"Company: {sample.get('symbol', 'Unknown')}")
    
    # Validate
    results = validate_sample(
        model=model,
        tokenizer=tokenizer,
        sample_markdown=sample['markdown'],
        mask_ratio=mask_ratio,
        device=device,
        verbose=True,
    )
    
    return results


def validate_multiple_samples(
    checkpoint_path: str,
    data_dir: str = "financial_statements_data", 
    split_info_path: str = None,
    num_samples: int = 10,
    mask_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Validate model on multiple random samples and aggregate results.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing financial data
        split_info_path: Path to train_val_split.json saved during training.
                         If provided, uses exactly the same validation companies.
                         If None, falls back to creating a new split (NOT RECOMMENDED).
        num_samples: Number of samples to validate
        mask_ratio: Fraction of numbers to mask
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer, config = load_model(checkpoint_path, device)
    
    print("Loading validation data...")
    
    if split_info_path:
        # Use the saved split from training (RECOMMENDED)
        split_info = load_split_info(split_info_path)
        val_dataset = load_validation_dataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            split_info=split_info,
            mask_ratio=mask_ratio,
        )
    else:
        # Fallback: create new split (NOT RECOMMENDED - may leak data!)
        print("WARNING: No split_info_path provided. Creating new split which may cause data leakage!")
        _, val_dataset, _ = create_train_val_split(
            data_dir=data_dir,
            tokenizer=tokenizer,
            train_ratio=0.9,
            mask_ratio=mask_ratio,
        )
    
    # Sample indices
    all_indices = list(range(len(val_dataset)))
    sample_indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    
    all_results = []
    
    for i, idx in enumerate(sample_indices):
        print(f"\n--- Sample {i+1}/{len(sample_indices)} (idx={idx}) ---")
        sample = val_dataset[idx]
        
        results = validate_sample(
            model=model,
            tokenizer=tokenizer,
            sample_markdown=sample['markdown'],
            mask_ratio=mask_ratio,
            device=device,
            verbose=False,
        )
        
        if results:
            all_results.extend(results)
            
            # Quick summary for this sample
            avg_rel = sum(r['relative_error'] for r in results) / len(results)
            sign_acc = sum(r['sign_correct'] for r in results) / len(results)
            print(f"  {len(results)} masked numbers, Avg Rel Error: {avg_rel:.2%}, Sign Acc: {sign_acc:.1%}")
    
    # Overall summary
    if all_results:
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        print(f"Total predictions: {len(all_results)}")
        print(f"Average Relative Error: {sum(r['relative_error'] for r in all_results) / len(all_results):.2%}")
        print(f"Average Log Magnitude Error: {sum(r['log_error'] for r in all_results) / len(all_results):.4f}")
        print(f"Sign Accuracy: {sum(r['sign_correct'] for r in all_results) / len(all_results):.1%}")
        
        # Error distribution
        errors = [r['relative_error'] for r in all_results]
        errors.sort()
        print(f"\nRelative Error Percentiles:")
        print(f"  25th: {errors[len(errors)//4]:.2%}")
        print(f"  50th (median): {errors[len(errors)//2]:.2%}")
        print(f"  75th: {errors[3*len(errors)//4]:.2%}")
        print(f"  90th: {errors[9*len(errors)//10]:.2%}")
    
    return all_results


# Example usage - just run this file directly!
if __name__ == "__main__":
    # Single sample validation (detailed view)
    # Use the split_info_path from training to ensure no data leakage!
    validate_random_sample(
        checkpoint_path="financial_bert_best.pt",
        data_dir="financial_statements_data",
        split_info_path="checkpoints/train_val_split.json",  # Path to saved split from training
        mask_ratio=0.10,
        seed=None,  # Random seed, set to a number for reproducibility
    )
    
    # Uncomment below to validate multiple samples instead:
    # validate_multiple_samples(
    #     checkpoint_path="financial_bert_best.pt",
    #     data_dir="financial_statements_data",
    #     split_info_path="checkpoints/train_val_split.json",
    #     num_samples=10,
    #     mask_ratio=0.10,
    #     seed=42,
    # )

