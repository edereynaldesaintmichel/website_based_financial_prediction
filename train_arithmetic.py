"""
Training and evaluation script for the FinancialModernBert model on arithmetic tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from functools import partial

from financial_bert import (
    FinancialBertTokenizer,
    FinancialModernBert,
    build_model,
)


class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, num_samples=2000):
        self.tokenizer = tokenizer
        self.data = self._generate_data(num_samples)

    def _generate_data(self, num_samples):
        samples = []
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
        }

        templates = [
            "The result of <number>{a}</number> {op_word} <number>{b}</number> is [MASK].",
            "Calculate <number>{a}</number> {op_symbol} <number>{b}</number> = [MASK].",
            "What is <number>{a}</number> {op_word} <number>{b}</number>? It is [MASK].",
            "<number>{a}</number> {op_symbol} <number>{b}</number> equals [MASK].",
        ]

        op_map = {
            '+': ("plus", "+"),
            '-': ("minus", "-"),
            '*': ("multiplied by", "*"),
            '/': ("divided by", "/")
        }

        for _ in range(num_samples):
            a = round(random.normalvariate(0, 1000), 2)
            b = round(random.normalvariate(0, 1000), 2)
            if b == 0: b = 1

            op_key = random.choice(list(ops.keys()))
            result_val = ops[op_key](a, b)
            
            temp = random.choice(templates)
            op_word, op_sym = op_map[op_key]

            text = temp.format(a=a, b=b, op_word=op_word, op_symbol=op_sym)
            samples.append({"text": text, "result_val": result_val})
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def arithmetic_collate_fn(batch, tokenizer):
    texts = [item['text'] for item in batch]
    results = [item['result_val'] for item in batch]

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encodings['input_ids']
    is_number_mask = encodings['is_number_mask']
    number_values = encodings['number_values']
    attention_mask = encodings['attention_mask']

    batch_size, seq_len = input_ids.shape
    labels_text = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels_magnitude = torch.full((batch_size, seq_len), -100.0, dtype=torch.float)
    labels_sign = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    mask_token_id = tokenizer.mask_token_id

    for i, res_val in enumerate(results):
        mask_indices = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_indices) > 0:
            mask_idx = mask_indices[0].item()
            is_number_mask[i, mask_idx] = 1.0

            sign_idx, log_val = tokenizer.parse_number_to_log(str(res_val))

            labels_magnitude[i, mask_idx] = log_val
            labels_sign[i, mask_idx] = sign_idx

    return {
        "input_ids": input_ids,
        "is_number_mask": is_number_mask,
        "number_values": number_values,
        "attention_mask": attention_mask,
        "labels_text": labels_text,
        "labels_magnitude": labels_magnitude,
        "labels_sign": labels_sign
    }


def train_arithmetic(model, tokenizer, num_epochs=3, batch_size=32, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataset = ArithmeticDataset(tokenizer, num_samples=100000)
    collate = partial(arithmetic_collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Start training on {device}...")

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "arithmetic_model_log.pth")
    return model


def predict_arithmetic(model, tokenizer, text_prompt, k=5):
    """
    Predict the numeric result for a given arithmetic prompt.
    
    Uses the tokenizer's decode_number method with peak_window strategy.
    
    Args:
        model: Trained FinancialModernBert model
        tokenizer: FinancialBertTokenizer
        text_prompt: Text with [MASK] where the number should be predicted
        k: Window size for peak_window strategy (default=5)
    
    Returns:
        The predicted numeric value
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(text_prompt, return_tensors="pt")
    mask_token_id = tokenizer.mask_token_id
    input_ids = inputs['input_ids']
    
    # Find mask location
    mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)
    if len(mask_indices[0]) == 0:
        print("Error: No [MASK] token found.")
        return None

    batch_idx, seq_idx = mask_indices[0][0].item(), mask_indices[1][0].item()
    inputs['is_number_mask'][batch_idx, seq_idx] = 1.0
    inputs = {key: v.to(device) for key, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Use tokenizer's decode_number method
    decoded = tokenizer.decode_number(
        sign_logits=outputs['sign_logits'][batch_idx, seq_idx],
        magnitude_logits=outputs['magnitude_logits'][batch_idx, seq_idx],
        model_config=model.config,
        k=k,
    )

    final_value = decoded['value']
    sign_val = -1.0 if decoded['sign'] == 1 else 1.0
    pred_log_val = decoded['log_magnitude']

    print(f"\nPrompt: {text_prompt}")
    print(f"  > Strategy: peak_window (k={k})")
    print(f"  > Sign: {sign_val}")
    print(f"  > Log Val: {pred_log_val:.4f}")
    print(f"  > Result: {final_value:.6f}")

    return final_value


if __name__ == "__main__":
    print("Initializing Log-Magnitude Model with Soft Encoding...")
    my_tokenizer = FinancialBertTokenizer(
        "answerdotai/ModernBERT-base",
        magnitude_min=-3,
        magnitude_max=12.0
    )
    
    trained_model = build_model(
        "answerdotai/ModernBERT-base",
        num_magnitude_bins=128
    )
    
    trained_model = train_arithmetic(trained_model, my_tokenizer, num_epochs=1, batch_size=128, lr=1e-4)

    print("\n=== Testing Capabilities ===")

    predict_arithmetic(trained_model, my_tokenizer, "The result of <number>500</number> plus <number>200</number> is [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "The result of <number>1000</number> divided by <number>8</number> is [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "Calculate <number>122</number> - <number>345</number> = [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "Calculate <number>10</number> * <number>5.5</number> = [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "What is <number>15.5</number> minus <number>2.25</number>? It is [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "What is <number>-12</number> multiplied by <number>-5</number>? It is [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "<number>15420</number> + <number>4580</number> equals [MASK].")
    predict_arithmetic(trained_model, my_tokenizer, "<number>200</number> / <number>2</number> equals [MASK].")
