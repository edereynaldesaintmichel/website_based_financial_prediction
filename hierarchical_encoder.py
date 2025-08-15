import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import transformers
print(transformers.__version__)

import os
import glob
import random
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import pickle
import gc # ### CHANGE: Import the garbage collector module



# --- 1. Configuration ---
class Config:
    # Data and Model Paths
    DATA_DIR = "/Users/eloireynal/Documents/My projects/crawl_data/sanitized_txt/"  # Update for Colab
    BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
    CHECKPOINT_DIR = "checkpoints"
    CACHE_DIR = "embedding_cache"  # New: directory for cached embeddings
    
    # Hierarchical Model Architecture
    EMBEDDING_DIM = 768  # From modern-bert-base
    NUM_LAYERS = 3
    NUM_ATTENTION_HEADS = 4
    FFN_DIM_MULTIPLIER = 4
    
    # Training Parameters
    EPOCHS = 7
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16  # Batch size for the hierarchical model
    TRAIN_SPLIT_RATIO = 0.9
    
    # Data Processing
    MAX_CONTEXT_LEN = 8192  # ModernBERT's context window
    MIN_CONTEXT_LEN = 512
    MIN_SUB_SEQUENCES = 2
    MAX_SUB_SEQUENCES = 128
    
    # Preprocessing
    MAX_EXAMPLES_TO_GENERATE = 10000  # Limit for memory constraints
    EXAMPLES_PER_FILE = 5  # Generate multiple examples from each file
    ### CHANGE: Add a specific batch size for the memory-intensive preprocessing step
    PREPROCESSING_BATCH_SIZE = 8 # Lower this if you still get OOM errors. 4 or 2 might be necessary.
    
    # System
    DEVICE = "mps"

# --- Setup Logging and Directories ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)


# --- 2. Keep your RoPE and Model Architecture (unchanged) ---
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _cache_cos_sin(self, seq_len, device):
        if self.seq_len_cached != seq_len:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
    
    def _apply_rotary_emb(self, x, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x_rotated = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        return (x * cos) + (x_rotated * sin)

    def forward(self, q, k):
        seq_len = q.shape[2]
        device = q.device
        self._cache_cos_sin(seq_len, device)
        q_rotated = self._apply_rotary_emb(q, self.cos_cached, self.sin_cached)
        k_rotated = self._apply_rotary_emb(k, self.cos_cached, self.sin_cached)
        return q_rotated, k_rotated

class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

class HierarchicalEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention = HierarchicalAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x

class HierarchicalBert(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ffn_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            HierarchicalEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        # self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# --- 3. Pre-processing and Caching Functions ---
def sensible_split(text, num_chunks):
    """Splits text into a number of chunks at sensible points (punctuation/newlines)."""
    split_points = [m.start() for m in re.finditer(r'[.?!]\s|\n', text)]
    if len(split_points) < num_chunks - 1:
        chunk_size = len(text) // num_chunks
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)][:num_chunks]
    chosen_indices = sorted(random.sample(split_points, num_chunks - 1))
    chunks = []
    start_idx = 0
    for split_idx in chosen_indices:
        chunks.append(text[start_idx:split_idx+1].strip())
        start_idx = split_idx + 1
    chunks.append(text[start_idx:].strip())
    return [chunk for chunk in chunks if chunk]

def preprocess_and_cache_embeddings(file_paths, tokenizer, base_model, device, cache_file_prefix, 
                                    max_examples=None, examples_per_file=4, batch_size=8): # Batch size from config
    """
    Pre-process files and create cached embeddings with batch processing.
    Now with more aggressive memory management.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{cache_file_prefix}_embeddings.pkl")
    
    if os.path.exists(cache_file):
        logging.info(f"Cache file {cache_file} already exists. Loading...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logging.info(f"Creating new cache file: {cache_file}")
    
    texts_to_process = []
    metadata = []
    
    pbar = tqdm(file_paths, desc=f"Reading files for {cache_file_prefix}")
    for file_path in pbar:
        if max_examples and len(texts_to_process) >= max_examples:
            break
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
            if len(full_text.strip()) < 100: continue
            
            for _ in range(examples_per_file):
                if max_examples and len(texts_to_process) >= max_examples: break
                # Tokenize the full text without truncation first
                tokenized_full = tokenizer(full_text, return_tensors="pt", truncation=False)
                input_ids = tokenized_full['input_ids'][0]

                # If the text is longer than max length, randomly sample a chunk
                if len(input_ids) > Config.MAX_CONTEXT_LEN:
                    start_idx = torch.randint(0, len(input_ids) - Config.MAX_CONTEXT_LEN + 1, (1,)).item()
                else:
                    start_idx = 0
                
                chunk_ids = input_ids[start_idx:start_idx + torch.randint(low=Config.MIN_CONTEXT_LEN, high=Config.MAX_CONTEXT_LEN, size=(1,)).item()]

                # Decode the chunk
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                
                num_sub_sequences = random.randint(Config.MIN_SUB_SEQUENCES, min(Config.MAX_SUB_SEQUENCES, len(chunk_text) // 50))
                sub_sequence_texts = sensible_split(chunk_text, num_sub_sequences)
                
                if len(sub_sequence_texts) < 2: continue
                
                texts_to_process.append((chunk_text, sub_sequence_texts))
                metadata.append({'source_file': file_path, 'num_sequences': len(sub_sequence_texts)})
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {e}")
            continue

    torch.save(texts_to_process, f"{cache_file_prefix}_texts_to_process.pt")
    torch.save(metadata, f"{cache_file_prefix}_metadata.pt")
    
    return []


class CachedEmbeddingDataset(Dataset):
    def __init__(self, cached_examples):
        self.examples = cached_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def cached_collate_fn(batch):
    max_len = max(item['num_sequences'] for item in batch)
    batch_size = len(batch)
    padded_inputs = torch.zeros(batch_size, max_len, Config.EMBEDDING_DIM)
    attention_mask = torch.zeros(batch_size, 1, 1, max_len)
    target_embeddings = torch.zeros(batch_size, Config.EMBEDDING_DIM)
    original_lengths = []
    
    for i, item in enumerate(batch):
        seq_len = item['num_sequences']
        original_lengths.append(seq_len)
        padded_inputs[i, :seq_len, :] = torch.from_numpy(item['input_embeddings'])
        attention_mask[i, 0, 0, :seq_len] = 1
        target_embeddings[i] = torch.from_numpy(item['target_embedding'])
    
    return {
        'input_embeddings': padded_inputs,
        'attention_mask': attention_mask,
        'target_embedding': target_embeddings,
        'original_lengths': torch.tensor(original_lengths)
    }

def criterion(pred:torch.Tensor, target:torch.Tensor):
    return nn.functional.mse_loss(pred, target)

# --- 4. Main Training Script ---
def main():
    logging.info(f"Using device: {Config.DEVICE}")
    
    # --- Load Base Model and Tokenizer (only for preprocessing) ---
    logging.info(f"Loading base model: {Config.BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
    base_model = AutoModel.from_pretrained(Config.BASE_MODEL_NAME).to(Config.DEVICE)
    base_model.eval()
    
    # logging.info(f"Loading data from: {Config.DATA_DIR}")
    # all_files = glob.glob(os.path.join(Config.DATA_DIR, "*.txt"))
    # if not all_files:
    #     logging.error(f"No .txt files found in {Config.DATA_DIR}. Please check the path.")
    #     return
    
    # # Using a smaller subset for demonstration/debugging if needed
    # # all_files = all_files[:100]
    
    # train_files, val_files = train_test_split(all_files, train_size=Config.TRAIN_SPLIT_RATIO, random_state=42)
    # logging.info(f"Found {len(all_files)} files. Training on {len(train_files)}, validating on {len(val_files)}.")
    
    # logging.info("Pre-processing and caching embeddings...")
    # train_examples = preprocess_and_cache_embeddings(
    #     train_files, tokenizer, base_model, Config.DEVICE, 
    #     "train", max_examples=Config.MAX_EXAMPLES_TO_GENERATE, 
    #     examples_per_file=Config.EXAMPLES_PER_FILE,
    #     batch_size=Config.PREPROCESSING_BATCH_SIZE # ### CHANGE: Use the new batch size
    # )
    # val_examples = preprocess_and_cache_embeddings(
    #     val_files, tokenizer, base_model, Config.DEVICE, 
    #     "val", max_examples=Config.MAX_EXAMPLES_TO_GENERATE // 5,
    #     examples_per_file=Config.EXAMPLES_PER_FILE,
    #     batch_size=Config.PREPROCESSING_BATCH_SIZE # ### CHANGE: Use the new batch size
    # )

    with open('embedding_cache_2/train_embeddings.pkl', 'rb') as f:
        train_examples = pickle.load(f)

    with open('embedding_cache_2/val_embeddings.pkl', 'rb') as f:
        val_examples = pickle.load(f)

    logging.info(f"Created {len(train_examples)} training examples and {len(val_examples)} validation examples")
    
    # --- Free up memory by deleting the large base model ---
    # del base_model
    # del tokenizer
    gc.collect() # ### CHANGE: Force garbage collection
    if Config.DEVICE == 'cuda':
        torch.cuda.empty_cache() # ### CHANGE: Clear cache after deleting the model
    
    # --- Create DataLoaders ---
    train_dataset = CachedEmbeddingDataset(train_examples)
    val_dataset = CachedEmbeddingDataset(val_examples)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=cached_collate_fn, num_workers=2, pin_memory=True if Config.DEVICE == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=cached_collate_fn, num_workers=2, pin_memory=True if Config.DEVICE == 'cuda' else False)
    
    logging.info("Initializing HierarchicalBert model")
    hierarchical_model = HierarchicalBert(
        embed_dim=Config.EMBEDDING_DIM,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_ATTENTION_HEADS,
        ffn_dim=Config.EMBEDDING_DIM * Config.FFN_DIM_MULTIPLIER
    ).to(Config.DEVICE)
    print(f"Number of params: {sum(p.numel() for p in hierarchical_model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(hierarchical_model.parameters(), lr=Config.LEARNING_RATE)
    # criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    # --- Training Loop ---
    for epoch in range(Config.EPOCHS):
        hierarchical_model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for batch in train_pbar:
            batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            output_embeddings = hierarchical_model(batch['input_embeddings'], batch['attention_mask'])
            last_seq_indices = batch['original_lengths'] - 1
            last_token_embeddings = output_embeddings[torch.arange(output_embeddings.size(0)), last_seq_indices]
            loss = criterion(last_token_embeddings, batch['target_embedding'])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        hierarchical_model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(Config.DEVICE) for k, v in batch.items()}
                output_embeddings = hierarchical_model(batch['input_embeddings'], batch['attention_mask'])
                last_seq_indices = batch['original_lengths'] - 1
                last_token_embeddings = output_embeddings[torch.arange(output_embeddings.size(0)), last_seq_indices]
                loss = criterion(last_token_embeddings, batch['target_embedding'])
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': hierarchical_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            logging.info(f"Validation loss decreased. Saving model to {checkpoint_path}")
    
    logging.info("Training complete.")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    # ### CHANGE: Wrap the main execution in a try...finally block
    try:
        main()
    except Exception as e:
        # Log any exception that causes the script to crash
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # This block will run NO MATTER WHAT.
        # It ensures that GPU memory is released if the script crashes or finishes.
        if torch.cuda.is_available():
            logging.info("Script finished or crashed. Clearing CUDA cache to release GPU memory.")
            torch.cuda.empty_cache()
            gc.collect()