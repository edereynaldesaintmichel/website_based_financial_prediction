"""
Experiment: average pairwise cosine similarity of ModernBERT last-hidden-state
token vectors (excluding CLS and SEP), bucketed by sequence length.

Takes chunks from a few training_data/processed/wikipedia documents, truncates
them to target lengths from 128 to 512, runs them through ModernBERT-base, and
reports the mean off-diagonal pairwise cosine similarity of the token hidden
states per length bucket.

Usage:
    python -m tests.test_modernbert_hidden_cosine
"""
import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "answerdotai/ModernBERT-base"
WIKI_DIR = "training_data/processed/wikipedia"
NUM_DOCS = 5
CHUNKS_PER_DOC = 3
LENGTH_BUCKETS = [128, 192, 256, 384, 512]


def strip_number_tags(text):
    return re.sub(r"</?number>", "", text)


def pick_docs(wiki_dir, n):
    files = sorted(f for f in os.listdir(wiki_dir) if f.endswith(".txt"))
    # Evenly spaced picks across the directory.
    step = max(1, len(files) // n)
    return [os.path.join(wiki_dir, files[i * step]) for i in range(n)]


def mean_pairwise_cosine(hidden):
    """Mean off-diagonal pairwise cosine sim over token vectors.

    hidden: (L, D) tensor of token hidden states.
    """
    normed = F.normalize(hidden, dim=-1)
    sim = normed @ normed.t()  # (L, L)
    L = sim.shape[0]
    # Exclude the diagonal.
    off_diag_sum = sim.sum() - sim.diag().sum()
    return (off_diag_sum / (L * (L - 1))).item()


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    doc_paths = pick_docs(WIKI_DIR, NUM_DOCS)
    print(f"Sampling {CHUNKS_PER_DOC} chunks from each of {len(doc_paths)} docs:")
    for p in doc_paths:
        print(f"  {os.path.basename(p)}")

    # Tokenize each full doc once (no truncation, no special tokens), then
    # slice out several chunks per doc. We'll re-tokenize per bucket by
    # decoding the slice and letting the tokenizer add CLS/SEP.
    doc_token_ids = []
    for p in doc_paths:
        with open(p, "r") as f:
            text = strip_number_tags(f.read())
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        doc_token_ids.append(ids)

    # Pick non-overlapping chunk start positions per doc, spaced across the doc.
    max_bucket = max(LENGTH_BUCKETS)
    chunk_slices = []  # list of token-id tensors, each at least max_bucket long
    for ids in doc_token_ids:
        if ids.numel() < max_bucket + 8:
            continue
        usable = ids.numel() - max_bucket
        for k in range(CHUNKS_PER_DOC):
            start = (usable * k) // max(1, CHUNKS_PER_DOC)
            chunk_slices.append(ids[start:start + max_bucket])

    print(f"\nCollected {len(chunk_slices)} chunks (each >= {max_bucket} tokens).")

    print(f"\n{'='*60}")
    print(f"  Mean pairwise cosine sim of token hidden states")
    print(f"  (excluding CLS at position 0 and SEP at the end)")
    print(f"{'='*60}")
    print(f"\n  {'seq_len':>8}  {'input_cos':>10}  {'in_std':>8}  "
          f"{'hidden_cos':>11}  {'hid_std':>8}  "
          f"{'in_vs_hid':>10}  {'ivh_std':>8}  "
          f"{'pool_xdoc':>10}  {'n':>4}")
    print(f"  {'-'*95}")

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    embed_layer = model.get_input_embeddings()

    for bucket in LENGTH_BUCKETS:
        # bucket = target full seq length including CLS+SEP.
        inner = bucket - 2
        input_sims = []
        hidden_sims = []
        in_vs_hid_sims = []
        pooled = []  # one mean-pooled vector per chunk, for cross-doc cosine.
        for chunk_ids in chunk_slices:
            piece = chunk_ids[:inner]
            input_ids = torch.cat([
                torch.tensor([cls_id]),
                piece,
                torch.tensor([sep_id]),
            ]).unsqueeze(0).to(device)
            attn = torch.ones_like(input_ids)

            with torch.no_grad():
                input_embeds = embed_layer(input_ids)[0]
                out = model(input_ids=input_ids, attention_mask=attn)
            hidden = out.last_hidden_state[0]
            # Drop CLS (position 0) and SEP (last position).
            in_inner = input_embeds[1:-1].float()
            hid_inner = hidden[1:-1].float()
            input_sims.append(mean_pairwise_cosine(in_inner))
            hidden_sims.append(mean_pairwise_cosine(hid_inner))
            # Per-position cosine between each token's input embedding and its
            # final hidden state, averaged over positions.
            per_pos = F.cosine_similarity(in_inner, hid_inner, dim=-1)
            in_vs_hid_sims.append(per_pos.mean().item())
            pooled.append(hid_inner.mean(dim=0).cpu())

        in_t = torch.tensor(input_sims)
        hid_t = torch.tensor(hidden_sims)
        ivh_t = torch.tensor(in_vs_hid_sims)
        pooled_mat = torch.stack(pooled)  # (n_chunks, D)
        pool_xdoc = mean_pairwise_cosine(pooled_mat)
        print(f"  {bucket:8d}  {in_t.mean().item():10.4f}  {in_t.std().item():8.4f}  "
              f"{hid_t.mean().item():11.4f}  {hid_t.std().item():8.4f}  "
              f"{ivh_t.mean().item():10.4f}  {ivh_t.std().item():8.4f}  "
              f"{pool_xdoc:10.4f}  {len(hidden_sims):4d}")


if __name__ == "__main__":
    main()
