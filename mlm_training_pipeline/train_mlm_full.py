"""
Full-parameter MLM training for FinancialModernBert (no LoRA).

Trains ALL parameters of the ModernBERT backbone, number embedder, and
number head.

Data preparation produces whole-document tokenizations (documents.pt).
At the start of each epoch, documents are chunked with random boundaries
and variable lengths, then sorted by length and batched by token budget.

Wikipedia regularization documents (.txt source files) are interleaved
with financial data (.md source files) at a configurable ratio.

Usage:
    python -m mlm_training_pipeline.train_mlm_full \
        --data mlm_data/documents.pt \
        --tokens_per_batch 8192 \
        --epochs 3 \
        --lr 5e-5
"""
import argparse
import bisect
import math
import os
import random
import sys
import time
import threading
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import build_model, FinancialBertTokenizer, TABLE_START_ID, TABLE_END_ID
from split_utils import split_documents, is_val_document


def _get_create_masked_inputs():
    """Lazy import to avoid circular import chain."""
    from t5_style_training_pipeline.train import create_masked_inputs
    return create_masked_inputs


CHUNK_MIN = 128
CHUNK_MAX = 4096
CHUNK_HARD_MAX = 8192  # absolute upper bound – no chunk may exceed this

# Flash Attention is O(n) in memory but with a constant that grows with
# sequence length.  We apply a mild quadratic penalty so that batches
# with longer sequences get a slightly smaller token budget, preventing
# OOM on the longest-chunk batches.
# Effective budget = base_budget / (1 + LENGTH_PENALTY * (max_len / CHUNK_MAX))
LENGTH_PENALTY = 0.15


# ─── Token info ──────────────────────────────────────────────────────────────

def get_token_info(pretrained_id):
    """Extract special token IDs needed for training."""
    tokenizer = FinancialBertTokenizer(pretrained_id)
    base = tokenizer.base_tokenizer
    newline_ids = set(base.encode("\n", add_special_tokens=False))
    info = {
        "cls_id": base.cls_token_id,
        "sep_id": base.sep_token_id,
        "pad_id": base.pad_token_id,
        "mask_id": tokenizer.mask_token_id,
        "newline_ids": newline_ids,
    }
    del tokenizer
    return info


# ─── Chunking ────────────────────────────────────────────────────────────────

def get_boundaries(input_ids, newline_ids):
    """Find chunk boundary candidates: positions after newline tokens,
    excluding newlines that fall inside TABLE_START..TABLE_END regions.

    Works in content space (CLS/SEP stripped: position 0 = first content token).
    Returns sorted list of boundary positions.
    """
    content = input_ids[1:-1]  # strip CLS at 0 and SEP at -1

    # Build mask of positions inside tables using cumsum
    starts = (content == TABLE_START_ID)
    ends = (content == TABLE_END_ID)
    # depth = cumsum(starts) - cumsum(ends shifted right by 1)
    # A position is inside a table if depth > 0.
    # TABLE_START itself is inside; TABLE_END itself is inside (cleared after).
    depth = starts.long().cumsum(0) - ends.long().cumsum(0).roll(1, 0)
    depth[0] = starts[0].long()  # fix rolled position
    in_table = depth > 0

    # Newline mask (excluding inside-table positions)
    nl_mask = torch.zeros_like(content, dtype=torch.bool)
    for nl_id in newline_ids:
        nl_mask |= (content == nl_id)
    nl_mask &= ~in_table

    # Boundary = position AFTER the newline (start of next chunk)
    return (nl_mask.nonzero(as_tuple=False).squeeze(-1) + 1).tolist()


def snap_to_boundary(target, boundaries, min_pos):
    """Snap a target position to the nearest boundary candidate."""
    if not boundaries:
        return target
    idx = bisect.bisect_left(boundaries, target)
    candidates = []
    if idx > 0 and boundaries[idx - 1] > min_pos:
        candidates.append(boundaries[idx - 1])
    if idx < len(boundaries):
        candidates.append(boundaries[idx])
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))


MIN_TAIL_CHUNK = 16  # absorb trailing chunks shorter than this


def chunk_document(doc, cls_id, sep_id, newline_ids):
    """Chunk a single document into variable-length pieces.

    Each chunk gets CLS prepended and SEP appended.
    Chunk target sizes are random per chunk.
    """
    content_len = len(doc["input_ids"]) - 2  # strip CLS/SEP
    if content_len < 1:
        return []

    boundaries = get_boundaries(doc["input_ids"], newline_ids)

    # Generate chunk spans with random target per chunk
    spans = []
    pos = 0
    while pos < content_len:
        target = random.randint(CHUNK_MIN, CHUNK_MAX)
        raw_end = min(pos + target, content_len)

        if raw_end < content_len:
            end = snap_to_boundary(raw_end, boundaries, pos)
            if end <= pos:
                end = min(pos + target, content_len)
            # Absorb tiny tail
            if content_len - end < MIN_TAIL_CHUNK:
                end = content_len
        else:
            end = content_len

        # Hard cap: no chunk may exceed CHUNK_HARD_MAX content tokens
        # (+2 for CLS/SEP added later)
        if end - pos > CHUNK_HARD_MAX - 2:
            end = pos + CHUNK_HARD_MAX - 2

        spans.append((pos, end))
        pos = end

    # Extract chunks with CLS/SEP wrapping
    chunks = []
    for start, end in spans:
        offset = 1 + start  # +1 to skip document CLS
        length = end - start

        c_ids = doc["input_ids"][offset:offset + length]
        c_mask = doc["is_number_mask"][offset:offset + length]
        c_vals = doc["number_values"][offset:offset + length]

        chunk_ids = torch.cat([
            torch.tensor([cls_id], dtype=c_ids.dtype), c_ids,
            torch.tensor([sep_id], dtype=c_ids.dtype),
        ])
        chunk_mask = torch.cat([
            torch.tensor([False]), c_mask.bool(),
            torch.tensor([False]),
        ])
        chunk_vals = torch.cat([
            torch.tensor([0.0]), c_vals.float(),
            torch.tensor([0.0]),
        ])

        chunks.append({
            "input_ids": chunk_ids,
            "is_number_mask": chunk_mask,
            "number_values": chunk_vals,
            "seq_length": len(chunk_ids),
            "source_file": doc["source_file"],
        })

    return chunks


def chunk_all_documents(documents, cls_id, sep_id, newline_ids):
    """Chunk all documents, return list of chunk dicts."""
    # Use single thread – the per-document tensor ops are tiny and the
    # thread-pool spawn overhead dominates otherwise (460x slower).
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_document(doc, cls_id, sep_id, newline_ids))
    finally:
        torch.set_num_threads(prev_threads)
    return all_chunks


# ─── Batching ────────────────────────────────────────────────────────────────

def form_batches_1d(chunks, token_budget, bucket_width=16):
    """Bucket chunks by length, form batches per bucket, shuffle batch order.

    Applies a mild length penalty: longer sequences get a reduced effective
    budget to account for Flash Attention's per-sequence memory overhead.

    Returns list of batches (each batch is a list of chunk dicts).
    """
    buckets = defaultdict(list)
    for chunk in chunks:
        key = (chunk["seq_length"] + bucket_width - 1) // bucket_width * bucket_width
        buckets[key].append(chunk)

    all_batches = []
    for bucket_len, bucket_items in buckets.items():
        random.shuffle(bucket_items)
        # Reduce budget for longer sequences
        effective_budget = token_budget / (1 + LENGTH_PENALTY * (bucket_len / CHUNK_MAX))
        batch_size = max(1, int(effective_budget / bucket_len))
        for i in range(0, len(bucket_items), batch_size):
            all_batches.append(bucket_items[i:i + batch_size])

    random.shuffle(all_batches)
    return all_batches


def pad_and_collate(chunks, pad_id):
    """Pad and stack chunk dicts into batch tensors (vectorized)."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [c["input_ids"].long() for c in chunks], batch_first=True, padding_value=pad_id,
    )
    is_number_mask = torch.nn.utils.rnn.pad_sequence(
        [c["is_number_mask"].long() for c in chunks], batch_first=True, padding_value=0,
    )
    number_values = torch.nn.utils.rnn.pad_sequence(
        [c["number_values"].float() for c in chunks], batch_first=True, padding_value=0.0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(c["seq_length"], dtype=torch.long) for c in chunks],
        batch_first=True, padding_value=0,
    )
    return input_ids, is_number_mask, number_values, attention_mask


def _prefetch_iter(batches, prepare_fn):
    """Yield prepared batches, prefetching the next one on a background thread."""
    result = [None]
    error = [None]

    def _worker(batch):
        try:
            result[0] = prepare_fn(batch)
        except Exception as e:
            error[0] = e

    it = iter(batches)
    # Prepare the first batch on the main thread
    first = next(it, None)
    if first is None:
        return
    current = prepare_fn(first)

    # Kick off prefetch for the second batch
    nxt = next(it, None)
    thread = None
    if nxt is not None:
        thread = threading.Thread(target=_worker, args=(nxt,))
        thread.start()

    while True:
        yield current
        if thread is not None:
            thread.join()
            if error[0] is not None:
                raise error[0]
            current = result[0]
            result[0] = None
            error[0] = None
            nxt = next(it, None)
            if nxt is not None:
                thread = threading.Thread(target=_worker, args=(nxt,))
                thread.start()
            else:
                thread = None
        else:
            break


# ─── Training ────────────────────────────────────────────────────────────────

def run_epoch(model, documents, reg_documents, optimizer, scheduler, scaler,
              device, epoch, args, tok_info, use_amp, amp_dtype, training=True):
    """Run one training or validation epoch."""
    prefix = "Train" if training else "Val"
    if training:
        model.train()
    else:
        model.eval()

    # Deterministic chunking per epoch
    rng_state = random.getstate()
    random.seed(args.seed + epoch + (0 if training else 10000))

    cls_id = tok_info["cls_id"]
    sep_id = tok_info["sep_id"]
    pad_id = tok_info["pad_id"]
    mask_id = tok_info["mask_id"]
    nl_ids = tok_info["newline_ids"]

    # ─── Step 1: Chunk all documents ────────────────────────────────────
    print(f"  Chunking documents...")
    fin_chunks = chunk_all_documents(documents, cls_id, sep_id, nl_ids)
    n_fin = len(fin_chunks)
    avg_fin = sum(c["seq_length"] for c in fin_chunks) / max(n_fin, 1)

    reg_chunks = []
    if reg_documents:
        reg_chunks = chunk_all_documents(reg_documents, cls_id, sep_id, nl_ids)
    n_reg = len(reg_chunks)
    avg_reg = sum(c["seq_length"] for c in reg_chunks) / max(n_reg, 1)

    print(f"  Chunks: {n_fin} financial (avg {avg_fin:.0f}), "
          f"{n_reg} regularization (avg {avg_reg:.0f})")

    # ─── Step 2: Form batches ───────────────────────────────────────────
    fin_batches = form_batches_1d(fin_chunks, args.tokens_per_batch)
    del fin_chunks

    reg_batches = []
    if reg_chunks:
        reg_batches = form_batches_1d(reg_chunks, args.tokens_per_batch)
    del reg_chunks

    # Estimate padding waste
    total_useful = 0
    total_padded = 0
    for batch in fin_batches + reg_batches:
        max_len = max(c["seq_length"] for c in batch)
        total_useful += sum(c["seq_length"] for c in batch)
        total_padded += max_len * len(batch)
    waste = 1.0 - total_useful / max(total_padded, 1)
    print(f"  {len(fin_batches)} financial batches, "
          f"{len(reg_batches)} regularization batches, "
          f"padding waste: {waste:.1%}")

    random.setstate(rng_state)

    # ─── Step 3: Train/eval ─────────────────────────────────────────────
    total_loss = 0.0
    total_loss_text = 0.0
    total_loss_mag = 0.0
    n_batches = 0
    MA_WINDOW = 100
    ma = {k: deque(maxlen=MA_WINDOW) for k in ("loss", "text", "mag", "reg")}

    # Build interleaved batch iterator: financial batches with probabilistic
    # regularization insertion
    reg_iter = iter(reg_batches) if reg_batches else None

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)

    magnitude_sentinel = model.config.magnitude_max + 1.0 if hasattr(model, 'config') else 13.0
    create_masked_inputs = _get_create_masked_inputs()

    def _prepare_batch(batch_chunks):
        """Collate, pin memory, transfer to device, and apply MLM masking."""
        ids, num_mask, num_vals, attn_mask = pad_and_collate(batch_chunks, pad_id)
        ids = ids.pin_memory().to(device, non_blocking=True)
        num_mask = num_mask.pin_memory().to(device, non_blocking=True)
        num_vals = num_vals.pin_memory().to(device, non_blocking=True)
        attn_mask = attn_mask.pin_memory().to(device, non_blocking=True)
        m_ids, m_nums, m_num_mask, labels_t, labels_m = create_masked_inputs(
            ids, num_mask, num_vals,
            mask_token_id=mask_id, pad_token_id=pad_id,
            magnitude_sentinel=magnitude_sentinel,
            vocab_size=model.config.vocab_size,
            magnitude_min=model.config.magnitude_min,
            magnitude_max=model.config.magnitude_max,
            mask_prob_min=args.mask_prob_min,
            mask_prob_max=args.mask_prob_max,
        )
        return m_ids, m_nums, m_num_mask, labels_t, labels_m, attn_mask

    try:
        pbar = tqdm(
            _prefetch_iter(fin_batches, _prepare_batch),
            total=len(fin_batches), desc=f"  {prefix}", unit="batch",
        )
        for m_ids, m_nums, m_num_mask, labels_t, labels_m, attn_mask in pbar:

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=m_ids,
                    attention_mask=attn_mask,
                    is_number_mask=m_num_mask,
                    number_values=m_nums,
                    labels_text=labels_t,
                    labels_magnitude=labels_m,
                )

            loss = outputs["loss"]

            if training:
                scaler.scale(loss).backward()

                # Regularization batch (interleaved)
                if reg_iter and random.random() < args.regularization_ratio:
                    try:
                        reg_batch = next(reg_iter)
                    except StopIteration:
                        reg_iter = iter(reg_batches)
                        reg_batch = next(reg_iter)

                    rm_ids, rm_nums, rm_nm, rl_t, rl_m, r_am = _prepare_batch(reg_batch)

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        reg_outputs = model(
                            input_ids=rm_ids,
                            attention_mask=r_am,
                            is_number_mask=rm_nm,
                            number_values=rm_nums,
                            labels_text=rl_t,
                            labels_magnitude=rl_m,
                        )
                    reg_loss = reg_outputs["loss"]
                    scaler.scale(reg_loss).backward()
                    ma["reg"].append(reg_loss.item())

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            total_loss_text += outputs["loss_text"].item()
            total_loss_mag += outputs["loss_mag"].item()
            ma["loss"].append(loss.item())
            ma["text"].append(outputs["loss_text"].item())
            ma["mag"].append(outputs["loss_mag"].item())
            n_batches += 1

            if n_batches % 20 == 0:
                postfix = {
                    "loss": f"{sum(ma['loss'])/len(ma['loss']):.4f}",
                    "txt": f"{sum(ma['text'])/len(ma['text']):.4f}",
                    "mag": f"{sum(ma['mag'])/len(ma['mag']):.4f}",
                }
                if training:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                if ma["reg"]:
                    postfix["reg"] = f"{sum(ma['reg'])/len(ma['reg']):.4f}"
                pbar.set_postfix(**postfix)

    finally:
        torch.set_grad_enabled(prev_grad)

    avg_loss = total_loss / max(n_batches, 1)
    avg_lt = total_loss_text / max(n_batches, 1)
    avg_lm = total_loss_mag / max(n_batches, 1)

    return {
        "loss": avg_loss,
        "loss_text": avg_lt,
        "loss_mag": avg_lm,
        "n_batches": n_batches,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def train(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision('medium')

    if device.type == "cuda" and args.dtype in ("fp16", "bf16"):
        use_amp = True
        amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        use_scaler = args.dtype == "fp16"
    else:
        use_amp = False
        amp_dtype = torch.float32
        use_scaler = False
    print(f"Device: {device}, dtype: {args.dtype}" + (" (AMP)" if use_amp else ""))

    # Build model
    print("Building model (full fine-tuning, no LoRA)...")
    model = build_model(args.model_name)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    # ------------------------------------------------------------------
    # Optimizer (set up before torch.compile to access submodules directly)
    # ------------------------------------------------------------------
    number_params = set()
    for m in (model.number_embedder, model.number_head):
        for p in m.parameters():
            number_params.add(id(p))

    backbone_group = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in number_params]
    number_group = [p for p in model.parameters()
                    if p.requires_grad and id(p) in number_params]

    print(f"\nParam groups -- backbone: {sum(p.numel() for p in backbone_group):,} params @ lr={args.lr}, "
          f"number: {sum(p.numel() for p in number_group):,} params @ lr={args.number_lr}")

    optimizer = torch.optim.AdamW([
        {"params": backbone_group, "lr": args.lr},
        {"params": number_group, "lr": args.number_lr},
    ], weight_decay=args.weight_decay)

    # Token info
    tok_info = get_token_info(args.model_name)

    # ------------------------------------------------------------------
    # Load documents
    # ------------------------------------------------------------------
    print(f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)

    # Split into financial (.md) vs regularization (.txt) by source_file extension
    financial_docs = [d for d in documents if not d["source_file"].endswith(".txt")]
    reg_docs = [d for d in documents if d["source_file"].endswith(".txt")]
    del documents

    # Split financial docs into train/val using deterministic hash
    train_docs, val_docs = split_documents(financial_docs)
    del financial_docs

    print(f"  {len(train_docs)} train, {len(val_docs)} val financial documents")
    print(f"  {len(reg_docs)} regularization documents")

    total_train_tokens = sum(d["seq_length"] for d in train_docs)
    total_reg_tokens = sum(d["seq_length"] for d in reg_docs)
    print(f"  Train tokens: {total_train_tokens:,}, regularization tokens: {total_reg_tokens:,}")

    # Estimate total steps (rough: total_tokens / budget * epochs)
    est_batches_per_epoch = total_train_tokens / args.tokens_per_batch
    total_steps = int(est_batches_per_epoch * args.epochs)
    lr_warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < lr_warmup_steps:
            return step / max(1, lr_warmup_steps)
        progress = (step - lr_warmup_steps) / max(1, total_steps - lr_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, lr_lambda])
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(f"Estimated ~{est_batches_per_epoch:.0f} batches/epoch, {total_steps} total steps")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(tag, epoch, global_step, train_m=None, val_m=None):
        if not args.save_dir:
            return
        save_path = os.path.join(args.save_dir, f"checkpoint_{tag}")
        os.makedirs(save_path, exist_ok=True)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        if train_m:
            state["train_loss"] = train_m["loss"]
        if val_m:
            state["val_loss"] = val_m["loss"]
        torch.save(state, os.path.join(save_path, "full_model.pt"))
        print(f"  Saved checkpoint to {save_path} (epoch={epoch}, step={global_step})")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0

    if args.resume_from:
        ckpt_file = os.path.join(args.resume_from, "full_model.pt")
        print(f"Resuming from {ckpt_file}...")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        # Strip _orig_mod. prefix only if model is not compiled
        if not args.compile:
            state_dict = {k.replace("_orig_mod.", ""): v
                          for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"  Resumed at epoch={start_epoch}, global_step={global_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        t0 = time.time()
        train_m = run_epoch(
            model, train_docs, reg_docs, optimizer, scheduler, scaler,
            device, epoch, args, tok_info, use_amp, amp_dtype, training=True)
        t_train = time.time() - t0
        global_step += train_m["n_batches"]

        print(f"\n  Train: loss={train_m['loss']:.4f} "
              f"(text={train_m['loss_text']:.4f}, mag={train_m['loss_mag']:.4f}) "
              f"[{t_train:.0f}s]")

        # Validate
        t0 = time.time()
        val_m = run_epoch(
            model, val_docs, None, optimizer, scheduler, scaler,
            device, epoch, args, tok_info, use_amp, amp_dtype, training=False)
        t_val = time.time() - t0

        print(f"  Val:   loss={val_m['loss']:.4f} "
              f"(text={val_m['loss_text']:.4f}, mag={val_m['loss_mag']:.4f}) "
              f"[{t_val:.0f}s]")

        # Save checkpoint
        save_checkpoint(
            f"epoch{epoch + 1}", epoch + 1, global_step,
            train_m=train_m, val_m=val_m)

    print("\nTraining complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Full-parameter MLM training for FinancialModernBert")
    parser.add_argument("--data", default="mlm_data/documents.pt",
                        help="Path to documents.pt (from prepare_dataset.py)")
    parser.add_argument("--save_dir", default="checkpoints/mlm_full",
                        help="Checkpoint save directory")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base",
                        help="Base model")
    parser.add_argument("--tokens_per_batch", type=int, default=8192,
                        help="Target total tokens per batch")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Backbone learning rate")
    parser.add_argument("--number_lr", type=float, default=2e-4,
                        help="Learning rate for number_embedder + number_head")
    parser.add_argument("--mask_prob_min", type=float, default=0.15,
                        help="Minimum mask probability per sequence")
    parser.add_argument("--mask_prob_max", type=float, default=0.15,
                        help="Maximum mask probability per sequence "
                             "(set > min for variable masking)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps before cosine decay")
    parser.add_argument("--regularization_ratio", type=float, default=0.3,
                        help="Probability of inserting a regularization batch "
                             "after each financial batch (0.3 = ~30%%)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
