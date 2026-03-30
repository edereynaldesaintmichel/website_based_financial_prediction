"""
Test CLS embedding importance in the T5-style model.

Chunks 2 full validation documents using the same intelligent chunking as the
training pipeline, then for each chunk:
1. Correct CLS: encoder CLS from the SAME chunk
2. Wrong CLS: encoder CLS from a RANDOM chunk of the OTHER document

Sweeps across multiple masking rates and averages over all chunks.

Usage:
    python -m tests.test_cls_importance
"""
import os
import random
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import FinancialBertTokenizer, FinancialModernBertConfig
from mlm_training_pipeline.chunk_markdown import chunk_file
from t5_style_training_pipeline.train import create_masked_inputs


CHECKPOINT = "checkpoints/t5_expanded_memory/model_only.pt"
PRETRAINED_ID = "answerdotai/ModernBERT-base"
MAX_LENGTH = 1024
NUM_TRIALS = 5  # masking trials per (chunk, mask_rate) pair
MASK_RATES = [0.15, 0.25, 0.45, 0.60, 0.80]
NUM_DOCS = 2
CHUNK_MAX_TOKENS = 512  # chunking target (will be truncated to 1024 after tokenization)
DOC_DIR = "training_data/processed/SEC_10k_markdown_tagged"
VAL_BUCKET = "t5_training_data/val/bucket_530.pt"


def load_model(checkpoint_path):
    """Load the T5 model from a checkpoint without needing the original encoder file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = ckpt["args"].get("num_magnitude_bins", 128)

    from t5_style_training_pipeline.decoder import T5StyleModel
    model = T5StyleModel(config)

    state_dict = {k.removeprefix("_orig_mod."): v
                  for k, v in ckpt["model_state_dict"].items()}
    del ckpt
    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    print(f"  Loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def tokenize_chunk_text(text, tokenizer, max_length=MAX_LENGTH):
    """Tokenize a chunk of text, truncating to max_length."""
    return tokenizer(text, padding=False, truncation=True,
                     max_length=max_length, return_tensors="pt")


def get_cls_embedding(model, encoded, device):
    input_ids = encoded["input_ids"].to(device)
    number_values = encoded["number_values"].to(device)
    is_number_mask = encoded["is_number_mask"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "mps"):
        enc_embeds = model._build_embeds(
            input_ids, number_values, is_number_mask,
            model._get_encoder_tok_embeddings(),
            model.encoder.number_embedder,
        )
        encoder_out = model.encoder.modernbert(
            inputs_embeds=enc_embeds, attention_mask=attention_mask
        )
    return encoder_out.last_hidden_state[:, 0, :].float()


def evaluate_reconstruction(model, encoded, cls_hidden, device, config,
                            tokenizer, mask_prob, num_trials=NUM_TRIALS):
    input_ids = encoded["input_ids"].to(device)
    number_values = encoded["number_values"].to(device)
    is_number_mask = encoded["is_number_mask"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    magnitude_sentinel = config.magnitude_max + 1.0
    vocab_size = config.vocab_size
    mag_min = config.magnitude_min
    mag_max = config.magnitude_max

    totals = {"loss": 0.0, "loss_text": 0.0, "loss_mag": 0.0,
              "text_acc": 0.0, "n_text_masked": 0, "n_num_masked": 0}

    for _ in range(num_trials):
        (dec_input_ids, dec_number_values, dec_is_number_mask,
         labels_text, labels_magnitude) = create_masked_inputs(
            input_ids, is_number_mask, number_values,
            mask_token_id=mask_token_id, pad_token_id=pad_token_id,
            magnitude_sentinel=magnitude_sentinel, vocab_size=vocab_size,
            magnitude_min=mag_min, magnitude_max=mag_max,
            mask_prob_min=mask_prob, mask_prob_max=mask_prob,
        )

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "mps"):
            dec_embeds = model._build_embeds(
                dec_input_ids, dec_number_values, dec_is_number_mask,
                model.decoder._get_tok_embeddings(),
                model.decoder.number_embedder,
            )
            text_logits, mag_logits = model.decoder(
                dec_embeds, cls_hidden, attention_mask
            )
        text_logits = text_logits.float()
        mag_logits = mag_logits.float()

        text_mask = labels_text.view(-1) != -100
        if text_mask.any():
            masked_logits = text_logits.view(-1, vocab_size)[text_mask]
            masked_labels = labels_text.view(-1)[text_mask]
            loss_text = F.cross_entropy(masked_logits, masked_labels).item()
            acc = (masked_logits.argmax(dim=-1) == masked_labels).float().mean().item()
            n_text = text_mask.sum().item()
        else:
            loss_text, acc, n_text = 0.0, 0.0, 0

        mag_mask = labels_magnitude.view(-1) != -100
        if mag_mask.any():
            target_mags = labels_magnitude.view(-1)[mag_mask]
            pred_mag_logits = mag_logits.view(-1, config.num_magnitude_bins)[mag_mask]
            n_bins = config.num_magnitude_bins
            norm_pos = (target_mags.clamp(mag_min, mag_max) - mag_min) / (mag_max - mag_min) * (n_bins - 1)
            lower_idx = norm_pos.floor().long()
            upper_idx = norm_pos.ceil().long()
            weight_upper = norm_pos - lower_idx.float()
            weight_lower = 1.0 - weight_upper
            log_probs = F.log_softmax(pred_mag_logits, dim=-1)
            loss_mag = -(
                weight_lower * log_probs.gather(1, lower_idx.unsqueeze(1)).squeeze(1)
                + weight_upper * log_probs.gather(1, upper_idx.unsqueeze(1)).squeeze(1)
            ).mean().item()
            n_num = mag_mask.sum().item()
        else:
            loss_mag, n_num = 0.0, 0

        totals["loss"] += loss_text + loss_mag
        totals["loss_text"] += loss_text
        totals["loss_mag"] += loss_mag
        totals["text_acc"] += acc
        totals["n_text_masked"] += n_text
        totals["n_num_masked"] += n_num

    n = num_trials
    return {k: v / n for k, v in totals.items()}


def get_val_filenames(bucket_path, doc_dir, num_docs):
    """Get filenames from validation bucket, verify they exist."""
    data = torch.load(bucket_path, map_location="cpu", weights_only=False)
    unique = sorted(set(data["source_files"]))
    del data

    existing = [f for f in unique if os.path.exists(os.path.join(doc_dir, f))]
    random.seed(42)
    random.shuffle(existing)
    return existing[:num_docs]


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"\nLoading model from {CHECKPOINT}...")
    model = load_model(CHECKPOINT)
    model = model.to(device)
    config = model.config
    tokenizer = FinancialBertTokenizer(PRETRAINED_ID)

    # Get 2 validation filenames
    filenames = get_val_filenames(VAL_BUCKET, DOC_DIR, NUM_DOCS)
    print(f"\nSelected {len(filenames)} validation documents:")
    for f in filenames:
        print(f"  {f}")

    # Chunk both documents intelligently
    print(f"\nChunking documents (max_tokens={CHUNK_MAX_TOKENS})...")
    doc_chunks = []  # list of 2 lists of tokenized chunks
    doc_cls = []     # list of 2 lists of CLS embeddings

    for fname in filenames:
        path = os.path.join(DOC_DIR, fname)
        raw_chunks = chunk_file(path, max_tokens=CHUNK_MAX_TOKENS)
        print(f"\n  {fname}: {len(raw_chunks)} chunks")

        chunks_encoded = []
        chunks_cls = []
        for ci, chunk in enumerate(raw_chunks):
            encoded = tokenize_chunk_text(chunk["text"], tokenizer)
            seq_len = encoded["input_ids"].shape[1]
            n_nums = encoded["is_number_mask"].sum().int().item()
            cls = get_cls_embedding(model, encoded, device).cpu()
            if device.type == "mps":
                torch.mps.empty_cache()
            print(f"    chunk {ci:2d}: seq_len={seq_len:4d}, numbers={n_nums:3d}, "
                  f"est_tokens={chunk['estimated_tokens']}")
            chunks_encoded.append(encoded)
            chunks_cls.append(cls)

        doc_chunks.append(chunks_encoded)
        doc_cls.append(chunks_cls)

    total_chunks = sum(len(c) for c in doc_chunks)
    print(f"\nTotal: {total_chunks} chunks across {len(filenames)} documents")

    # Run sweep
    print(f"\n{'='*80}")
    print(f"  CLS IMPORTANCE SWEEP: {total_chunks} chunks x {len(MASK_RATES)} mask rates")
    print(f"  {NUM_TRIALS} masking trials per condition")
    print(f"  Correct CLS = same chunk, Wrong CLS = random chunk from other doc")
    print(f"{'='*80}")

    results = {rate: {"correct": {"loss": 0, "text": 0, "mag": 0, "acc": 0},
                      "wrong":   {"loss": 0, "text": 0, "mag": 0, "acc": 0},
                      "n": 0}
               for rate in MASK_RATES}

    random.seed(123)

    for rate in MASK_RATES:
        print(f"\n--- Mask rate: {rate*100:.0f}% ---")
        chunk_idx = 0
        for doc_i in range(len(doc_chunks)):
            other_doc_i = 1 - doc_i  # the other document (0 or 1)
            other_cls_list = doc_cls[other_doc_i]

            for ci, (enc, cls_correct) in enumerate(zip(doc_chunks[doc_i], doc_cls[doc_i])):
                # Wrong CLS: random chunk from the other document
                cls_wrong = random.choice(other_cls_list)

                m_correct = evaluate_reconstruction(
                    model, enc, cls_correct.to(device), device, config, tokenizer, mask_prob=rate)
                m_wrong = evaluate_reconstruction(
                    model, enc, cls_wrong.to(device), device, config, tokenizer, mask_prob=rate)
                if device.type == "mps":
                    torch.mps.empty_cache()

                results[rate]["correct"]["loss"] += m_correct["loss"]
                results[rate]["correct"]["text"] += m_correct["loss_text"]
                results[rate]["correct"]["mag"] += m_correct["loss_mag"]
                results[rate]["correct"]["acc"] += m_correct["text_acc"]
                results[rate]["wrong"]["loss"] += m_wrong["loss"]
                results[rate]["wrong"]["text"] += m_wrong["loss_text"]
                results[rate]["wrong"]["mag"] += m_wrong["loss_mag"]
                results[rate]["wrong"]["acc"] += m_wrong["text_acc"]
                results[rate]["n"] += 1

                chunk_idx += 1
                seq_len = enc["input_ids"].shape[1]
                print(f"  chunk {chunk_idx:2d}/{total_chunks} (doc{doc_i} c{ci}, len={seq_len})  "
                      f"correct: loss={m_correct['loss']:.3f} acc={m_correct['text_acc']:.1%}  "
                      f"wrong: loss={m_wrong['loss']:.3f} acc={m_wrong['text_acc']:.1%}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  RESULTS (averaged over {total_chunks} chunks from {len(filenames)} documents)")
    print(f"  max_length={MAX_LENGTH}, chunk_max_tokens={CHUNK_MAX_TOKENS}")
    print(f"{'='*80}")
    print(f"\n  {'Mask%':>6}  {'Corr Loss':>10}  {'Wrong Loss':>11}  {'Loss +%':>8}  "
          f"{'Corr Acc':>9}  {'Wrong Acc':>10}  {'Acc Drop':>9}  "
          f"{'Corr Mag':>9}  {'Wrong Mag':>10}  {'Mag +%':>8}")
    print(f"  {'-'*105}")

    for rate in MASK_RATES:
        n = results[rate]["n"]
        c = {k: v / n for k, v in results[rate]["correct"].items()}
        w = {k: v / n for k, v in results[rate]["wrong"].items()}
        dl = (w["loss"] - c["loss"]) / max(c["loss"], 1e-8) * 100
        da = c["acc"] - w["acc"]
        dm = (w["mag"] - c["mag"]) / max(c["mag"], 1e-8) * 100
        print(f"  {rate*100:5.0f}%  {c['loss']:10.4f}  {w['loss']:11.4f}  {dl:+7.1f}%  "
              f"{c['acc']:8.1%}  {w['acc']:9.1%}  {da:+8.1%}  "
              f"{c['mag']:9.4f}  {w['mag']:10.4f}  {dm:+7.1f}%")


if __name__ == "__main__":
    main()
