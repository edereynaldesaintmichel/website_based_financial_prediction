"""Check table position embedding norms from a checkpoint."""
import torch, re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mlm_full/checkpoint_epoch3/full_model.pt"
sd = torch.load(path, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]

emb_keys = sorted([k for k in sd.keys() if "row_emb" in k or "col_emb" in k])
print(f"Embedding keys found: {len(emb_keys)}")
if not emb_keys:
    print("No table embeddings found — is this a TableFinancialModernBert checkpoint?")
    sys.exit(1)

layers = {}
for k in emb_keys:
    m = re.search(r"layers\.(\d+)\.attn\.(row_emb|col_emb)\.weight", k)
    if m:
        li = int(m.group(1))
        et = m.group(2)
        w = sd[k]
        if li not in layers:
            layers[li] = {}
        layers[li][et] = {
            "norm": w.norm().item(),
            "max": w.abs().max().item(),
            "mean": w.abs().mean().item(),
        }

print()
print(f"{'Layer':>6} | {'row norm':>10} {'row max':>10} {'row mean':>10} | {'col norm':>10} {'col max':>10} {'col mean':>10}")
print("-" * 85)
for li in sorted(layers.keys()):
    r = layers[li].get("row_emb", {})
    c = layers[li].get("col_emb", {})
    rn = r.get("norm", 0)
    rm = r.get("max", 0)
    ra = r.get("mean", 0)
    cn = c.get("norm", 0)
    cm = c.get("max", 0)
    ca = c.get("mean", 0)
    print(f"{li:6d} | {rn:10.6f} {rm:10.6f} {ra:10.6f} | {cn:10.6f} {cm:10.6f} {ca:10.6f}")
