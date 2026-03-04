"""
Numeracy test for nomic-ai/nomic-embed-text-v1.5.

For each test group, an "anchor" sentence has a clear semantic magnitude
(e.g. "very powerful", "very hot"). Two candidate sentences express the same
concept with a small vs. a large number. If the model understands numeracy,
the candidate whose number matches the anchor's magnitude should get a higher
cosine similarity score.

Expected direction legend:
  HIGH  – the high-number sentence should be more similar to the anchor
  LOW   – the low-number sentence should be more similar to the anchor
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# ---------------------------------------------------------------------------
# Test cases: (anchor, low_sentence, high_sentence, expected_winner)
#   expected_winner ∈ {"HIGH", "LOW"}
# ---------------------------------------------------------------------------
TEST_GROUPS = [
    # --- BHP / engine power ---
    {
        "category": "Engine power",
        "anchor":   "A very powerful car",
        "low":      "A 100 bhp car",
        "high":     "A 1000 bhp car",
        "expected": "HIGH",
    },
    # --- Temperature ---
    {
        "category": "Temperature – hot",
        "anchor":   "An extremely hot day",
        "low":      "A 15°C day",
        "high":     "A 45°C day",
        "expected": "HIGH",
    },
    {
        "category": "Temperature – cold",
        "anchor":   "A freezing cold day",
        "low":      "A -30°C day",
        "high":     "A 5°C day",
        "expected": "LOW",
    },
    # --- Running speed ---
    {
        "category": "Running speed – fast",
        "anchor":   "A sprinter running at full speed",
        "low":      "A runner doing 5 km/h",
        "high":     "A runner doing 35 km/h",
        "expected": "HIGH",
    },
    # --- Building height ---
    {
        "category": "Building height – tall",
        "anchor":   "An extremely tall skyscraper",
        "low":      "A 10-metre building",
        "high":     "A 500-metre building",
        "expected": "HIGH",
    },
    # --- Weight ---
    {
        "category": "Object weight – heavy",
        "anchor":   "An extremely heavy truck",
        "low":      "A 2-tonne truck",
        "high":     "A 40-tonne truck",
        "expected": "HIGH",
    },
    # --- Price ---
    {
        "category": "Price – expensive",
        "anchor":   "A very expensive luxury car",
        "low":      "A $15,000 car",
        "high":     "A $500,000 car",
        "expected": "HIGH",
    },
    {
        "category": "Price – cheap",
        "anchor":   "A dirt-cheap budget phone",
        "low":      "A $50 phone",
        "high":     "A $1,500 phone",
        "expected": "LOW",
    },
    # --- Age / duration ---
    {
        "category": "Age – ancient",
        "anchor":   "An ancient historical monument",
        "low":      "A 20-year-old monument",
        "high":     "A 2,000-year-old monument",
        "expected": "HIGH",
    },
    # --- Crowd / quantity ---
    {
        "category": "Crowd size – huge",
        "anchor":   "An enormous crowd at the concert",
        "low":      "A crowd of 50 people",
        "high":     "A crowd of 50,000 people",
        "expected": "HIGH",
    },
    # --- Financial returns ---
    {
        "category": "Investment return – high",
        "anchor":   "An outstanding annual return on investment",
        "low":      "An annual return of 1%",
        "high":     "An annual return of 40%",
        "expected": "HIGH",
    },
    {
        "category": "Debt – severe",
        "anchor":   "A company drowning in debt",
        "low":      "A company with $10,000 in debt",
        "high":     "A company with $50,000,000 in debt",
        "expected": "HIGH",
    },
    # --- Brightness ---
    {
        "category": "Light – blinding",
        "anchor":   "A blinding, intensely bright light",
        "low":      "A 10-lumen torch",
        "high":     "A 100,000-lumen floodlight",
        "expected": "HIGH",
    },
    # --- Salary ---
    {
        "category": "Salary – very high",
        "anchor":   "A very well-paid executive salary",
        "low":      "A salary of $25,000 per year",
        "high":     "A salary of $2,000,000 per year",
        "expected": "HIGH",
    },
]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, rotary_scaling_factor=1)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    return tokenizer, model, device


def encode(texts: list[str], tokenizer, model, device: str) -> torch.Tensor:
    """Return L2-normalised sentence embeddings (batch_size, hidden_dim)."""
    # nomic-embed-text-v1.5 recommends task prefixes; "clustering" suits similarity comparisons
    prefixed = [f"clustering: {t}" for t in texts]
    encoded = tokenizer(
        prefixed,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    # Mean pool over non-padding tokens
    attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
    token_embeddings = output.last_hidden_state
    sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
    count = attention_mask.sum(dim=1).clamp(min=1e-9)
    embeddings = sum_embeddings / count

    return F.normalize(embeddings, p=2, dim=-1)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a * b).sum().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests():
    tokenizer, model, device = load_model(MODEL_NAME)

    correct = 0
    total = len(TEST_GROUPS)

    print("=" * 70)
    print(f"NUMERACY TEST  –  {MODEL_NAME}")
    print("=" * 70)

    for g in TEST_GROUPS:
        texts = [g["anchor"], g["low"], g["high"]]
        embs = encode(texts, tokenizer, model, device)
        anchor_emb, low_emb, high_emb = embs[0], embs[1], embs[2]

        sim_low  = cosine_sim(anchor_emb, low_emb)
        sim_high = cosine_sim(anchor_emb, high_emb)

        winner   = "HIGH" if sim_high > sim_low else "LOW"
        is_right = winner == g["expected"]
        correct += int(is_right)
        mark     = "✓" if is_right else "✗"

        print(f"\n[{mark}] {g['category']}")
        print(f"    Anchor : {g['anchor']}")
        print(f"    Low    : {g['low']:<45}  sim={sim_low:.4f}")
        print(f"    High   : {g['high']:<45}  sim={sim_high:.4f}")
        print(f"    Winner : {winner}  (expected {g['expected']})")

    print("\n" + "=" * 70)
    print(f"RESULT: {correct}/{total} correct  ({100 * correct / total:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
