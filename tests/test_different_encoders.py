import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_embedding_quality():
    """Compare semantic similarity detection."""
    
    # Test pairs: (sentence1, sentence2, should_be_similar)
    test_pairs = [
        # Similar meaning
        ("The cat sat on the mat.", "A feline rested on the rug.", True),
        ("I love programming in Python.", "Python coding is my passion.", True),
        ("The weather is beautiful today.", "It's a gorgeous day outside.", True),
        ("He drove his car to work.", "He commuted by automobile.", True),
        ("The movie was boring.", "The film was dull and uninteresting.", True),
        
        # Different meaning
        ("The cat sat on the mat.", "Dogs are loyal animals.", False),
        ("I love programming in Python.", "The snake slithered away.", False),
        ("The bank was steep.", "I deposited money at the bank.", False),
        ("She played the bass guitar.", "We caught bass at the lake.", False),
        ("The movie was boring.", "I love exciting action films.", False),
    ]
    
    models_to_test = [
        ("princeton-nlp/sup-simcse-bert-base-uncased", "encoder", "SimCSE-BERT"),
        ("sentence-transformers/all-MiniLM-L6-v2", "encoder", "MiniLM"),
        ("google/gemma-3-270m-it", "decoder", "Gemma-270M-IT"),
    ]
    
    results_summary = []
    
    for model_name, model_type, short_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"{short_name}")
        print('='*60)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Failed to load: {e}\n")
            continue
        
        def get_embedding(text):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            hidden = outputs.last_hidden_state
            
            if model_type == "encoder":
                # Use [CLS] token
                embedding = hidden[:, 0, :]
            else:
                # Mean pooling for decoder
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                embedding = summed / mask.sum(dim=1)
            
            return embedding.float().cpu().detach().numpy()
        
        similarities_similar = []
        similarities_different = []
        
        for s1, s2, should_match in test_pairs:
            emb1 = get_embedding(s1)
            emb2 = get_embedding(s2)
            sim = cosine_similarity(emb1, emb2)[0, 0]
            
            if should_match:
                similarities_similar.append(sim)
            else:
                similarities_different.append(sim)
        
        # Calculate metrics
        avg_similar = np.mean(similarities_similar)
        avg_different = np.mean(similarities_different)
        separation = avg_similar - avg_different
        
        # Calculate accuracy at threshold 0.5
        all_sims = similarities_similar + similarities_different
        all_labels = [True] * len(similarities_similar) + [False] * len(similarities_different)
        preds = [s > 0.5 for s in all_sims]
        accuracy = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
        
        print(f"Similar pairs avg:      {avg_similar:.3f}")
        print(f"Different pairs avg:    {avg_different:.3f}")
        print(f"Separation:             {separation:.3f}")
        print(f"Accuracy (threshold=0.5): {accuracy*100:.1f}%\n")
        
        results_summary.append({
            "name": short_name,
            "separation": separation,
            "accuracy": accuracy,
            "avg_similar": avg_similar,
            "avg_different": avg_different,
        })
    
    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Separation':>12} {'Accuracy':>12}")
    print("-" * 60)
    
    for result in sorted(results_summary, key=lambda x: -x["separation"]):
        print(f"{result['name']:<20} {result['separation']:>12.3f} {result['accuracy']*100:>11.1f}%")
    
    print("\nHigher separation = better at distinguishing similar from different pairs")


if __name__ == "__main__":
    compare_embedding_quality()