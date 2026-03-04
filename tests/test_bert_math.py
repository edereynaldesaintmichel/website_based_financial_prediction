import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def test_math_ability():
    # Load ModernBERT
    model_name = "answerdotai/ModernBERT-base"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    mask = tokenizer.mask_token
    
    def get_mask_tokens(answer: str) -> str:
        """Generate enough mask tokens to cover the tokenized answer."""
        num_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
        return " ".join([mask] * num_tokens)
    
    # Test cases: (template, expected_answer) - {MASK} will be replaced with appropriate masks
    test_cases = [
        # === SYMBOLIC FORMAT ===
        # 1-digit results
        ("2 + 3 = {MASK}", "5"),
        ("9 - 4 = {MASK}", "5"),
        ("3 * 2 = {MASK}", "6"),
        ("8 / 2 = {MASK}", "4"),
        
        # 2-digit results
        ("7 + 8 = {MASK}", "15"),
        ("50 - 25 = {MASK}", "25"),
        ("6 * 7 = {MASK}", "42"),
        ("12 + 34 = {MASK}", "46"),
        
        # 3-digit results
        ("100 + 200 = {MASK}", "300"),
        ("500 - 123 = {MASK}", "377"),
        ("250 + 250 = {MASK}", "500"),
        ("150 * 2 = {MASK}", "300"),
        
        # === NATURAL LANGUAGE FORMAT ===
        # Simple phrasing
        ("The result of 2 plus 3 is {MASK}.", "5"),
        ("The result of 9 minus 4 is {MASK}.", "5"),
        ("The result of 3 times 2 is {MASK}.", "6"),
        ("The result of 8 divided by 2 is {MASK}.", "4"),
        
        # 2-digit answers
        ("The result of 7 plus 8 is {MASK}.", "15"),
        ("The result of 50 minus 25 is {MASK}.", "25"),
        ("The result of 6 times 7 is {MASK}.", "42"),
        
        # 3-digit answers
        ("The result of 100 plus 200 is {MASK}.", "300"),
        ("The result of 500 minus 123 is {MASK}.", "377"),
        ("The result of 150 times 2 is {MASK}.", "300"),
        
        # Alternative natural language phrasings
        ("If you add 5 and 7, you get {MASK}.", "12"),
        ("What is 25 plus 25? The answer is {MASK}.", "50"),
        ("Calculate 100 minus 37. The answer is {MASK}.", "63"),
        ("The sum of 45 and 55 is {MASK}.", "100"),
        ("The difference between 200 and 75 is {MASK}.", "125"),
        ("The product of 12 and 8 is {MASK}.", "96"),
        ("When you multiply 15 by 4, you get {MASK}.", "60"),
        ("Dividing 144 by 12 gives {MASK}.", "12"),
        
        # Word problems
        ("I have 15 apples and eat 7. I have {MASK} apples left.", "8"),
        ("A book costs 25 dollars. Two books cost {MASK} dollars.", "50"),
        ("There are 100 students. If 35 leave, {MASK} remain.", "65"),
    ]
    
    print(f"\nTesting {len(test_cases)} math problems\n" + "=" * 60)
    
    correct = 0
    results_by_category = {"symbolic": [], "natural": [], "word_problem": []}
    
    for template, expected in test_cases:
        # Create proper mask tokens for this answer
        masks = get_mask_tokens(expected)
        problem = template.replace("{MASK}", masks)
        
        # Tokenize
        inputs = tokenizer(problem, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Find all mask positions
        mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Predict each mask token
        predicted_tokens = []
        for pos in mask_positions:
            top_id = logits[0, pos, :].argmax().item()
            predicted_tokens.append(top_id)
        
        # Decode prediction
        prediction = tokenizer.decode(predicted_tokens).strip()
        
        # Get top 5 alternatives for first mask (for insight)
        top5_ids = logits[0, mask_positions[0], :].topk(5).indices.tolist()
        top5_first = [tokenizer.decode([t]).strip() for t in top5_ids]
        
        is_correct = prediction == expected
        correct += int(is_correct)
        
        # Categorize
        if "=" in template:
            category = "symbolic"
        elif "apples" in template or "book" in template or "students" in template:
            category = "word_problem"
        else:
            category = "natural"
        results_by_category[category].append(is_correct)
        
        # Display
        symbol = "✓" if is_correct else "✗"
        display = template.replace("{MASK}", "___")
        print(f"{symbol} {display}")
        print(f"   Expected: '{expected}' | Predicted: '{prediction}' | Masks used: {len(mask_positions)}")
        print(f"   Top-5 for first mask: {top5_first}\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)\n")
    
    for category, results in results_by_category.items():
        if results:
            cat_correct = sum(results)
            cat_total = len(results)
            print(f"  {category.title():15} : {cat_correct}/{cat_total} ({100*cat_correct/cat_total:.1f}%)")
    
    print("\n" + "=" * 60)
    print("NOTES:")
    print("- Multi-digit answers use multiple [MASK] tokens")
    print("- Tokenization affects how numbers are split")
    print("- MLM models aren't designed for computation")
    print("=" * 60)


def analyze_tokenization(model_name="answerdotai/ModernBERT-base"):
    """Helper to see how numbers are tokenized."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("\n" + "=" * 60)
    print("NUMBER TOKENIZATION ANALYSIS")
    print("=" * 60)
    
    test_numbers = ["5", "12", "42", "100", "125", "300", "377", "500", "1000"]
    
    for num in test_numbers:
        tokens = tokenizer.tokenize(num)
        ids = tokenizer.encode(num, add_special_tokens=False)
        print(f"  '{num}' -> {tokens} (token IDs: {ids})")


if __name__ == "__main__":
    test_math_ability()
    analyze_tokenization()