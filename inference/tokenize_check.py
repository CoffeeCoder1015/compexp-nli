"""
Tokenization Verification Script

This script is part of the prep work to verify how NLI words are tokenized
by the decoder model before implementing the full analysis.

Run this first to determine if "entailment", "neutral", "contradiction"
are single tokens or multiple tokens.

Usage:
    python data/tokenize_check.py
"""

import sys
sys.path.insert(0, '.')

from lora_loader import get_model_with_lora


MODEL_ID = "LiquidAI/LFM2.5-1.2B-Base"
LORA_PATH = "../finetune/model/checkpoint-1000"


def check_tokenization():
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_with_lora(MODEL_ID, lora_path=LORA_PATH)
    
    print("\n" + "="*60)
    print("TOKENIZATION CHECK")
    print("="*60)
    
    nli_words = ["entailment", "neutral", "contradiction"]
    
    for word in nli_words:
        ids = tokenizer.encode(word)
        print(f"\nWord: '{word}'")
        print(f"  Token IDs: {ids}")
        print(f"  Token count: {len(ids)}")
        
        if len(ids) == 1:
            print(f"  ✅ SINGLE TOKEN")
            token = tokenizer.decode(ids[0])
            print(f"  Token text: '{token}'")
        else:
            print(f"  ⚠️ MULTIPLE TOKENS")
            tokens = [tokenizer.decode([i]) for i in ids]
            print(f"  Token texts: {tokens}")
    
    print("\n" + "="*60)
    print("CHAT TEMPLATE TEST")
    print("="*60)
    
    system_prompt = {
        "role": "system",
        "content": """Determine the relationship between the `Premise` and `Hypothesis` and respond with an answer. 
You must respond with an answer of `Entailment`, `Neutral` or `Contradiction`
You need to respond in the format shown in the following by chosing one of thosw answers:
<my_answer>[place your answer here]</my_answer>
You are lay out the steps to your final answer before responding with your final answer, but you must respond in this format or else your answer will be rejected."""
    }
    
    test_example = {
        "role": "user",
        "content": "Premise: The cat sleeps. Hypothesis: An animal is resting."
    }
    
    prompt = [system_prompt, test_example]
    
    print("\nPrompt structure:")
    for msg in prompt:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    formatted = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    print(f"\nFormatted text (first 500 chars):\n{formatted[:500]}...")
    
    tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    print(f"\nTotal token count: {len(tokens)}")
    print(f"Last 15 tokens: {tokens[-15:]}")
    print(f"Last 15 decoded: {[tokenizer.decode([t]) for t in tokens[-15:]]}")
    
    print("\n" + "="*60)
    print("VOCABULARY PREFIX CHECK (if multi-token)")
    print("="*60)
    
    vocab = tokenizer.get_vocab()
    
    for prefix in ["ent", "neu", "con"]:
        matching = [(k, v) for k, v in vocab.items() if k.startswith(prefix)]
        print(f"\nPrefix '{prefix}': {len(matching)} tokens")
        if matching:
            sorted_matches = sorted(matching, key=lambda x: x[1])[:5]
            print(f"  First 5: {sorted_matches}")


if __name__ == "__main__":
    check_tokenization()
