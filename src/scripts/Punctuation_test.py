from transformers import AutoTokenizer
import re
import numpy as np

# Load Longformer Tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def compare_punctuation_cost(text):
    # 1. Original
    tokens_orig = count_tokens(text)
    
    # 2. No Punctuation (Aggressive cleaning)
    # Remove everything except words and spaces
    text_no_punct = re.sub(r'[^\w\s]', '', text) 
    tokens_clean = count_tokens(text_no_punct)
    
    diff = tokens_orig - tokens_clean
    pct_increase = (diff / tokens_clean) * 100 if tokens_clean > 0 else 0
    
    return tokens_orig, tokens_clean, pct_increase

# Test on a "Fake" Legal Sentence
sample_text = """
Article 6, section 1, of the Convention requires that: everyone is entitled to a fair hearing. 
The court noted (in paragraph 42) that the delay was unreasonable; therefore, a violation was found.
"""

print("\n--- SAMPLE TEST ---")
orig, clean, pct = compare_punctuation_cost(sample_text)
print(f"Original Text: {sample_text.strip()}")
print(f"Clean Text:    {re.sub(r'[^\w\s]', '', sample_text).strip()}")
print(f"\nTokens (With Punctuation):    {orig}")
print(f"Tokens (Without Punctuation): {clean}")
print(f"Cost of Punctuation:          +{pct:.2f}%")

# Decision Logic
print("\n--- RECOMMENDATION ---")
if pct > 20:
    print("WARNING: Punctuation is taking up massive space. We might need a hybrid approach.")
else:
    print(f"Punctuation costs {pct:.1f}%. KEEP IT. The accuracy drop from removing it is worse than the token savings.")