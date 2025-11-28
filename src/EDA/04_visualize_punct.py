import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import re
import warnings

# Configuration
sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings('ignore')
LIMIT = 4096
MODEL_CKPT = "allenai/longformer-base-4096"

def load_resources():
    print(f"[v0] Loading Tokenizer ({MODEL_CKPT})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)
    print("[v0] Loading dataset...")
    try:
        dataset = load_dataset("lex_glue", "ecthr_b")
        return dataset, tokenizer
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def strip_punctuation(text):
    """
    Replaces all punctuation with spaces to prevent word merging.
    'state-of-the-art' -> 'state of the art' (keeps token count logic consistent)
    """
    # Replace non-word/non-space characters with a space
    return re.sub(r'[^\w\s]', ' ', text)

def compare_token_counts(dataset, tokenizer):
    print("\n[v0] Running Comparative Analysis (This runs 2x Tokenization)...")
    
    results = []
    
    # We combine all splits for a full dataset analysis
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            print(f"   > Processing {split} set...")
            
            raw_texts = dataset[split]['text']
            
            # 1. Prepare strings
            texts_with_punct = [
                " ".join(str(x) for x in t) if isinstance(t, list) else str(t) 
                for t in raw_texts
            ]
            
            # 2. Prepare stripped strings
            texts_no_punct = [strip_punctuation(t) for t in texts_with_punct]
            
            # 3. Batch Tokenize - PASS 1 (With Punctuation)
            enc_with = tokenizer(texts_with_punct, add_special_tokens=True, truncation=False, verbose=False)
            lens_with = [len(x) for x in enc_with['input_ids']]
            
            # 4. Batch Tokenize - PASS 2 (No Punctuation)
            enc_no = tokenizer(texts_no_punct, add_special_tokens=True, truncation=False, verbose=False)
            lens_no = [len(x) for x in enc_no['input_ids']]
            
            # Store data
            for i in range(len(lens_with)):
                results.append({
                    'with_punct': lens_with[i],
                    'no_punct': lens_no[i]
                })

    df = pd.DataFrame(results)
    
    # --- CALCULATE THE "TAX" ---
    df['diff'] = df['with_punct'] - df['no_punct']
    df['pct_increase'] = (df['diff'] / df['no_punct']) * 100
    
    # Critical Metric: "The Lost Files"
    # Files that fit without punctuation (<= 4096) BUT fail with punctuation (> 4096)
    lost_files = df[(df['no_punct'] <= LIMIT) & (df['with_punct'] > LIMIT)]
    num_lost = len(lost_files)
    
    avg_tax = df['pct_increase'].mean()
    
    print("\n" + "="*60)
    print("PUNCTUATION COST REPORT")
    print("="*60)
    print(f"Total Documents:           {len(df)}")
    print(f"Average 'Punctuation Tax': +{avg_tax:.2f}% tokens per doc")
    print("-" * 60)
    print(f"Docs fitting limit (NO Punct):   {len(df[df['no_punct'] <= LIMIT])}")
    print(f"Docs fitting limit (WITH Punct): {len(df[df['with_punct'] <= LIMIT])}")
    print("-" * 60)
    print(f"DOCUMENTS 'LOST' TO PUNCTUATION: {num_lost}")
    print(f"(These {num_lost} docs would have fit perfectly, but now get truncated)")
    print("="*60)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: The Cost Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['pct_increase'], bins=50, color='orange', kde=True, ax=ax1)
    ax1.axvline(x=avg_tax, color='red', linestyle='--', label=f'Avg Cost: +{avg_tax:.1f}%')
    ax1.set_title('Punctuation Tax: % Increase in Token Count')
    ax1.set_xlabel('% Increase')
    ax1.legend()
    
    # Plot 2: Scatter Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    # Scatter plot
    ax2.scatter(df['no_punct'], df['with_punct'], alpha=0.1, color='purple', s=10)
    # Reference line (y=x)
    max_val = 10000 # Zoom in a bit
    ax2.plot([0, max_val], [0, max_val], color='black', linestyle='--', label='No Cost Line')
    
    # The Danger Zone
    rect = plt.Rectangle((0, LIMIT), LIMIT, max_val-LIMIT, 
                         facecolor="red", alpha=0.1, label="Truncation Zone")
    ax2.add_patch(rect)
    
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.set_title('Token Count: With vs Without Punctuation')
    ax2.set_xlabel('Tokens (No Punct)')
    ax2.set_ylabel('Tokens (With Punct)')
    ax2.legend()
    
    # Plot 3: Cumulative Coverage Gap
    ax3 = fig.add_subplot(gs[1, :])
    
    sorted_with = np.sort(df['with_punct'])
    sorted_no = np.sort(df['no_punct'])
    yvals = np.arange(len(df)) / float(len(df)) * 100
    
    ax3.plot(sorted_no, yvals, color='green', linestyle='--', linewidth=2, label='Potential Coverage (No Punct)')
    ax3.plot(sorted_with, yvals, color='red', linewidth=2, label='Actual Coverage (With Punct)')
    
    # Fill the gap
    ax3.fill_betweenx(yvals, sorted_no, sorted_with, color='yellow', alpha=0.3, label='The "Cost" Gap')
    
    ax3.axvline(x=LIMIT, color='black', linestyle='-', linewidth=2)
    ax3.text(LIMIT+200, 10, 'Longformer Limit (4096)', rotation=90)
    
    ax3.set_title('The Coverage Gap: How much capacity are we losing?')
    ax3.set_xlabel('Token Count')
    ax3.set_ylabel('Cumulative % of Data')
    ax3.set_xlim(0, 10000)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('punctuation_cost_analysis.png', dpi=300)
    print("✓ Saved: punctuation_cost_analysis.png")

if __name__ == "__main__":
    dataset, tokenizer = load_resources()
    if dataset and tokenizer:
        compare_token_counts(dataset, tokenizer)