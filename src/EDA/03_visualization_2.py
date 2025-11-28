import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import warnings

# Professional plotting style
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')

# Constants
MODEL_CKPT = "allenai/longformer-base-4096"

def load_data_and_tokenizer():
    """Load ECTHR_B dataset and Longformer Tokenizer"""
    print(f"[v0] Loading Tokenizer ({MODEL_CKPT})...")
    # use_fast=True is much faster for counting
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)
    
    print("[v0] Loading dataset (lex_glue/ecthr_b)...")
    try:
        dataset = load_dataset("lex_glue", "ecthr_b")
        return dataset, tokenizer
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_text_stats(text_input, tokenizer):
    """
    Returns both word count (space split) and ACTUAL token count (Longformer).
    """
    # Normalize text (handle list of strings vs single string)
    text = " ".join(str(x) for x in text_input) if isinstance(text_input, list) else str(text_input)
    
    # 1. Word Count (for Row 1 Graphs)
    word_count = len(text.split())
    
    # 2. Actual Token Count (for Row 2 Graph)
    # add_special_tokens=True ensures we count [CLS] and [SEP] which take up space
    token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    token_count = len(token_ids)
    
    return word_count, token_count

def visualize_comprehensive_analysis(dataset, tokenizer):
    print("\n[v0] Generating comprehensive visualizations (this takes ~30s for tokenization)...")
    
    # --- 1. Data Preparation ---
    data_map = {
        'train': {'words': [], 'tokens': []}, 
        'validation': {'words': [], 'tokens': []}, 
        'test': {'words': [], 'tokens': []}
    }
    all_tokens = []
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            print(f"Processing {split} split...")
            for item in dataset[split]:
                # Pass tokenizer to the stats function
                w_count, t_count = get_text_stats(item['text'], tokenizer)
                
                # Store specific split data
                data_map[split]['words'].append(w_count)
                data_map[split]['tokens'].append(t_count)
                
                # Store aggregate for coverage report
                all_tokens.append(t_count)

    # --- 2. Calculate Coverage Stats (Based on ACTUAL TOKENS) ---
    all_tokens_np = np.array(all_tokens)
    limit = 4096
    covered_count = np.sum(all_tokens_np <= limit)
    total_count = len(all_tokens_np)
    coverage_pct = (covered_count / total_count) * 100
    
    print("\n" + "="*60)
    print(f"LONGFORMER COVERAGE REPORT (Limit: {limit} tokens)")
    print("="*60)
    print(f"Total Documents: {total_count}")
    print(f"Documents within limit: {covered_count}")
    print(f"Coverage Percentage:    {coverage_pct:.2f}%")
    print("="*60)

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3)

    # ---------------------------------------------------------
    # Row 1: Split-wise Distributions (Word Count) - INTACT
    # ---------------------------------------------------------
    splits = ['train', 'validation', 'test']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    
    for i, split in enumerate(splits):
        ax = fig.add_subplot(gs[0, i])
        
        # Get Word Counts for this split
        words = data_map[split]['words']
        mean_words = np.mean(words)
        
        # Plot Histogram
        sns.histplot(words, kde=True, color=colors[i], bins=40, ax=ax)
        
        # Update Titles and Labels
        ax.set_title(f'{split.capitalize()} Set (Word Count)')
        ax.set_xlabel('Number of Words')
        
        # Add Vertical Line for MEAN
        ax.axvline(x=mean_words, color='red', linestyle='-', linewidth=2, label=f'Mean: {int(mean_words)}')
        ax.legend()

    # ---------------------------------------------------------
    # Row 2: Cumulative Coverage Analysis (Actual Tokens)
    # ---------------------------------------------------------
    ax_cov = fig.add_subplot(gs[1, :]) # Spans full width
    
    # Sort token data for CDF
    sorted_data = np.sort(all_tokens_np)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data)) * 100
    
    ax_cov.plot(sorted_data, yvals, color='purple', linewidth=3)
    ax_cov.fill_between(sorted_data, yvals, alpha=0.1, color='purple')
    
    # Add the 4096 Marker
    ax_cov.axvline(x=limit, color='red', linestyle='--', linewidth=2)
    ax_cov.axhline(y=coverage_pct, color='red', linestyle='--', linewidth=2)
    
    # Annotation text
    ax_cov.text(limit + 500, coverage_pct - 5, 
                f'  {coverage_pct:.1f}% of data\n  fits in 4096 tokens', 
                color='red', fontweight='bold', fontsize=12)
    
    ax_cov.set_title('Model Readiness: Actual Token Coverage Analysis', fontsize=14)
    ax_cov.set_xlabel('Token Length (Actual Longformer Tokens)') # Label updated
    ax_cov.set_ylabel('Percentage of Dataset Captured (%)')
    ax_cov.set_xlim(0, max(all_tokens_np) + 500)
    ax_cov.set_yticks(np.arange(0, 101, 10))
    ax_cov.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_analysis_v2.png', dpi=300)
    print("✓ Saved: comprehensive_analysis_v2.png")

if __name__ == "__main__":
    # Load dataset AND tokenizer
    dataset, tokenizer = load_data_and_tokenizer()
    
    if dataset and tokenizer:
        visualize_comprehensive_analysis(dataset, tokenizer)