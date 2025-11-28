import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import warnings

# Set style for professional academic plots
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')

def load_data():
    """Load ECTHR_B dataset"""
    print("[v0] Loading dataset (lex_glue/ecthr_b)...")
    try:
        dataset = load_dataset("lex_glue", "ecthr_b")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_approx_tokens(text_list):
    """
    Estimate token count. 
    Rule of thumb: 1 word ≈ 1.3 tokens for BERT/RoBERTa/Longformer subword tokenizers.
    """
    if isinstance(text_list, list):
        text = " ".join(str(x) for x in text_list)
    else:
        text = str(text_list)
    
    words = text.split()
    return len(words) * 1.3

def analyze_longformer_readiness(dataset):
    """
    Advanced analysis specifically for Longformer training.
    Checks how many documents exceed the 4096 token limit.
    """
    print("\n" + "="*80)
    print("LONGFORMER READINESS ANALYSIS")
    print("="*80)
    
    limit = 4096
    all_lengths = []
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            # Calculate approx tokens for every doc
            lengths = [get_approx_tokens(x['text']) for x in dataset[split]]
            all_lengths.extend(lengths)
    
    all_lengths = np.array(all_lengths)
    percent_below = np.sum(all_lengths <= limit) / len(all_lengths) * 100
    
    print(f"Total Samples Analyzed: {len(all_lengths)}")
    print(f"Longformer Limit: {limit} tokens")
    print(f"Documents fully fitting in context: {percent_below:.2f}%")
    print(f"Documents requiring truncation: {100 - percent_below:.2f}%")
    print(f"Max token length found: {int(np.max(all_lengths))}")
    
    return all_lengths

def visualize_advanced_stats(dataset):
    """
    Generates advanced visualizations:
    1. Label Frequency (Corrected)
    2. Label Co-occurrence Heatmap (New - Crucial for Multi-label)
    3. Token Length Distribution (New - For Longformer)
    4. Labels per Document
    """
    print("\n[v0] Generating advanced visualizations...")
    
    # --- Data Aggregation ---
    all_labels = []
    text_lengths = []
    
    for split in ['train', 'validation', 'test']:
        for item in dataset[split]:
            all_labels.append(item['labels'])
            text_lengths.append(get_approx_tokens(item['text']))

    # Flatten labels for counting
    flat_labels = [l for sublist in all_labels for l in sublist]
    label_counts = Counter(flat_labels)
    unique_labels = sorted(label_counts.keys())
    
    # --- Figure Setup ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3) # Grid layout
    
    # --- Plot 1: Label Distribution (Corrected) ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Dynamic top N (limits to 10 if only 10 exist)
    n_display = min(len(label_counts), 20)
    common_labels = label_counts.most_common(n_display)
    
    labels_x = [f"Art. {lbl}" for lbl, count in common_labels]
    counts_y = [count for lbl, count in common_labels]
    
    sns.barplot(x=counts_y, y=labels_x, ax=ax1, palette="viridis")
    ax1.set_title(f'Frequency of Labels')
    ax1.set_xlabel('Occurrences')

    # --- Plot 2: Label Co-occurrence Heatmap (Advanced) ---
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Create Co-occurrence Matrix
    co_matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)
    for labels in all_labels:
        if len(labels) > 1:
            for l1, l2 in combinations(sorted(labels), 2):
                co_matrix.loc[l1, l2] += 1
                co_matrix.loc[l2, l1] += 1 # Symmetric
    
    # Normalize matrix (optional, helps visibility) - Log scale + 1 to handle zeros
    mask = np.triu(np.ones_like(co_matrix, dtype=bool))
    sns.heatmap(co_matrix, annot=True, fmt='d', cmap="Reds", mask=mask, ax=ax2,
                xticklabels=[f"Art {i}" for i in unique_labels],
                yticklabels=[f"Art {i}" for i in unique_labels])
    ax2.set_title('Label Co-occurrence Matrix\n(Which articles appear together?)')

    # --- Plot 3: Token Length Distribution (Longformer Context) ---
    ax3 = fig.add_subplot(gs[1, 0:2])
    sns.histplot(text_lengths, kde=True, color="teal", ax=ax3, bins=50)
    ax3.axvline(x=4096, color='r', linestyle='--', linewidth=2, label='Longformer Limit (4096)')
    ax3.set_title('Estimated Token Length Distribution')
    ax3.set_xlabel('Estimated Tokens (Words * 1.3)')
    ax3.legend()

    # --- Plot 4: Labels per Sample ---
    ax4 = fig.add_subplot(gs[1, 2])
    labels_per_doc = [len(l) for l in all_labels]
    sns.countplot(x=labels_per_doc, ax=ax4, palette="magma")
    ax4.set_title('Count of Labels per Document')
    ax4.set_xlabel('Number of Labels')
    ax4.set_ylabel('Number of Documents')

    plt.tight_layout()
    plt.savefig('advanced_eda_longformer.png', dpi=300)
    print("✓ Saved: advanced_eda_longformer.png")

if __name__ == "__main__":
    dataset = load_data()
    
    if dataset:
        # 1. Run Text Stats (reusing your logical flow)
        analyze_longformer_readiness(dataset)
        
        # 2. Run Visualization
        visualize_advanced_stats(dataset)
        
        print("\n" + "="*80)
        print("NEXT STEPS FOR LONGFORMER:")
        print("="*80)
        print("1. Weighted Loss: Use Class weights from Plot 1 to handle imbalance.")
        print("2. Truncation Strategy: If Plot 3 shows heavy tails > 4096, consider 'global attention' on the CLS token.")
        print("3. Co-occurrence: High overlap in Plot 2 suggests the model might benefit from learning label correlations.")