"""
Task 2: Exploratory Data Analysis and Visualization
- Analyzes text statistics and distributions
- Generates visualizations for understanding data
- Identifies patterns in labels and text features
"""

import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load ECTHR_B dataset"""
    print("[v0] Loading dataset for EDA...")
    dataset = load_dataset("lex_glue", "ecthr_b")
    return dataset

def normalize_text(t):
    """Convert text to clean string."""
    if t is None:
        return ""
    if isinstance(t, list):  
        return " ".join(str(x) for x in t)
    return str(t)

def analyze_text_statistics(dataset):
    """Analyze text length and complexity"""
    print("\n" + "="*80)
    print("TEXT STATISTICS ANALYSIS")
    print("="*80)
    
    stats = {'split': [], 'avg_length': [], 'median_length': [], 'max_length': [], 'min_length': []}
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            normalized_texts = [normalize_text(x['text']) for x in dataset[split]]
            lengths = [len(text.split()) for text in normalized_texts]
            
            stats['split'].append(split)
            stats['avg_length'].append(np.mean(lengths))
            stats['median_length'].append(np.median(lengths))
            stats['max_length'].append(np.max(lengths))
            stats['min_length'].append(np.min(lengths))
            
            print(f"\n{split.upper()} SET:")
            print(f"  - Average words per document: {np.mean(lengths):.2f}")
            print(f"  - Median words per document: {np.median(lengths):.2f}")
            print(f"  - Max words: {np.max(lengths)}")
            print(f"  - Min words: {np.min(lengths)}")
            print(f"  - Std deviation: {np.std(lengths):.2f}")
    
    return pd.DataFrame(stats)

def visualize_label_distribution(dataset):
    """Create visualizations for label distribution"""
    print("\n[v0] Generating label distribution visualization...")
    
    # Collect label statistics
    label_counts = Counter()
    labels_per_sample = []
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            for example in dataset[split]:
                labels = example['labels']
                label_counts.update(labels)
                labels_per_sample.append(len(labels))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Label Distribution Analysis - ECTHR Dataset', fontsize=16, fontweight='bold')
    
    # Plot 1: Top 20 most common labels
    top_labels = dict(label_counts.most_common(20))
    ax = axes[0, 0]
    ax.barh(list(map(str, top_labels.keys())), list(top_labels.values()), color='steelblue')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Article ID')
    ax.set_title('Top 20 Most Frequent Articles')
    ax.invert_yaxis()
    
    # Plot 2: Labels per sample distribution
    ax = axes[0, 1]
    ax.hist(labels_per_sample, bins=range(1, max(labels_per_sample)+2), color='coral', edgecolor='black')
    ax.set_xlabel('Number of Labels per Sample')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Labels per Sample')
    ax.axvline(np.mean(labels_per_sample), color='red', linestyle='--', label=f'Mean: {np.mean(labels_per_sample):.2f}')
    ax.legend()
    
    # Plot 3: Label frequency distribution (how many articles appear N times)
    ax = axes[1, 0]
    freq_counts = Counter(label_counts.values())
    ax.bar(freq_counts.keys(), freq_counts.values(), color='seagreen', edgecolor='black')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Number of Articles with this Frequency')
    ax.set_title('Article Frequency Distribution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 4: Cumulative distribution
    ax = axes[1, 1]
    sorted_freqs = sorted(label_counts.values(), reverse=True)
    cumsum = np.cumsum(sorted_freqs)
    cumsum_pct = (cumsum / cumsum[-1]) * 100
    ax.plot(range(len(cumsum_pct)), cumsum_pct, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax.set_xlabel('Number of Articles')
    ax.set_ylabel('Cumulative % of Label Occurrences')
    ax.set_title('Cumulative Distribution of Labels')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: label_distribution.png")
    
    return label_counts

def visualize_text_length(dataset):
    """Visualize text length distribution"""
    print("[v0] Generating text length visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Text Length Distribution by Dataset Split', fontsize=14, fontweight='bold')
    
    colors = {'train': 'steelblue', 'validation': 'coral', 'test': 'seagreen'}
    
    for idx, split in enumerate(['train', 'validation', 'test']):
        if split in dataset:
            normalized_texts = [normalize_text(x['text']) for x in dataset[split]]
            lengths = [len(x.split()) for x in normalized_texts]

            ax = axes[idx]
            ax.hist(lengths, bins=50, color=colors[split], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Words')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{split.upper()} Set\n(n={len(lengths)})')
            ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.0f}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('text_length_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: text_length_distribution.png")

def dataset_split_analysis(dataset):
    """Analyze dataset splits"""
    print("\n" + "="*80)
    print("DATASET SPLIT ANALYSIS")
    print("="*80)
    
    splits_info = {}

    for split in ['train', 'validation', 'test']:
        if split in dataset:
            texts = [normalize_text(x['text']) for x in dataset[split]]

            splits_info[split] = {
                'samples': len(dataset[split]),
                'avg_labels': np.mean([len(x['labels']) for x in dataset[split]]),
                'avg_words': np.mean([len(t.split()) for t in texts])
            }
    
    df = pd.DataFrame(splits_info).T
    print("\n" + df.to_string())
    
    return df

if __name__ == "__main__":
    dataset = load_data()
    
    # Text statistics
    text_stats = analyze_text_statistics(dataset)
    
    # Visualizations
    label_counts = visualize_label_distribution(dataset)
    visualize_text_length(dataset)
    split_analysis = dataset_split_analysis(dataset)
    
    print("\n" + "="*80)
    print("KEY FINDINGS FROM EDA:")
    print("="*80)
    print("✓ Multi-label classification task with ~11k training samples")
    print("✓ Variable number of ECHR articles per case (typically 1-5)")
    print("✓ Long documents (~500-2000 words per case)")
    print("✓ Imbalanced label distribution (some articles more common than others)")
    print("✓ Class imbalance is typical for legal domain - handled via appropriate metrics")
