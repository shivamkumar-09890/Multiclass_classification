import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.cluster import KMeans
import warnings

# Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

def load_data():
    """Load ECTHR_B dataset"""
    print("[v0] Loading ECTHR_B dataset...")
    return load_dataset("lex_glue", "ecthr_b")

def clean_text_minimal(text_input):
    """Minimal cleaning for Transformers/Longformer."""
    if text_input is None: return ""
    if isinstance(text_input, list):
        text = " ".join(str(x) for x in text_input)
    else:
        text = str(text_input)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_duplicates(dataset_split):
    """Removes duplicate entries based on exact text match."""
    df = pd.DataFrame(dataset_split)
    initial_len = len(df)
    
    df = df.drop_duplicates(subset=['text'])
    
    removed = initial_len - len(df)
    if removed > 0:
        print(f"   > Removed {removed} duplicate entries.")
    
    # Reset index to fix the "index_level_0" bug
    df = df.reset_index(drop=True)
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])
        
    return Dataset.from_pandas(df, preserve_index=False)

def detect_and_remove_outliers(dataset_split, split_name="train"):
    """
    1. Clusters documents by length.
    2. VISUALIZES the clusters.
    3. Removes the 'Extreme' cluster if safe.
    """
    print(f"\n[Analysis] Clustering length values for '{split_name}'...")
    
    # 1. Calculate Lengths
    df = pd.DataFrame(dataset_split)
    df['length'] = df['text'].apply(lambda x: len(x.split()))
    
    # 2. Run K-Means
    X = df[['length']].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # 3. Identify Clusters
    cluster_means = df.groupby('cluster')['length'].mean()
    extreme_cluster_id = cluster_means.idxmax()
    extreme_cluster_mean = cluster_means.max()
    
    # Sort clusters by mean length for better plotting labels (Short -> Long)
    sorted_map = {old: new for new, old in enumerate(cluster_means.sort_values().index)}
    df['cluster_label'] = df['cluster'].map(sorted_map)
    df['cluster_name'] = df['cluster_label'].map({0: 'Short', 1: 'Medium', 2: 'Long', 3: 'Extreme'})

    # --- 4. VISUALIZATION BLOCK (NEW) ---
    print(f"   > Generating visualization: outliers_{split_name}.png")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot A: Scatter (Index vs Length)
    # This shows the "buckets" clearly
    sns.scatterplot(
        data=df, 
        x=df.index, 
        y='length', 
        hue='cluster_name', 
        palette='viridis', 
        ax=axes[0],
        alpha=0.6,
        s=15
    )
    axes[0].set_title(f'Document Length Clusters ({split_name})')
    axes[0].set_ylabel('Word Count')
    axes[0].set_xlabel('Document Index')
    
    # Plot B: Boxplot (Distribution per Cluster)
    sns.boxplot(
        data=df,
        x='cluster_name',
        y='length',
        palette='viridis',
        ax=axes[1],
        order=['Short', 'Medium', 'Long', 'Extreme']
    )
    axes[1].set_title('Length Distribution per Cluster')
    
    plt.tight_layout()
    plt.savefig(f'outliers_{split_name}.png', dpi=150)
    # ------------------------------------

    # 5. Logic to Remove
    n_extreme = len(df[df['cluster'] == extreme_cluster_id])
    pct_extreme = (n_extreme / len(df)) * 100
    
    print(f"   > Clusters Found (Avg Lengths): {sorted(cluster_means.values.astype(int))}")
    print(f"   > Potential Outliers (Extreme): {n_extreme} docs ({pct_extreme:.2f}%)")
    
    df_clean = df
    
    # Threshold: Must be < 5% of data AND Avg length > 10k words
    if pct_extreme < 5.0 and extreme_cluster_mean > 10000:
        print(f"   > Action: REMOVING {n_extreme} extreme outliers.")
        # Filter out the extreme cluster
        df_clean = df[df['cluster'] != extreme_cluster_id]
    else:
        print(f"   > Action: KEEPING all data (Outliers not distinct/rare enough).")
    
    # 6. Cleanup DataFrame before converting back to Dataset
    # Drop calculation columns
    cols_to_drop = ['length', 'cluster', 'cluster_label', 'cluster_name', '__index_level_0__']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    # Reset index to ensure schema consistency
    df_clean = df_clean.reset_index(drop=True)

    return Dataset.from_pandas(df_clean, preserve_index=False)

def preprocess_pipeline():
    raw_dataset = load_data()
    processed_splits = {}
    
    print("\n" + "="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)

    for split in ['train', 'validation', 'test']:
        if split in raw_dataset:
            print(f"\nProcessing Split: {split.upper()}")
            print(f"   > Original Count: {len(raw_dataset[split])}")
            
            # A. Clean
            current_ds = raw_dataset[split].map(
                lambda x: {'text': clean_text_minimal(x['text'])},
                desc="Cleaning text"
            )
            
            # B. Dedup
            current_ds = remove_duplicates(current_ds)
            
            # C. Outliers (Train only)
            if split == 'train':
                current_ds = detect_and_remove_outliers(current_ds, split)
            
            print(f"   > Final Count:    {len(current_ds)}")
            processed_splits[split] = current_ds

    final_dataset = DatasetDict(processed_splits)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    return final_dataset

if __name__ == "__main__":
    clean_dataset = preprocess_pipeline()
    clean_dataset.save_to_disk("processed_ecthr_b")