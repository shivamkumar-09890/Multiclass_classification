import pandas as pd
import numpy as np
import re
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load ECTHR_B dataset"""
    print("[v0] Loading ECTHR_B dataset...")
    return load_dataset("lex_glue", "ecthr_b")

def clean_text_minimal(text_input):
    """
    Minimal cleaning for Transformers/Longformer.
    Crucial: KEEPS punctuation and case information.
    """
    if text_input is None:
        return ""
    
    # 1. Handle List vs String
    if isinstance(text_input, list):
        text = " ".join(str(x) for x in text_input)
    else:
        text = str(text_input)

    # 2. Normalize Whitespace (Tab/Newline -> Space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Remove artifacts but KEEP punctuation (.,;: etc are needed for attention)
    # Only remove obviously broken encoding artifacts if necessary
    return text

def remove_duplicates(dataset_split):
    """
    Removes duplicate entries based on exact text match.
    """
    df = pd.DataFrame(dataset_split)
    initial_len = len(df)
    
    # Drop duplicates based on the 'text' column
    df = df.drop_duplicates(subset=['text'])
    
    removed = initial_len - len(df)
    if removed > 0:
        print(f"   > Removed {removed} duplicate entries.")
    
    return Dataset.from_pandas(df)

def detect_and_remove_outliers(dataset_split, split_name="train"):
    """
    Uses K-Means Clustering on document lengths to find and remove 
    extreme outliers (documents that are suspiciously long).
    """
    print(f"\n[Analysis] clustering length values for '{split_name}'...")
    
    # 1. Calculate Lengths (Word Count)
    df = pd.DataFrame(dataset_split)
    df['length'] = df['text'].apply(lambda x: len(x.split()))
    
    # 2. Prepare for Clustering (Reshape for sklearn)
    X = df[['length']].values
    
    # 3. Run K-Means (k=4: Short, Medium, Long, Extreme)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # 4. Identify the "Extreme" cluster (The one with the highest average length)
    cluster_means = df.groupby('cluster')['length'].mean()
    extreme_cluster_id = cluster_means.idxmax()
    extreme_cluster_mean = cluster_means.max()
    
    # 5. Safety Check: Don't remove if it deletes > 5% of data
    n_extreme = len(df[df['cluster'] == extreme_cluster_id])
    pct_extreme = (n_extreme / len(df)) * 100
    
    print(f"   > Clusters Found (Avg Lengths): {sorted(cluster_means.values.astype(int))}")
    print(f"   > Potential Outliers (Cluster {extreme_cluster_id}): {n_extreme} docs ({pct_extreme:.2f}%)")
    
    if pct_extreme < 5.0 and extreme_cluster_mean > 10000: # Threshold: must be <5% count AND >10k words
        print(f"   > Action: REMOVING {n_extreme} extreme outliers.")
        df_clean = df[df['cluster'] != extreme_cluster_id]
        return Dataset.from_pandas(df_clean.drop(columns=['length', 'cluster']))
    else:
        print(f"   > Action: KEEPING all data. (Outliers not distinct enough or too numerous to delete safely).")
        return Dataset.from_pandas(df.drop(columns=['length', 'cluster']))

def preprocess_pipeline():
    # 1. Load
    raw_dataset = load_data()
    processed_splits = {}
    
    print("\n" + "="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)

    for split in ['train', 'validation', 'test']:
        if split in raw_dataset:
            print(f"\nProcessing Split: {split.upper()}")
            print(f"   > Original Count: {len(raw_dataset[split])}")
            
            # A. Clean Text
            # We map the function to the dataset
            current_ds = raw_dataset[split].map(
                lambda x: {'text': clean_text_minimal(x['text'])},
                desc="Cleaning text"
            )
            
            # B. Remove Duplicates
            current_ds = remove_duplicates(current_ds)
            
            # C. Remove Outliers (Only for Training Data usually, but useful for analysis)
            # We usually don't remove outliers from Test/Validation to keep benchmarks fair,
            # but for this specific request, we will apply it to Train only.
            if split == 'train':
                current_ds = detect_and_remove_outliers(current_ds, split)
            
            print(f"   > Final Count:    {len(current_ds)}")
            processed_splits[split] = current_ds

    # 2. Reassemble into DatasetDict
    final_dataset = DatasetDict(processed_splits)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    return final_dataset

if __name__ == "__main__":
    clean_dataset = preprocess_pipeline()
    
    # Quick sanity check of a sample
    print("\nSample Processed Text (Train[0]):")
    print(clean_dataset['train'][0]['text'][:300] + "...")