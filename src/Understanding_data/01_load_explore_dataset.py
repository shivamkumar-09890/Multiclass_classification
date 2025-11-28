"""
Task 1: Load and Explore ECTHR_B Dataset
- Loads the dataset from HuggingFace
- Analyzes dataset structure and features
- Provides overview of labels and data distribution
"""

import json
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter

def load_ecthr_dataset():
    """Load ECTHR_B dataset from HuggingFace"""
    print("[v0] Loading ECTHR_B dataset from HuggingFace...")
    dataset = load_dataset("lex_glue", "ecthr_b")
    return dataset

def explore_dataset_structure(dataset):
    """Explore and describe dataset structure"""
    print("\n" + "="*80)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*80)
    
    # Basic info
    print(f"\nDataset splits: {list(dataset.keys())}")
    
    for split in dataset.keys():
        print(f"\n{split.upper()} SET:")
        print(f"  - Number of samples: {len(dataset[split])}")
        print(f"  - Features: {dataset[split].column_names}")
    
    # Sample inspection
    print("\n" + "-"*80)
    print("SAMPLE DATA:")
    print("-"*80)
    sample = dataset['train'][0]
    
    # --- FIX STARTS HERE ---
    # 1. Join the list of strings into one large string
    full_text = " ".join(sample['text'])
    
    print(f"\nSample from training set:")
    # Update label to reflect we are counting chars of the joined text, not list items
    print(f"  - Text length: {len(full_text)} characters") 
    # Now slice the string, not the list
    print(f"  - First 300 chars of facts: {full_text[:300]}...")
    # --- FIX ENDS HERE ---
    
    print(f"  - Labels: {sample['labels']}")
    
    return sample

def analyze_word_counts(dataset):
    """Analyze word counts across all splits"""
    print("\n" + "="*80)
    print("WORD COUNT ANALYSIS")
    print("="*80)
    
    global_max_words = 0
    global_max_info = {}
    
    for split in dataset.keys():
        print(f"\nAnalyzing {split.upper()} set...")
        word_counts = []
        
        for i, sample in enumerate(dataset[split]):
            # 1. Join the list of paragraphs into one string
            full_text = " ".join(sample['text'])
            
            # 2. Split by whitespace to get word count
            # (Note: This is an approximation. For exact tokens, use a tokenizer)
            n_words = len(full_text.split())
            word_counts.append(n_words)
            
            # Track global maximum
            if n_words > global_max_words:
                global_max_words = n_words
                global_max_info = {'split': split, 'index': i}

        # Print statistics for this split
        print(f"  - Average words: {np.mean(word_counts):.0f}")
        print(f"  - Median words:  {np.median(word_counts):.0f}")
        print(f"  - Max words:     {np.max(word_counts)}")
        
    print("\n" + "-"*80)
    print(f"OVERALL MAXIMUM WORD COUNT: {global_max_words}")
    print(f"Found in: {global_max_info['split']} set, Sample Index: {global_max_info['index']}")
    print("-"*80)

def analyze_labels(dataset):
    """Analyze label distribution and statistics"""
    print("\n" + "="*80)
    print("LABEL ANALYSIS")
    print("="*80)
    
    # Get all unique labels across dataset
    all_labels = set()
    label_counts = Counter()
    sample_label_counts = []
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            for example in dataset[split]:
                labels = example['labels']
                all_labels.update(labels)
                label_counts.update(labels)
                sample_label_counts.append(len(labels))
    
    print(f"\nTotal unique labels: {len(all_labels)}")
    print(f"Label IDs range: {min(all_labels)} to {max(all_labels)}")
    
    print(f"\nLabels per sample statistics:")
    print(f"  - Mean labels per sample: {np.mean(sample_label_counts):.2f}")
    print(f"  - Median labels per sample: {np.median(sample_label_counts):.2f}")
    print(f"  - Min: {np.min(sample_label_counts)}, Max: {np.max(sample_label_counts)}")
    
    print(f"\nTop 10 most common labels:")
    for label_id, count in label_counts.most_common(10):
        print(f"  - Article {label_id}: {count} occurrences")
    
    print(f"\nBottom 5 least common labels:")
    for label_id, count in sorted(label_counts.items(), key=lambda x: x[1])[:5]:
        print(f"  - Article {label_id}: {count} occurrences")
    
    return all_labels, label_counts

def normalize_text(t):
    if t is None:
        return ""
    if isinstance(t, list):
        # join list into a string if list of strings
        return " ".join(str(x) for x in t)
    return str(t)  # convert anything else to string

def check_missing_duplicates(dataset):
    """Check for missing values and duplicates"""
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            print(f"\n{split.upper()} SET:")
            
            # Check for missing values
            missing_facts = sum(
                1
                for x in dataset[split]
                if len(normalize_text(x['text']).strip()) == 0
            )
            missing_labels = sum(1 for x in dataset[split] if not x['labels'] or len(x['labels']) == 0)
            
            print(f"  - Missing facts: {missing_facts}")
            print(f"  - Missing labels: {missing_labels}")
            
            # Check for duplicates
            # facts_list = [x['text'] for x in dataset[split]]
            facts_list = [normalize_text(x['text']) for x in dataset[split]]
            unique_facts = len(set(facts_list))
            duplicates = len(facts_list) - unique_facts
            
            print(f"  - Duplicate samples: {duplicates}")

if __name__ == "__main__":
    # Load dataset
    dataset = load_ecthr_dataset()
    
    # Explore structure
    explore_dataset_structure(dataset)
    analyze_word_counts(dataset)
    
    # Analyze labels
    all_labels, label_counts = analyze_labels(dataset)
    
    # Check data quality
    check_missing_duplicates(dataset)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Dataset successfully loaded and explored")
    print(f"✓ Total samples: {len(dataset['train']) + len(dataset['validation']) + len(dataset['test'])}")
    print(f"✓ Unique ECHR articles: {len(all_labels)}")
    print(f"✓ This is a multi-label classification task (multiple articles per case)")
