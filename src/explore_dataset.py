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
    print(f"\nSample from training set:")
    print(f"  - Text length: {len(sample['facts'])} characters")
    print(f"  - First 300 chars of facts: {sample['facts'][:300]}...")
    print(f"  - Labels: {sample['labels']}")
    
    return sample

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

def check_missing_duplicates(dataset):
    """Check for missing values and duplicates"""
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            print(f"\n{split.upper()} SET:")
            
            # Check for missing values
            missing_facts = sum(1 for x in dataset[split] if not x['facts'] or len(x['facts'].strip()) == 0)
            missing_labels = sum(1 for x in dataset[split] if not x['labels'] or len(x['labels']) == 0)
            
            print(f"  - Missing facts: {missing_facts}")
            print(f"  - Missing labels: {missing_labels}")
            
            # Check for duplicates
            facts_list = [x['facts'] for x in dataset[split]]
            unique_facts = len(set(facts_list))
            duplicates = len(facts_list) - unique_facts
            
            print(f"  - Duplicate samples: {duplicates}")

if __name__ == "__main__":
    # Load dataset
    dataset = load_ecthr_dataset()
    
    # Explore structure
    explore_dataset_structure(dataset)
    
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
