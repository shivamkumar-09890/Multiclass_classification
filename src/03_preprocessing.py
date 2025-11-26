"""
Task 3: Data Preprocessing and Splitting
- Cleans and normalizes text
- Handles missing values and duplicates
- Prepares data for model training
"""

from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_dataset_ecthr():
    """Load ECTHR_B dataset"""
    print("[v0] Loading ECTHR_B dataset...")
    dataset = load_dataset("lex_glue", "ecthr_b")
    return dataset

def clean_text(text):
    """Clean and normalize legal text"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation for legal text
    text = re.sub(r'[^\w\s.,;:\-]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def preprocess_dataset(dataset):
    """Preprocess the entire dataset"""
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    processed_data = {}
    
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            print(f"\nProcessing {split} set...")
            
            # Check for issues before processing
            print(f"  - Before: {len(dataset[split])} samples")
            
            # Remove duplicates and empty samples
            data_dict = {'facts': [], 'labels': []}
            seen_facts = set()
            
            for example in dataset[split]:
                facts = clean_text(example['facts'])
                labels = example['labels']
                
                # Skip empty or duplicate samples
                if facts and len(facts.strip()) > 0 and facts not in seen_facts and labels:
                    seen_facts.add(facts)
                    data_dict['facts'].append(facts)
                    data_dict['labels'].append(labels)
            
            print(f"  - After: {len(data_dict['facts'])} samples")
            print(f"  - Duplicates/empty removed: {len(dataset[split]) - len(data_dict['facts'])}")
            
            # Create processed dataset
            processed_data[split] = data_dict
    
    return processed_data

def analyze_preprocessing_effects(original, processed):
    """Analyze the effects of preprocessing"""
    print("\n" + "="*80)
    print("PREPROCESSING EFFECTS")
    print("="*80)
    
    for split in ['train', 'validation', 'test']:
        if split in original:
            original_size = len(original[split])
            processed_size = len(processed[split]['facts'])
            removed = original_size - processed_size
            
            print(f"\n{split.upper()} SET:")
            print(f"  - Original samples: {original_size}")
            print(f"  - After preprocessing: {processed_size}")
            print(f"  - Samples removed: {removed} ({removed/original_size*100:.2f}%)")
            
            # Sample before and after
            print(f"\n  - Original sample:")
            print(f"    {original[split][0]['facts'][:100]}...")
            print(f"  - Cleaned sample:")
            print(f"    {processed[split]['facts'][0][:100]}...")

def create_train_val_split(processed_data):
    """Create train/validation split from preprocessed data"""
    print("\n" + "="*80)
    print("DATA SPLITTING FOR TRAINING")
    print("="*80)
    
    # Use the provided train split for training
    train_facts = processed_data['train']['facts']
    train_labels = processed_data['train']['labels']
    
    # Further split train into train and validation (80-20)
    X_train, X_val, y_train, y_val = train_test_split(
        train_facts, train_labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"\nTrain/Validation Split (from original train set):")
    print(f"  - Training samples: {len(X_train)} (80%)")
    print(f"  - Validation samples: {len(X_val)} (20%)")
    print(f"  - Test samples: {len(processed_data['test']['facts'])}")
    
    split_data = {
        'train': {'facts': X_train, 'labels': y_train},
        'validation': {'facts': X_val, 'labels': y_val},
        'test': processed_data['test']
    }
    
    return split_data

def get_label_statistics(split_data):
    """Get label statistics for each split"""
    print("\n" + "="*80)
    print("LABEL STATISTICS AFTER PREPROCESSING")
    print("="*80)
    
    for split in ['train', 'validation', 'test']:
        if split in split_data:
            labels_list = split_data[split]['labels']
            
            # Count label distribution
            from collections import Counter
            label_counter = Counter()
            labels_per_sample = []
            
            for labels in labels_list:
                label_counter.update(labels)
                labels_per_sample.append(len(labels))
            
            print(f"\n{split.upper()} SET:")
            print(f"  - Total samples: {len(labels_list)}")
            print(f"  - Unique articles: {len(label_counter)}")
            print(f"  - Avg labels per sample: {np.mean(labels_per_sample):.2f}")
            print(f"  - Min/Max labels: {min(labels_per_sample)}/{max(labels_per_sample)}")

if __name__ == "__main__":
    # Load original dataset
    original_dataset = load_dataset_ecthr()
    
    # Preprocess
    processed_data = preprocess_dataset(original_dataset)
    
    # Analyze preprocessing effects
    analyze_preprocessing_effects(original_dataset, processed_data)
    
    # Create train/val/test split
    split_data = create_train_val_split(processed_data)
    
    # Get label statistics
    get_label_statistics(split_data)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print("✓ Data cleaned and normalized")
    print("✓ Duplicates and empty samples removed")
    print("✓ Train/validation/test splits created")
    print("✓ Ready for model training")
