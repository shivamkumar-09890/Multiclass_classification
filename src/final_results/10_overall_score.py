import torch
import numpy as np
import joblib
import pandas as pd
from datasets import load_from_disk, concatenate_datasets  # Added concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import gc
import os
import re

# --- CONFIGURATION ---
MODELS = {
    "Stage 1 (General Training)": {
        "path": "./ecthr_longformer_finetuned", 
        "max_len": 4096
    },
    "Stage 2 (Weighted Refinement)": {
        "path": "./ecthr_longformer_weighted", 
        "max_len": 4096
    }
}

DATA_PATH = "processed_ecthr_b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model_name, config, dataset):
    print(f"\n[Process] Loading {model_name}...")
    try:
        # Load Resources
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = AutoModelForSequenceClassification.from_pretrained(config['path'])
        mlb = joblib.load(f"{config['path']}/mlb.pkl")
        
        model.to(device)
        model.eval()
        
        preds_list = []
        truths_list = []
        
        print(f"   > Running Inference on FULL dataset ({len(dataset)} samples)...")
        
        # Inference Loop
        for i in range(len(dataset)):
            text = dataset[i]['text']
            true_labels = mlb.transform([dataset[i]['labels']])[0]
            
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=config['max_len']
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits)
            
            preds_list.append(probs.cpu().numpy()[0])
            truths_list.append(true_labels)
            
            # Print progress every 500 samples
            if i % 500 == 0 and i > 0: 
                print(f"     Processed {i}/{len(dataset)}")

        # Clean up GPU memory
        del model
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return np.array(truths_list), np.array(preds_list), mlb.classes_

    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None, None

def save_predictions_to_csv(model_name, y_true, y_probs, classes):
    print(f"   > Saving raw predictions to CSV for {model_name}...")
    
    data = {}
    for idx, cls_label in enumerate(classes):
        data[f'Article_{cls_label}_True'] = y_true[:, idx].astype(int)
        data[f'Article_{cls_label}_Prob'] = y_probs[:, idx]

    df = pd.DataFrame(data)
    
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
    filename = f"predictions_FULL_DATASET_{clean_name}.csv"
    
    df.to_csv(filename, index_label="Index")
    print(f"   ✅ File saved: {filename}")

def calculate_comprehensive_metrics(y_true, y_probs, threshold=0.5):
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    exact_acc = accuracy_score(y_true, y_pred_binary)
    hamming = hamming_loss(y_true, y_pred_binary)
    
    return micro, macro, exact_acc, hamming

if __name__ == "__main__":
    print("Loading Data...")
    try:
        # 1. Load the dataset dictionary (Train/Test/Val)
        data_dict = load_from_disk(DATA_PATH)
        
        # 2. Concatenate ALL splits into one (approx 11k)
        print("   > Merging Train + Validation + Test splits...")
        dataset = concatenate_datasets([data_dict['train'], data_dict['validation'], data_dict['test']])
        print(f"   > Total Samples Loaded: {len(dataset)}")
        
    except Exception as e:
        print(f"Error: Could not load data from {DATA_PATH}. Check path.")
        print(e)
        exit()
    
    final_metrics = []

    for name, config in MODELS.items():
        if not os.path.exists(config['path']):
            print(f"⚠️ Skipping {name}: Path not found.")
            continue
            
        y_true, y_probs, classes = get_predictions(name, config, dataset)
        
        if y_true is not None:
            # Save raw predictions
            save_predictions_to_csv(name, y_true, y_probs, classes)
            
            # Calculate metrics
            mic, mac, acc, hamm = calculate_comprehensive_metrics(y_true, y_probs)
            
            final_metrics.append({
                "Model Strategy": name,
                "Micro F1": f"{mic:.4f}",
                "Macro F1": f"{mac:.4f}",
                "Exact Match Acc": f"{acc:.4f}",
                "Hamming Loss": f"{hamm:.4f}"
            })
    
    if final_metrics:
        df_results = pd.DataFrame(final_metrics)
        print("\n" + "="*80)
        print("FINAL EVALUATION METRICS ON COMPLETE DATASET (11k Samples)")
        print("="*80)
        print(df_results.to_string(index=False))
        
        df_results.to_csv("final_model_metrics_full_dataset.csv", index=False)
        print("\n✅ Metrics summary saved to 'final_model_metrics_full_dataset.csv'")
    else:
        print("No models were evaluated.")