import torch
import numpy as np
import joblib
import json
import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, accuracy_score

# --- CONFIGURATION ---
MODEL_PATH = "./ecthr_longformer_weighted"
DATA_PATH = "processed_ecthr_b"
OUTPUT_FILE = "final_test_results.txt"
JSON_FILE = "test_metrics.json"

def run_final_evaluation():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Resources
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    mlb = joblib.load(f"{MODEL_PATH}/mlb.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Load Data
    print("Loading test data...")
    dataset = load_from_disk(DATA_PATH)
    test_data = dataset['test']

    # 3. Prediction Loop
    print("Running predictions on Test set...")
    all_preds = []
    all_labels = []

    for i in range(len(test_data)):
        text = test_data[i]['text']
        true_labels = mlb.transform([test_data[i]['labels']])[0]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
        
        all_preds.append(probs.cpu().numpy()[0])
        all_labels.append(true_labels)
        
        if i % 100 == 0: print(f"Processed {i}/{len(test_data)}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    # 4. Generate & Save Report
    target_names = [f"Article {c}" for c in mlb.classes_]
    
    # Calculate Metrics
    acc = accuracy_score(all_labels, pred_binary)
    f1_mic = f1_score(all_labels, pred_binary, average='micro')
    f1_mac = f1_score(all_labels, pred_binary, average='macro')
    report_str = classification_report(all_labels, pred_binary, target_names=target_names, zero_division=0)
    report_dict = classification_report(all_labels, pred_binary, target_names=target_names, zero_division=0, output_dict=True)

    # --- SAVE TO TEXT FILE (For Reading) ---
    with open(OUTPUT_FILE, "w") as f:
        f.write("="*60 + "\n")
        f.write("FINAL RESULTS (Weighted Loss + Clean Data)\n")
        f.write("="*60 + "\n")
        f.write(f"Exact Match Accuracy: {acc:.4f}\n")
        f.write(f"F1 Micro:             {f1_mic:.4f}\n")
        f.write(f"F1 Macro:             {f1_mac:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("CLASS-WISE PERFORMANCE:\n")
        f.write(report_str)
        f.write("\n" + "="*60 + "\n")

    # --- SAVE TO JSON FILE (For Graphs) ---
    metrics_data = {
        "accuracy": acc,
        "f1_micro": f1_mic,
        "f1_macro": f1_mac,
        "class_wise": report_dict
    }
    with open(JSON_FILE, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("\n" + "="*60)
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print(f"✓ Metrics JSON saved to: {JSON_FILE}")
    print("="*60)

if __name__ == "__main__":
    run_final_evaluation()