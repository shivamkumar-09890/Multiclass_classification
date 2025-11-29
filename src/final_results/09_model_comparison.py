import torch
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report
import gc
import os

# --- CONFIGURATION (UPDATED) ---
MODELS = {
    # 1. The General Model (Initial strategy: Standard Loss)
    "Stage 1 (General Training)": {
        "path": "./ecthr_longformer_finetuned", 
        "max_len": 4096
    },
    
    # 2. The Refined Model (Retrained strategy: Weighted Loss)
    "Stage 2 (Weighted Refinement)": {
        "path": "./ecthr_longformer_weighted", 
        "max_len": 4096
    }
}

DATA_PATH = "processed_ecthr_b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model_name, config, dataset):
    print(f"\n[Process] Loading {model_name} from {config['path']}...")
    
    try:
        # Load Resources
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = AutoModelForSequenceClassification.from_pretrained(config['path'])
        mlb = joblib.load(f"{config['path']}/mlb.pkl")
        
        model.to(device)
        model.eval()
        
        print(f"   > Running Inference (Max Len: {config['max_len']})...")
        
        preds = []
        truths = []
        
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
            
            preds.append(probs.cpu().numpy()[0])
            truths.append(true_labels)
            
            if i % 200 == 0: print(f"     Processed {i}/{len(dataset)}")

        # Clean up GPU memory
        del model
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return np.array(truths), np.array(preds), mlb.classes_

    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None, None

def plot_comparisons(results):
    print("\n[Analysis] Generating Comparison Plots...")
    sns.set_theme(style="whitegrid")
    
    global_metrics = []
    class_metrics = []

    for name, data in results.items():
        y_true = data['y_true']
        y_pred = (data['y_pred'] > 0.5).astype(int)
        classes = data['classes']
        
        # 1. Global Stats
        mic = f1_score(y_true, y_pred, average='micro', zero_division=0)
        mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        global_metrics.append({
            "Model Strategy": name,
            "Micro F1": mic,
            "Macro F1": mac,
            "Exact Match Acc": acc
        })
        
        # 2. Class Stats
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for i, cls_label in enumerate(classes):
            key = str(i)
            if key in report:
                class_metrics.append({
                    "Model Strategy": name,
                    "Article": f"Art {cls_label}",
                    "F1 Score": report[key]['f1-score']
                })

    df_global = pd.DataFrame(global_metrics)
    df_class = pd.DataFrame(class_metrics)

    # --- PLOT 1: GLOBAL METRICS ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df_melt = df_global.melt(id_vars="Model Strategy", var_name="Metric", value_name="Score")
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model Strategy", palette="viridis", ax=ax1)
    ax1.set_title("Strategy Comparison: General vs. Weighted Refinement", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("comparison_global_metrics.png", dpi=300)
    
    # --- PLOT 2: PER-CLASS PERFORMANCE ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df_class, x="Article", y="F1 Score", hue="Model Strategy", 
        style="Model Strategy", markers=True, markersize=10, linewidth=2.5, palette="viridis", ax=ax2
    )
    ax2.set_title("Impact of Weighted Refinement on Individual Articles", fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("comparison_class_wise.png", dpi=300)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(df_global)

if __name__ == "__main__":
    print("Loading Test Data...")
    try:
        dataset = load_from_disk(DATA_PATH)['test']
    except Exception as e:
        print(f"Error: Could not load data from {DATA_PATH}. Check path.")
        exit()
    
    results = {}
    for name, config in MODELS.items():
        if not os.path.exists(config['path']):
            print(f"⚠️ Skipping {name}: Path not found ({config['path']})")
            continue
            
        y_true, y_pred, classes = get_predictions(name, config, dataset)
        if y_true is not None:
            results[name] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'classes': classes
            }
    
    if results:
        plot_comparisons(results)
    else:
        print("No models were successfully loaded.")