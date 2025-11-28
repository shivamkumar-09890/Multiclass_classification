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

# --- CONFIGURATION (UPDATE THESE PATHS) ---
MODELS = {
    # 1. The Baseline (First model you trained, standard loss)
    "Model 1 (Baseline)": {
        "path": "./ecthr_longformer_finetuned", 
        "max_len": 4096
    },
    
    # 2. The Specialist (The Weighted one we just finished)
    "Model 2 (Weighted)": {
        "path": "./ecthr_longformer_weighted", 
        "max_len": 4096
    },
    
    # 3. The Short/Frozen (The 512 token one)--->this is dropped
    "Model 3 (Short/512)": {
        "path": "./ecthr_bert_short_frozen", 
        "max_len": 512
    }
}

DATA_PATH = "processed_ecthr_b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model_name, config, dataset):
    print(f"\n[Process] Loading {model_name} from {config['path']}...")
    
    try:
        # Load Resources
        # Note: We load the tokenizer from the saved folder to ensure config matches
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = AutoModelForSequenceClassification.from_pretrained(config['path'])
        mlb = joblib.load(f"{config['path']}/mlb.pkl")
        
        model.to(device)
        model.eval()
        
        print(f"   > Running Inference (Max Len: {config['max_len']})...")
        
        preds = []
        truths = []
        
        # Inference Loop
        # We process one-by-one to avoid OOM since we are switching models
        for i in range(len(dataset)):
            text = dataset[i]['text']
            
            # Use the MLB from THIS model to ensure correct mapping
            true_labels = mlb.transform([dataset[i]['labels']])[0]
            
            # Tokenize
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

        # Clean up GPU memory immediately
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
    
    # Dataframes for plotting
    global_metrics = []
    class_metrics = []

    for name, data in results.items():
        y_true = data['y_true']
        # Convert probabilities to binary predictions
        y_pred = (data['y_pred'] > 0.5).astype(int)
        classes = data['classes']
        
        # 1. Global Stats
        mic = f1_score(y_true, y_pred, average='micro', zero_division=0)
        mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        global_metrics.append({
            "Model": name,
            "Micro F1": mic,
            "Macro F1": mac,
            "Exact Match Acc": acc
        })
        
        # 2. Class Stats
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        # We assume classes are 0..9, mapped to indices
        for i, cls_label in enumerate(classes):
            # The report keys are usually strings of the index "0", "1"...
            key = str(i)
            if key in report:
                class_metrics.append({
                    "Model": name,
                    "Article": f"Art {cls_label}",
                    "F1 Score": report[key]['f1-score']
                })

    df_global = pd.DataFrame(global_metrics)
    df_class = pd.DataFrame(class_metrics)

    # --- PLOT 1: GLOBAL METRICS (Bar Chart) ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df_melt = df_global.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="viridis", ax=ax1)
    ax1.set_title("Global Performance Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("comparison_global_metrics.png", dpi=300)
    
    # --- PLOT 2: PER-CLASS PERFORMANCE (Line Chart) ---
    # This is where Model 2 (Weighted) should shine on rare classes
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df_class, x="Article", y="F1 Score", hue="Model", 
        style="Model", markers=True, markersize=10, linewidth=2.5, palette="viridis", ax=ax2
    )
    ax2.set_title("Class-Wise Performance (The Impact of Weights)", fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("comparison_class_wise.png", dpi=300)

    # --- PLOT 3: RADAR CHART ---
    # Prepare data
    categories = list(df_global.columns[1:])
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("viridis", len(results))

    for i, (name, row) in enumerate(df_global.set_index("Model").iterrows()):
        values = row.tolist()
        values += values[:1]
        ax3.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i])
        ax3.fill(angles, values, color=colors[i], alpha=0.1)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_title("Holistic Model Comparison", fontsize=15, y=1.1)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig("comparison_radar.png", dpi=300)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("1. comparison_global_metrics.png (Summary)")
    print("2. comparison_class_wise.png (Detailed Analysis)")
    print("3. comparison_radar.png (Overview)")
    print("\nGlobal Metrics Table:")
    print(df_global)

if __name__ == "__main__":
    # 1. Load Test Data
    print("Loading Test Data...")
    dataset = load_from_disk(DATA_PATH)['test']
    
    # 2. Run Inference for all models
    results = {}
    for name, config in MODELS.items():
        # Skip if path doesn't exist (e.g., if you haven't trained Model 3 yet)
        import os
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
    
    # 3. Plot
    if results:
        plot_comparisons(results)
    else:
        print("No models were successfully loaded.")