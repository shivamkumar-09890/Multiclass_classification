"""
Task 5 (Grand Visual): Compare All 3 Models
1. Generic Baseline (Untrained Longformer) - The "Idiot" Baseline
2. Legal Expert (Legal-BERT) - The "Smart but Short-sighted" Baseline
3. Fine-Tuned (Your Model) - The "Smart and Long-sighted" Champion

Output: CSVs and 3-Bar Charts in '../comparison_results'
"""

import os
import torch
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODELS = {
    "Generic Baseline": "allenai/longformer-base-4096", # Untrained
    "Legal-BERT":       "./legal_bert_baseline",        # From Task 5a
    "Fine-Tuned":       "./ecthr_longformer_finetuned"  # From Task 4
}

# Where to save graphs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comparison_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GrandComparison:
    def __init__(self):
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
        # Load the "Ground Truth" Label Encoder from your main model
        # We use this ONE encoder for all 3 models to ensure consistency
        mlb_path = os.path.join(MODELS["Fine-Tuned"], "mlb.pkl")
        if os.path.exists(mlb_path):
            self.mlb = joblib.load(mlb_path)
            self.classes = self.mlb.classes_
            self.num_labels = len(self.classes)
            print(f"[Init] Loaded Label Encoder ({self.num_labels} labels)")
        else:
            raise FileNotFoundError("Run Task 4 first to generate mlb.pkl")

    def normalize_text(self, txt):
        if isinstance(txt, list): return " ".join(str(x) for x in txt)
        return str(txt)

    def get_test_data(self):
        print("\n[Data] Loading Test Set Raw Text...")
        dataset = load_dataset("lex_glue", "ecthr_b", split="test")
        
        texts = [self.normalize_text(t) for t in dataset["text"]]
        
        # Ground Truth Labels
        true_labels = []
        for example in dataset:
            true_labels.append(self.mlb.transform([example['labels']])[0])
            
        return texts, np.array(true_labels)

    def evaluate_single_model(self, model_name, model_path, raw_texts, true_labels):
        print(f"\n" + "="*50)
        print(f" EVALUATING: {model_name}")
        print("="*50)

        # 1. Check existence
        # (Generic baseline is a cloud path, so we skip file check for it)
        if "allenai" not in model_path and not os.path.exists(model_path):
            print(f"⚠️ SKIPPING {model_name}: Path '{model_path}' not found.")
            print("   (Did you run Task 5a for Legal-BERT?)")
            return None

        # 2. Dynamic Configuration based on Model Type
        # Legal-BERT allows 512 tokens. Longformer allows 4096.
        if "bert" in model_path.lower() or "bert" in model_name.lower():
            max_len = 512
            batch_size = 16 # BERT is smaller, can handle bigger batches
            print(f"   -> Detected BERT architecture. Max Length: {max_len}")
        else:
            max_len = 4096
            batch_size = 2  # Longformer is huge, needs small batch
            print(f"   -> Detected Longformer architecture. Max Length: {max_len}")

        # 3. Load Model & Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=self.num_labels, 
                problem_type="multi_label_classification"
            ).to(DEVICE).eval()
        except OSError:
            print(f"❌ Error loading {model_name}. Skipping.")
            return None

        # 4. Inference Loop
        all_probs = []
        print(f"   -> Running Inference on {len(raw_texts)} docs...")
        
        # We manually batch the raw text here
        for i in range(0, len(raw_texts), batch_size):
            batch_texts = raw_texts[i : i+batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                truncation=True, 
                padding="max_length", 
                max_length=max_len, 
                return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(): # FP16
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.cpu().float().numpy())
        
        all_probs = np.array(all_probs)
        pred_binary = (all_probs > 0.5).astype(int)

        # 5. Metrics
        return {
            "Model": model_name,
            "Accuracy": accuracy_score(true_labels, pred_binary),
            "F1 Micro": f1_score(true_labels, pred_binary, average='micro', zero_division=0),
            "F1 Macro": f1_score(true_labels, pred_binary, average='macro', zero_division=0),
            "Precision": precision_score(true_labels, pred_binary, average='micro', zero_division=0),
            "Recall": recall_score(true_labels, pred_binary, average='micro', zero_division=0),
            "Hamming Loss": hamming_loss(true_labels, pred_binary)
        }

    def save_and_plot(self, results):
        if not results: return

        df = pd.DataFrame(results)
        
        # --- Save CSV ---
        csv_path = os.path.join(OUTPUT_DIR, "grand_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Output] Saved CSV: {csv_path}")
        print("\n" + df.to_string())

        # --- Plot 1: Scores (Bar Chart) ---
        score_metrics = ["Accuracy", "F1 Micro", "F1 Macro", "Precision", "Recall"]
        df_melt = df.melt(id_vars="Model", value_vars=score_metrics, var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="magma")
        
        ax.set_title("Comparison of 3 Architectures (Higher is Better)", fontsize=16)
        ax.set_ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)

        plot_path = os.path.join(OUTPUT_DIR, "grand_comparison_scores.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Output] Saved Plot: {plot_path}")

        # --- Plot 2: Loss (Lower is Better) ---
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=df, x="Model", y="Hamming Loss", palette="Reds_d")
        ax.set_title("Hamming Loss (Lower is Better)", fontsize=14)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f')
            
        loss_path = os.path.join(OUTPUT_DIR, "grand_comparison_loss.png")
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Output] Saved Plot: {loss_path}")

    def run(self):
        texts, labels = self.get_test_data()
        results = []
        
        for name, path in MODELS.items():
            res = self.evaluate_single_model(name, path, texts, labels)
            if res:
                results.append(res)
        
        self.save_and_plot(results)

if __name__ == "__main__":
    GrandComparison().run()