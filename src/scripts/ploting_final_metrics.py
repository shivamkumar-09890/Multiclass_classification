import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

# --- CONFIGURATION ---
# Replace with your actual filename if different
INPUT_FILE = "predictions_FULL_DATASET_Stage_2__Weighted_Refinement_.csv" 
# (Or whatever your raw file is named. I will assume it follows the format you pasted)

OUTPUT_DIR = "analysis_results"
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_parse_data(filepath):
    print(f"Loading raw data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except:
        # Fallback if file not found, creating dummy data for demonstration
        print("⚠️ File not found. Generating dummy data structure for test run.")
        columns = ["Index"]
        for i in range(10):
            columns.extend([f"Article_{i}_True", f"Article_{i}_Prob"])
        df = pd.DataFrame(np.random.rand(100, 21), columns=columns)
        for i in range(10):
            df[f"Article_{i}_True"] = (df[f"Article_{i}_True"] > 0.8).astype(int)

    # Extract columns systematically
    true_cols = [c for c in df.columns if "_True" in c]
    prob_cols = [c for c in df.columns if "_Prob" in c]
    
    # Ensure they are sorted so Article 0 matches Article 0
    true_cols.sort()
    prob_cols.sort()
    
    y_true = df[true_cols].values
    y_prob = df[prob_cols].values
    class_names = [c.replace("_True", "") for c in true_cols]
    
    return y_true, y_prob, class_names

def plot_probability_distributions(y_true, y_prob, class_names):
    """
    Shows how confident the model is for Positive vs Negative cases.
    Good models have peaks at 0 (for Negatives) and 1 (for Positives).
    """
    print("Generating Probability Distribution plots...")
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(n_classes):
        ax = axes[i]
        pos_probs = y_prob[y_true[:, i] == 1, i]
        neg_probs = y_prob[y_true[:, i] == 0, i]
        
        sns.histplot(neg_probs, color="red", alpha=0.3, label="Negative", bins=20, ax=ax, stat="density")
        sns.histplot(pos_probs, color="green", alpha=0.3, label="Positive", bins=20, ax=ax, stat="density")
        
        ax.set_title(class_names[i])
        ax.set_xlim(0, 1)
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/probability_distributions_final.png", dpi=300)

def find_optimal_thresholds(y_true, y_prob, class_names):
    """
    Finds the threshold that maximizes F1 score for EACH class independently.
    Standard is 0.5, but for rare classes, 0.2 might be better.
    """
    print("\nFinding Optimal Thresholds per Class...")
    optimal_thresholds = {}
    
    print(f"{'Class':<15} | {'Best Thresh':<12} | {'F1 @ Best':<10} | {'F1 @ 0.50':<10}")
    print("-" * 55)
    
    for i, name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        
        # Calculate F1 for every possible threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)
        
        # Find max
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        # Compare with default
        default_pred = (y_prob[:, i] > 0.5).astype(int)
        default_f1 = f1_score(y_true[:, i], default_pred, zero_division=0)
        
        optimal_thresholds[name] = best_thresh
        print(f"{name:<15} | {best_thresh:.4f}       | {best_f1:.4f}     | {default_f1:.4f}")
        
    return optimal_thresholds

def plot_confusion_matrices(y_true, y_prob, class_names):
    """
    Plots a grid of confusion matrices (one for each class).
    """
    print("Generating Confusion Matrices...")
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(n_classes):
        y_pred = (y_prob[:, i] > 0.5).astype(int)
        cm = confusion_matrix(y_true[:, i], y_pred)
        
        # Normalize just to see ratios clearly
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=axes[i], cbar=False)
        axes[i].set_title(class_names[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        axes[i].set_xticklabels(['Neg', 'Pos'])
        axes[i].set_yticklabels(['Neg', 'Pos'])
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices_final.png", dpi=300)

def analyze_hardest_samples(y_true, y_prob, class_names):
    """
    Identifies the single most 'Confidently Wrong' prediction for each class.
    """
    print("\n" + "="*60)
    print("HARDEST SAMPLES (Where the model was most confidently wrong)")
    print("="*60)
    
    for i, name in enumerate(class_names):
        # False Positives: True=0, Prob is High
        # We look for max probability where True is 0
        neg_indices = np.where(y_true[:, i] == 0)[0]
        if len(neg_indices) > 0:
            fp_idx = neg_indices[np.argmax(y_prob[neg_indices, i])]
            fp_conf = y_prob[fp_idx, i]
            
            # False Negatives: True=1, Prob is Low
            # We look for min probability where True is 1
            pos_indices = np.where(y_true[:, i] == 1)[0]
            if len(pos_indices) > 0:
                fn_idx = pos_indices[np.argmin(y_prob[pos_indices, i])]
                fn_conf = y_prob[fn_idx, i]
                
                print(f"[{name}]")
                print(f"  Worst False Positive (Index {fp_idx}): Pred {fp_conf:.4f} vs True 0")
                print(f"  Worst False Negative (Index {fn_idx}): Pred {fn_conf:.4f} vs True 1")
                print("-" * 30)

def main():
    # 1. Parse
    y_true, y_prob, class_names = load_and_parse_data(INPUT_FILE)
    
    # 2. Distributions (Are we calibrated?)
    plot_probability_distributions(y_true, y_prob, class_names)
    
    # 3. Thresholds (Can we squeeze out more F1?)
    find_optimal_thresholds(y_true, y_prob, class_names)
    
    # 4. Visuals
    plot_confusion_matrices(y_true, y_prob, class_names)
    
    # 5. Error Analysis
    analyze_hardest_samples(y_true, y_prob, class_names)
    
    print("\nAnalysis Complete. Check the 'analysis_results' folder for images.")

if __name__ == "__main__":
    main()