import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

# --- CONFIGURATION ---
BASELINE_FILE = "predictions_FULL_DATASET_Stage_1__General_Training_.csv"
WEIGHTED_FILE = "predictions_FULL_DATASET_Stage_2__Weighted_Refinement_.csv"
OUTPUT_DIR = "comparison_results"

import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_parse_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        return None, None, None

    # Extract columns systematically
    true_cols = [c for c in df.columns if "_True" in c]
    prob_cols = [c for c in df.columns if "_Prob" in c]
    
    true_cols.sort()
    prob_cols.sort()
    
    y_true = df[true_cols].values
    y_prob = df[prob_cols].values
    class_names = [c.replace("_True", "") for c in true_cols]
    
    # Return Index for tracking specific cases
    indices = df['Index'].values if 'Index' in df.columns else np.arange(len(df))
    
    return y_true, y_prob, class_names, indices

def plot_class_performance_delta(y_true, prob_base, prob_weight, class_names):
    """
    Bar chart showing the GAIN or LOSS in F1 score for each class.
    """
    print("Generating Performance Delta plot...")
    
    f1_base = []
    f1_weight = []
    
    for i in range(len(class_names)):
        pred_base = (prob_base[:, i] > 0.5).astype(int)
        pred_weight = (prob_weight[:, i] > 0.5).astype(int)
        
        f1_base.append(f1_score(y_true[:, i], pred_base, zero_division=0))
        f1_weight.append(f1_score(y_true[:, i], pred_weight, zero_division=0))
        
    diffs = np.array(f1_weight) - np.array(f1_base)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    colors = ['green' if x >= 0 else 'red' for x in diffs]
    sns.barplot(x=class_names, y=diffs, palette=colors)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title("F1 Score Impact by Class (Weighted Model - Baseline)", fontsize=14, fontweight='bold')
    plt.ylabel("Change in F1 Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add labels
    for i, v in enumerate(diffs):
        plt.text(i, v + (0.01 if v > 0 else -0.02), f"{v:+.2f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_performance_delta.png", dpi=300)

def analyze_probability_shifts(prob_base, prob_weight, y_true, class_names):
    """
    Scatter plot comparing probabilities. 
    Shows if the Weighted model is more confident on Rare Classes.
    """
    print("Generating Probability Shift analysis...")
    
    # Focus on Rare Classes (usually indices 5, 7, 8 in ECtHR)
    target_indices = [5, 7, 8] 
    target_names = [class_names[i] for i in target_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, class_idx in enumerate(target_indices):
        ax = axes[idx]
        name = class_names[class_idx]
        
        # Only plot positive cases (Ground Truth = 1)
        # We want to see if the model learned to Identify them
        mask = y_true[:, class_idx] == 1
        p_base = prob_base[mask, class_idx]
        p_weight = prob_weight[mask, class_idx]
        
        ax.scatter(p_base, p_weight, alpha=0.6, color='purple')
        
        # Identity line (y=x)
        ax.plot([0, 1], [0, 1], 'r--', label="No Change")
        
        # Region of Improvement (Upper Left Triangle)
        # Where Weighted Prob > Baseline Prob
        ax.fill_between([0, 1], [0, 1], 1, color='green', alpha=0.1, label="Improvement Zone")
        
        ax.set_title(f"{name} (Positive Cases Only)")
        ax.set_xlabel("Baseline Probability")
        ax.set_ylabel("Weighted Probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if idx == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rare_class_shifts.png", dpi=300)

def generate_flip_report(y_true, prob_base, prob_weight, class_names, indices):
    """
    Identifies specific documents that changed status.
    Win: Wrong -> Right
    Loss: Right -> Wrong
    """
    print("\n" + "="*60)
    print("THE FLIP REPORT: Specific Cases Changed by Retraining")
    print("="*60)
    
    total_wins = 0
    total_losses = 0
    
    for i, name in enumerate(class_names):
        pred_base = (prob_base[:, i] > 0.5).astype(int)
        pred_weight = (prob_weight[:, i] > 0.5).astype(int)
        true = y_true[:, i]
        
        # Wins: Baseline was Wrong, Weighted is Right
        wins_mask = (pred_base != true) & (pred_weight == true)
        wins = indices[wins_mask]
        
        # Losses: Baseline was Right, Weighted is Wrong
        loss_mask = (pred_base == true) & (pred_weight != true)
        losses = indices[loss_mask]
        
        total_wins += len(wins)
        total_losses += len(losses)
        
        if len(wins) > 0 or len(losses) > 0:
            print(f"[{name}]")
            print(f"  Fixed {len(wins)} cases (Wins): {wins[:5]}..." if len(wins) > 0 else "  Fixed 0 cases")
            print(f"  Broke {len(losses)} cases (Regrets): {losses[:5]}..." if len(losses) > 0 else "  Broke 0 cases")
            
            # Deep dive into one 'Win' if available (Why did it switch?)
            if len(wins) > 0:
                idx = np.where(indices == wins[0])[0][0]
                print(f"     Example Win (ID {wins[0]}): True={true[idx]} | Base Prob={prob_base[idx,i]:.3f} -> Wgt Prob={prob_weight[idx,i]:.3f}")
            print("-" * 40)

    print("\nSUMMARY:")
    print(f"Total Corrections (Wins):    {total_wins}")
    print(f"Total Regressions (Losses):  {total_losses}")
    print(f"Net Improvement:             {total_wins - total_losses} predictions")

def main():
    # 1. Load Data
    y_true, prob_base, classes, idx_base = load_and_parse_data(BASELINE_FILE)
    _, prob_weight, _, idx_weight = load_and_parse_data(WEIGHTED_FILE)
    
    if y_true is None or prob_weight is None:
        return

    # Sanity Check
    if not np.array_equal(idx_base, idx_weight):
        print("⚠️ Warning: Index mismatch between files. Comparison might be invalid.")
        return

    # 2. Visuals
    plot_class_performance_delta(y_true, prob_base, prob_weight, classes)
    analyze_probability_shifts(prob_base, prob_weight, y_true, classes)
    
    # 3. Text Report
    generate_flip_report(y_true, prob_base, prob_weight, classes, idx_base)
    
    print(f"\nComparison Complete. Images saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()