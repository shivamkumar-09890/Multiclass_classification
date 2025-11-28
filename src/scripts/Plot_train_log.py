import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# --- CONFIGURATION ---
# UPDATE THIS PATH to your saved model folder
# Example: "./ecthr_longformer_weighted"
MODEL_PATH = "./ecthr_longformer_weighted/checkpoint-559" 

def plot_training_results(model_path):
    # 1. Locate the log file
    log_file = os.path.join(model_path, "trainer_state.json")
    
    if not os.path.exists(log_file):
        print(f"❌ Error: Could not find 'trainer_state.json' in {model_path}")
        print("   Did you delete the folder? This file is generated automatically by Trainer.")
        return

    print(f"✓ Found training history: {log_file}")
    
    # 2. Load the data
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    history = data['log_history']
    df = pd.DataFrame(history)
    
    # 3. Separate Training vs Validation Data
    # Training logs have 'loss' but no 'eval_loss'
    # Validation logs have 'eval_loss'
    train_df = df.dropna(subset=['loss'])
    val_df = df.dropna(subset=['eval_loss'])

    if train_df.empty:
        print("⚠️ Warning: No training loss found in logs.")
        return

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # --- GRAPH 1: Loss Convergence (Train vs Val) ---
    ax1 = axes[0]
    
    # Plot Training Loss (Blue line)
    sns.lineplot(
        data=train_df, x='step', y='loss', 
        label='Training Loss', color='#1f77b4', alpha=0.6, ax=ax1
    )
    
    # Plot Validation Loss (Red dots) - Happens less frequently
    if not val_df.empty:
        sns.lineplot(
            data=val_df, x='step', y='eval_loss', 
            label='Validation Loss', color='#d62728', marker='o', markersize=8, linewidth=2, ax=ax1
        )
    
    ax1.set_title("Training Convergence: Loss over Steps", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- GRAPH 2: Validation Metrics (F1 & Accuracy) ---
    ax2 = axes[1]
    
    if not val_df.empty:
        # Check which metrics are available (Hugging Face names them 'eval_X')
        metrics_to_plot = []
        if 'eval_f1_micro' in val_df.columns: metrics_to_plot.append(('eval_f1_micro', 'F1 Micro', 'green', 's'))
        if 'eval_f1_macro' in val_df.columns: metrics_to_plot.append(('eval_f1_macro', 'F1 Macro', 'orange', '^'))
        if 'eval_accuracy' in val_df.columns: metrics_to_plot.append(('eval_accuracy', 'Exact Match Acc', 'purple', 'D'))

        for metric_col, label, color, marker in metrics_to_plot:
            sns.lineplot(
                data=val_df, x='epoch', y=metric_col, 
                label=label, color=color, marker=marker, markersize=8, linewidth=2, ax=ax2
            )
        
        ax2.set_title("Validation Performance Evolution", fontsize=15, fontweight='bold')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Score (0.0 - 1.0)")
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 1.0) # Metrics are usually 0-1
    else:
        ax2.text(0.5, 0.5, "No Validation Data Available", ha='center', fontsize=12)
        ax2.set_title("Validation Metrics (Empty)")

    plt.tight_layout()
    
    # Save the plot
    output_img = "training_analysis_plots.png"
    plt.savefig(output_img, dpi=300)
    print(f"✓ Graphs saved to: {output_img}")
    
    # Print final stats to console
    if not val_df.empty:
        final_run = val_df.iloc[-1]
        print("\n" + "="*40)
        print(f"FINAL MODEL PERFORMANCE (Epoch {final_run['epoch']})")
        print("="*40)
        print(f"Validation Loss: {final_run['eval_loss']:.4f}")
        if 'eval_f1_micro' in final_run: print(f"F1 Micro:        {final_run['eval_f1_micro']:.4f}")
        if 'eval_f1_macro' in final_run: print(f"F1 Macro:        {final_run['eval_f1_macro']:.4f}")
        print("="*40)

if __name__ == "__main__":
    plot_training_results(MODEL_PATH)