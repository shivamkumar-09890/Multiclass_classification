import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

MODEL_PATH = "./ecthr_longformer_finetuned/checkpoint-1689"


def plot_training_results(model_path):

    # ---- Load trainer_state.json ----
    log_file = os.path.join(model_path, "trainer_state.json")
    if not os.path.exists(log_file):
        print(f"❌ trainer_state.json not found in {model_path}")
        return

    print(f"✓ Found log file: {log_file}")

    with open(log_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["log_history"])

    # Convert numeric columns safely
    for col in ["loss", "eval_loss", "eval_accuracy", "eval_f1_micro", "eval_f1_macro"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df = df[df["loss"].notna()]
    val_df = df[df["eval_loss"].notna()]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # ============================================================
    # 1️⃣ Plot Loss Curves
    # ============================================================
    ax1 = axes[0]

    sns.lineplot(
        data=train_df,
        x="step",
        y="loss",
        color="#1f77b4",
        label="Training Loss",
        ax=ax1
    )

    if not val_df.empty:
        plt.scatter(
            val_df["step"],
            val_df["eval_loss"],
            s=40,
            color="#d62728",
            label="Validation Loss"
        )

    ax1.set_title("Training vs Validation Loss", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # ============================================================
    # 2️⃣ Plot Validation Metrics
    # ============================================================
    ax2 = axes[1]

    if not val_df.empty:

        metric_map = {
            "eval_f1_micro": "F1 Micro",
            "eval_f1_macro": "F1 Macro",
            "eval_accuracy": "Exact Accuracy",
        }

        for metric, label in metric_map.items():
            if metric in val_df.columns:
                sns.lineplot(
                    data=val_df,
                    x="step",
                    y=metric,
                    label=label,
                    ax=ax2
                )

        ax2.set_title("Validation Metrics", fontsize=15, fontweight="bold")
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Metric Value")
        ax2.set_ylim(0, 1)
        ax2.legend()

    else:
        ax2.text(0.5, 0.5, "No validation logs found", ha="center")

    plt.tight_layout()
    plt.savefig("training_analysis_plots_2.png", dpi=300)
    print("\n✓ Saved: training_analysis_plots.png")


if __name__ == "__main__":
    plot_training_results(MODEL_PATH)
