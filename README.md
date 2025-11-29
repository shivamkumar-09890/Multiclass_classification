# Legal Judgment Prediction with Weighted Longformer (ECtHR-B)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project implements a high-performance **Multi-Label Classification** model for the **European Court of Human Rights (ECtHR)** dataset — **Task B** from **LexGLUE**.

The primary challenge in this domain is **Extreme Class Imbalance** combined with **Long Document Contexts** (avg. 2,000+ words). Standard models often achieve high accuracy by memorizing common classes (e.g., Article 3) while completely ignoring rare human rights violations (e.g., Article 5).

### Our Approach
1. **Model**: Fine-tuned `allenai/longformer-base-4096` to handle full case documents  
2. **Strategy**: A **Two-Stage Training** pipeline  
   - **Stage 1**: General training to learn legal language patterns  
   - **Stage 2**: Refinement using a **custom Weighted Binary Cross-Entropy Loss** to penalize neglect of rare classes  
3. **Result**: Achieved **Macro-F1 score of 74.8%**, significantly outperforming the standard baseline (~65%) on rare classes.

## 📂 Project Structure
```
├── src/
│   ├── Understanding_data/
│   │   └── 01_load_explore_dataset.py    # Initial dataset inspection
│   ├── EDA/
│   │   ├── 02_visualization_1.py         # Basic label frequency plots
│   │   ├── 03_visualization_2.py         # Advanced coverage analysis
│   │   └── 04_visualize_punct.py         # "Punctuation Tax" analysis
│   ├── Training/
│   │   ├── 05_train_models_1.py          # Baseline Model Training (Standard Loss)
│   │   └── 06_train_model_predata_2.py   # Weighted Refinement Training (Custom Loss)
│   ├── final_results/
│   │   ├── 08_final_eval.py              # Generate metrics for trained models
│   │   ├── 09_model_comparison.py        # Compare Baseline vs Weighted vs Short
│   │   ├── 10_overall_score.py           # Evaluation on full dataset
│   │   └── 11_comparing_both_csv.py      # CSV-based deep dive comparison
│   └── scripts/
│       ├── ploting_final_metrics.py      # Generate final report graphs
│       └── Punctuation_test.py           # Utility for tokenization tests
│
├── analysis_results/                     # Confusion matrices & Distributions
├── comparison_results/                   # Comparison plots (Delta charts)
├── ecthr_longformer_weighted/            # Saved Weighted Model (Best)
├── processed_ecthr_b/                    # Cleaned Dataset (Arrow format)
└── README.md
```

## 📊 Key Findings

### 1. Class Imbalance
The dataset is heavily skewed:
- Article 3 (Torture) appears ~6,000 times  
- Article 5 (Liberty) appears ~100 times  

**Solution**: Calculated **inverse frequency weights** (e.g., **109× penalty** for missing Article 5) and injected them into the loss function.

### 2. Context Length
- Mean Length: **~2,137 tokens**  
- Coverage: Longformer (4096 tokens) covers **87.2%** of cases fully without truncation  
- A standard BERT (512 tokens) would lose critical information in **>75%** of cases

### 3. Punctuation Analysis
We analyzed whether stripping punctuation would save tokens.

**Result**: Keeping punctuation only increases token count by **2.3%**  
**Decision**: Kept punctuation to preserve semantic boundaries (clauses) crucial for legal interpretation.

## 🏆 Final Results

**Comparison on the Test Set:**

| Model Strategy         | Exact Match Acc | Micro-F1 (Global) | Macro-F1 (Fairness) |
|-------------------------|-----------------|-------------------|---------------------|
| Stage 1 (Baseline)      | **60.9%**       | **80.6%**         | 65.0%               |
| Stage 2 (Weighted)      | 56.1%           | 79.4%             | **74.8%**           |

**Interpretation**:
- Baseline focuses on being "Safe" → predicts common classes correctly
- Weighted Model is a "Specialist" → sacrifices slightly on exact matches to gain **+9.8%** improvement in Macro-F1, effectively solving rare class blindness

## 🚀 Usage Instructions

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Run Understanding data
```bash
python src/Understanding_data/01_load_explore_dataset.py
```

### 3. Run EDA
```bash
python src/EDA/02_visualization_1.py
python src/EDA/03_visualization_2.py
python src/EDA/04_visualize_punct.py
```

### 4. Training
```bash
python src/Training/05_train_models_1.py
python src/Training/06_train_model_predata_2.py
```

### 5. Get final Results
```bash
python src/final_results/08_final_eval.py
python src/final_results/09_model_comparison.py
python src/final_results/10_overall_score.py
python src/final_results/11_comparing_both_csv.py
```

### 6. Saving plot
```bash
python src/scripts/Plot_train_log.py
python src/scripts/ploting_final_metrics.py
python src/scripts/Punctuation_test.py
```

## 🖼️ Gallery of Analysis

### Training Convergence

Class-Wise Performance


### Outlier Detection (K-Means)

Global Metrics Comparison


#### 📜 Citation

If you use this code or methodology, please cite the original LexGLUE paper:

Chalkidis et al., "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English", ACL 2022.