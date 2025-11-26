# Multi-Label Classification for Legal Documents (ECTHR)

## Assignment Overview

This project implements a multi-label classification system for European Court of Human Rights (ECTHR) legal case documents. The goal is to predict multiple human rights violations that may apply to each case, demonstrating how machine learning can be applied to complex legal domain tasks.

## Key Concepts

- **Multi-Label Classification**: Unlike binary classification where each sample belongs to one class, multi-label classification allows each sample to belong to multiple classes simultaneously. In this case, a legal document can involve multiple human rights violations.
- **Transfer Learning**: We leverage pre-trained language models (DistilBERT) and fine-tune them on the legal domain to improve performance.
- **Baseline vs Fine-tuned Models**: We compare out-of-the-box pre-trained models with domain-specific fine-tuned versions.

## Project Structure

\`\`\`
├── scripts/
│   ├── 01_load_explore_dataset.py      # Load and explore ECTHR_B dataset
│   ├── 02_eda_visualization.py         # Exploratory Data Analysis with visualizations
│   ├── 03_preprocessing.py             # Data cleaning and train/val/test splitting
│   ├── 04_train_models.py              # Train baseline and fine-tuned models
│   ├── 05_compare_models.py            # Evaluate and compare model performance
│   └── 06_documentation.py             # Generate comprehensive report
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
\`\`\`

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Ensure you have sufficient disk space** for model downloads (approximately 1-2 GB)

## Usage

Run the scripts in sequential order:

\`\`\`bash
# 1. Load and explore the dataset
python scripts/01_load_explore_dataset.py

# 2. Perform EDA and generate visualizations
python scripts/02_eda_visualization.py

# 3. Preprocess data and create splits
python scripts/03_preprocessing.py

# 4. Train baseline and fine-tuned models
python scripts/04_train_models.py

# 5. Compare model performance
python scripts/05_compare_models.py

# 6. Generate final documentation report
python scripts/06_documentation.py
\`\`\`

## Expected Outputs

Each script generates outputs in the console and optionally saves results:

- **Script 1-2**: Dataset statistics, label distribution charts, text length analysis
- **Script 3**: Data split information, preprocessing statistics
- **Script 4**: Training logs, model checkpoints, loss curves
- **Script 5**: Performance metrics (F1-scores, Hamming Loss, ROC-AUC), comparison tables
- **Script 6**: Comprehensive report with findings, challenges, and recommendations

## Dataset

**ECTHR_B (European Court of Human Rights - Binary Multi-label)**
- Contains legal cases from the European Court of Human Rights
- Each document can be labeled with multiple human rights violations
- Labels represent different articles of the European Convention on Human Rights
- Multi-label setup means documents often have 2-5 violations per case

## Models

### Baseline Model
- Pre-trained DistilBERT (no fine-tuning)
- Tests the zero-shot capability on legal domain
- Serves as comparison point

### Fine-tuned Model
- DistilBERT fine-tuned on ECTHR training set
- Adapts the model to legal domain language and structure
- Should demonstrate improved performance

## Evaluation Metrics

For multi-label classification:
- **Hamming Loss**: Fraction of labels that are incorrectly predicted
- **F1-Score (Macro)**: Average F1 across all labels (treats each label equally)
- **F1-Score (Micro)**: Calculates metrics globally by counting total TP, FN, FP
- **Subset Accuracy**: Percentage of samples with all labels predicted correctly
- **ROC-AUC**: Area under receiver operating characteristic curve per label

## Key Findings

After running the complete pipeline, you'll discover:
- How many labels are typically associated with legal cases
- Which violations are most common
- Performance gaps between baseline and fine-tuned models
- Domain-specific challenges in legal text classification
- Recommendations for further improvement

## Challenges in Legal Document Classification

1. **Class Imbalance**: Some human rights violations are much rarer than others
2. **Long Documents**: Legal cases can be very long, requiring careful handling
3. **Complex Language**: Legal terminology and phrasing differs from general text
4. **Multiple Valid Labels**: Legal complexity means documents genuinely have multiple violations
5. **Limited Training Data**: Domain-specific datasets are smaller than general NLP datasets

## Requirements

- Python 3.8+
- PyTorch (for transformer models)
- Hugging Face Transformers & Datasets
- Scikit-learn (for evaluation metrics)
- Pandas & NumPy (for data handling)
- Matplotlib & Seaborn (for visualizations)

See `requirements.txt` for specific versions.

## Notes

- First run will download pre-trained models (1-2 GB)
- Training may take time depending on your hardware (GPU recommended)
- Results are reproducible with fixed random seeds
- Adjust batch size and learning rate in scripts if running on limited hardware

## Further Improvements

- Experiment with different pre-trained models (RoBERTa, LEGAL-BERT)
- Implement ensemble methods combining multiple models
- Use hierarchical classification for label relationships
- Apply active learning to improve on hard examples
- Fine-tune on domain-specific legal documents

---

**Assignment Status**: Complete with full pipeline implementation and analysis
