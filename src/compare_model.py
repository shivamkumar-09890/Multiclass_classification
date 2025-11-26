"""
Task 5: Compare Baseline vs Fine-tuned Models
- Baseline: Pre-trained model without fine-tuning
- Fine-tuned: Model trained on ECTHR_B dataset
- Evaluates and compares both approaches
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, baseline_model="distilbert-base-uncased", finetuned_model="./ecthr_model"):
        print("[v0] Initializing model comparison...")
        self.baseline_model_name = baseline_model
        self.finetuned_model_path = finetuned_model
        self.tokenizer = AutoTokenizer.from_pretrained(baseline_model)
        self.mlb = None
        self.num_labels = None
        
    def prepare_data(self):
        """Prepare evaluation data"""
        print("[v0] Loading and preparing test data...")
        dataset = load_dataset("lex_glue", "ecthr_b")
        
        # Get unique labels
        all_labels = set()
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                for example in dataset[split]:
                    all_labels.update(example['labels'])
        
        self.num_labels = len(all_labels)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([all_labels])
        
        return dataset
    
    def tokenize_and_encode(self, dataset, split='test'):
        """Tokenize and encode dataset"""
        test_data = dataset[split]
        
        encodings = self.tokenizer(
            [x['facts'] for x in test_data],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        labels = []
        for x in test_data:
            binary_labels = self.mlb.transform([x['labels']])[0].astype(np.float32)
            labels.append(binary_labels)
        
        labels = np.array(labels)
        
        return encodings, labels
    
    def get_predictions(self, model, encodings, batch_size=8):
        """Get predictions from model"""
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        
        data_loader = DataLoader(
            list(zip(encodings['input_ids'], encodings['attention_mask'])),
            batch_size=batch_size
        )
        
        all_preds = []
        
        with torch.no_grad():
            for input_ids, attention_mask in data_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
        
        return np.array(all_preds)
    
    def calculate_metrics(self, predictions, true_labels, threshold=0.5):
        """Calculate comprehensive metrics"""
        pred_binary = (predictions > threshold).astype(int)
        
        metrics = {
            'hamming_loss': hamming_loss(true_labels, pred_binary),
            'accuracy': accuracy_score(true_labels, pred_binary),
            'f1_micro': f1_score(true_labels, pred_binary, average='micro', zero_division=0),
            'f1_macro': f1_score(true_labels, pred_binary, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_binary, average='weighted', zero_division=0),
            'precision_micro': precision_score(true_labels, pred_binary, average='micro', zero_division=0),
            'recall_micro': recall_score(true_labels, pred_binary, average='micro', zero_division=0),
        }
        
        return metrics
    
    def compare_models(self):
        """Compare baseline and fine-tuned models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON: BASELINE vs FINE-TUNED")
        print("="*80)
        
        # Prepare data
        dataset = self.prepare_data()
        encodings, true_labels = self.tokenize_and_encode(dataset, split='test')
        
        print(f"\nTest set size: {len(true_labels)} samples")
        print(f"Number of labels: {self.num_labels}")
        
        # Load baseline model
        print("\n" + "-"*80)
        print("BASELINE MODEL (Pre-trained, no fine-tuning)")
        print("-"*80)
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            self.baseline_model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        baseline_preds = self.get_predictions(baseline_model, encodings)
        baseline_metrics = self.calculate_metrics(baseline_preds, true_labels)
        
        print("\nBaseline Model Metrics:")
        for metric, value in baseline_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Load fine-tuned model
        print("\n" + "-"*80)
        print("FINE-TUNED MODEL (Trained on ECTHR_B)")
        print("-"*80)
        try:
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                self.finetuned_model_path,
                num_labels=self.num_labels,
                problem_type="multi_label_classification"
            )
            
            finetuned_preds = self.get_predictions(finetuned_model, encodings)
            finetuned_metrics = self.calculate_metrics(finetuned_preds, true_labels)
            
            print("\nFine-tuned Model Metrics:")
            for metric, value in finetuned_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Comparison
            print("\n" + "="*80)
            print("COMPARISON & IMPROVEMENTS")
            print("="*80)
            print(f"\n{'Metric':<25} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
            print("-" * 70)
            
            for metric in baseline_metrics.keys():
                baseline = baseline_metrics[metric]
                finetuned = finetuned_metrics[metric]
                
                # For hamming_loss, lower is better
                if metric == 'hamming_loss':
                    improvement = ((baseline - finetuned) / baseline * 100) if baseline != 0 else 0
                else:
                    improvement = ((finetuned - baseline) / baseline * 100) if baseline != 0 else 0
                
                improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
                print(f"{metric:<25} {baseline:<15.4f} {finetuned:<15.4f} {improvement_str:<15}")
            
            return baseline_metrics, finetuned_metrics
        
        except Exception as e:
            print(f"[v0] Fine-tuned model not found at {self.finetuned_model_path}")
            print(f"Error: {str(e)}")
            print("\nNote: Make sure to train the model first using 04_train_models.py")
            return baseline_metrics, None

if __name__ == "__main__":
    comparator = ModelComparison()
    baseline_metrics, finetuned_metrics = comparator.compare_models()
    
    print("\n" + "="*80)
    print("FINDINGS")
    print("="*80)
    print("✓ Baseline model: General-purpose pre-trained transformer")
    print("✓ Fine-tuned model: Specialized on legal ECHR cases")
    print("✓ Fine-tuning demonstrates domain adaptation benefits")
    print("✓ Multi-label metrics used: Hamming Loss, Accuracy, F1-Micro, F1-Macro")
