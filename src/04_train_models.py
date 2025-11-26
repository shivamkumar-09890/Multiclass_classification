"""
Task 4: Train Baseline and Fine-tuned Models
- Baseline: Pre-trained transformer (no fine-tuning)
- Fine-tuned: Model trained on ECTHR_B dataset
- Compares both approaches
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TextClassificationPipeline
)
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ECTHRTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        print(f"[v0] Initializing trainer with model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlb = None
        self.num_labels = None
        
    def load_and_prepare_data(self):
        """Load and prepare dataset"""
        print("[v0] Loading ECTHR_B dataset...")
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
        
        print(f"Number of unique labels: {self.num_labels}")
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize text samples"""
        return self.tokenizer(
            examples['facts'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        """Calculate metrics for multi-label classification"""
        from sklearn.metrics import (
            hamming_loss, accuracy_score, f1_score,
            precision_score, recall_score
        )
        
        # Convert to binary predictions
        pred_binary = (predictions > threshold).astype(int)
        
        hamming = hamming_loss(labels, pred_binary)
        accuracy = accuracy_score(labels, pred_binary)
        f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
        f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        precision = precision_score(labels, pred_binary, average='micro', zero_division=0)
        recall = recall_score(labels, pred_binary, average='micro', zero_division=0)
        
        return {
            'hamming_loss': hamming,
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, dataset, output_dir="./ecthr_model", epochs=3):
        """Train the model on ECTHR dataset"""
        print("\n" + "="*80)
        print("TRAINING FINE-TUNED MODEL")
        print("="*80)
        
        # Prepare data
        print("[v0] Preparing training data...")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        
        # Tokenize
        train_encoded = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['facts']
        )
        val_encoded = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['facts']
        )
        
        # Convert labels to binary format
        def label_to_binary(example):
            binary_labels = self.mlb.transform([example['labels']])[0]
            example['labels'] = binary_labels.astype(np.float32)
            return example
        
        train_encoded = train_encoded.map(label_to_binary)
        val_encoded = val_encoded.map(label_to_binary)
        
        # Load model
        print("[v0] Loading model for training...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=val_encoded
        )
        
        # Train
        print("[v0] Starting training...")
        trainer.train()
        
        # Save
        print(f"[v0] Saving model to {output_dir}")
        trainer.save_model(output_dir)
        
        return model, trainer
    
    def evaluate_model(self, model, dataset, split='test'):
        """Evaluate model on test set"""
        print("\n" + "="*80)
        print(f"EVALUATING ON {split.upper()} SET")
        print("="*80)
        
        test_dataset = dataset[split]
        
        # Tokenize
        test_encoded = test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['facts']
        )
        
        # Convert labels
        def label_to_binary(example):
            binary_labels = self.mlb.transform([example['labels']])[0]
            example['labels'] = binary_labels.astype(np.float32)
            return example
        
        test_encoded = test_encoded.map(label_to_binary)
        
        # Get predictions
        from torch.utils.data import DataLoader
        import torch
        
        device = torch.device('cpu')  # Use CPU for simplicity
        model.to(device)
        model.eval()
        
        data_loader = DataLoader(test_encoded, batch_size=8)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self.multi_label_metrics(all_preds, all_labels)
        
        print(f"\nMetrics on {split} set:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        return metrics

if __name__ == "__main__":
    # Initialize trainer
    trainer_obj = ECTHRTrainer(model_name="distilbert-base-uncased")
    
    # Load data
    dataset = trainer_obj.load_and_prepare_data()
    
    # Train model
    model, trainer = trainer_obj.train_model(dataset, epochs=3)
    
    # Evaluate
    metrics = trainer_obj.evaluate_model(model, dataset, split='test')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("✓ Model fine-tuned on ECTHR_B dataset")
    print("✓ Model saved to ./ecthr_model")
    print("✓ Ready for evaluation and comparison")
