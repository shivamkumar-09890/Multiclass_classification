"""
Task 5: Retrain Longformer with Weighted Loss on Processed Data
FIXED: compute_loss signature update for latest Transformers version
"""

import os
# Fix for tokenizer parallelism deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import joblib
import warnings
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn

warnings.filterwarnings('ignore')

# --- 1. CUSTOM TRAINER FOR WEIGHTED LOSS (FIXED) ---
class WeightedTrainer(Trainer):
    """
    Custom Trainer that injects Class Weights into the BCE Loss function.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # --- FIX IS HERE: Added **kwargs to accept num_items_in_batch ---
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Move weights to GPU if needed
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        # Weighted BCE Loss
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# --- 2. DATA COLLATOR ---
class MultiLabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()
        return batch

class ECTHRWeightedTrainer:
    def __init__(self, model_path, max_length=4096):
        print(f"[v0] Initializing Trainer with weights from: {model_path}")
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.mlb = None
        self.num_labels = None
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = None

    def load_local_data(self, data_path="processed_ecthr_b"):
        print(f"[v0] Loading local processed data from '{data_path}'...")
        try:
            dataset = load_from_disk(data_path)
            
            print("[v0] Fitting Label Binarizer on new data...")
            all_labels = [l for split in dataset.values() for l in split['labels']]
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(all_labels)
            self.num_labels = len(self.mlb.classes_)
            print(f"   > Found {self.num_labels} unique labels: {self.mlb.classes_}")
            
            return dataset
        except Exception as e:
            print(f"Error loading local data: {e}")
            exit(1)

    def calculate_class_weights(self, dataset):
        print("[v0] Calculating class weights...")
        
        train_labels = [item['labels'] for item in dataset['train']]
        binary_matrix = self.mlb.transform(train_labels)
        
        pos_counts = np.sum(binary_matrix, axis=0)
        total_samples = len(binary_matrix)
        
        neg_counts = total_samples - pos_counts
        # Balanced Weight Formula
        weights = neg_counts / (pos_counts + 1e-5)
        
        print("   > Class Weights (Impact Factor):")
        for cls, w in zip(self.mlb.classes_, weights):
            print(f"     - {cls}: {w:.2f}x")
            
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        return self.class_weights

    def train_model(self, dataset, output_dir="./ecthr_longformer_weighted", epochs=2):
        print("\n" + "="*60)
        print("STARTING RETRAINING")
        print("="*60)

        def tokenize_and_encode(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            bin_labels = self.mlb.transform(examples['labels'])
            tokenized['labels'] = bin_labels.astype(np.float32)
            return tokenized

        print("[v0] Tokenizing and Encoding...")
        cols_to_remove = dataset['train'].column_names 
        train_final = dataset['train'].map(tokenize_and_encode, batched=True, remove_columns=cols_to_remove)
        val_final = dataset['validation'].map(tokenize_and_encode, batched=True, remove_columns=cols_to_remove)

        train_final.set_format(type="torch")
        val_final.set_format(type="torch")

        print(f"[v0] Loading weights from {self.model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=False 
        )
        model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            per_device_eval_batch_size=1,
            optim="adamw_bnb_8bit",
            fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            learning_rate=2e-5,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=1
        )

        trainer = WeightedTrainer(
            class_weights=self.class_weights,
            model=model,
            args=training_args,
            train_dataset=train_final,
            eval_dataset=val_final,
            tokenizer=self.tokenizer,
            data_collator=MultiLabelDataCollator(self.tokenizer)
        )

        trainer.train()
        
        print(f"[v0] Saving new weighted model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        joblib.dump(self.mlb, f"{output_dir}/mlb.pkl")
        
        return model

    def evaluate_model(self, model, dataset, split='test'):
        print(f"\nEvaluating on {split}...")
        
        def prepare_test(examples):
            tokenized = self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=self.max_length
            )
            bin_labels = self.mlb.transform(examples['labels'])
            tokenized['labels'] = bin_labels.astype(np.float32)
            return tokenized

        test_final = dataset[split].map(prepare_test, batched=True, remove_columns=dataset[split].column_names)
        test_final.set_format(type="torch")

        from torch.utils.data import DataLoader
        loader = DataLoader(test_final, batch_size=2)
        
        model.eval()
        model.to(self.device)
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask=mask)
                probs = torch.sigmoid(outputs.logits)
                
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        from sklearn.metrics import f1_score, accuracy_score
        pred_binary = (all_preds > 0.5).astype(int)
        
        acc = accuracy_score(all_labels, pred_binary)
        f1_mic = f1_score(all_labels, pred_binary, average='micro', zero_division=0)
        f1_mac = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
        
        print(f"\nRESULTS ON {split.upper()} (Processed Data + Weighted Loss):")
        print(f"  Exact Match Accuracy: {acc:.4f}")
        print(f"  F1 Micro:             {f1_mic:.4f}")
        print(f"  F1 Macro:             {f1_mac:.4f}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Path to the folder containing your PREVIOUSLY trained model (the one with 80.6% F1)
    # Use the main folder, not the checkpoint folder
    PREVIOUS_MODEL_PATH = "./ecthr_longformer_finetuned"
    
    print(f"Resuming training from: {PREVIOUS_MODEL_PATH}")
    
    trainer = ECTHRWeightedTrainer(model_path=PREVIOUS_MODEL_PATH)
    
    # 1. Load your CLEANED data
    dataset = trainer.load_local_data("processed_ecthr_b")
    
    # 2. Calculate Weights
    trainer.calculate_class_weights(dataset)
    
    # 3. Retrain
    model = trainer.train_model(dataset, epochs=2)
    
    # 4. Evaluate
    trainer.evaluate_model(model, dataset)