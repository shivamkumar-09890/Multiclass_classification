"""
Task 4: Train Fine-tuned Longformer on ECTHR_B (Multi-label classification)
FIXED for RTX 4050 (6GB VRAM) Local Training
"""

import os
# Fix for tokenizer parallelism deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import joblib
import warnings
warnings.filterwarnings('ignore')

def normalize_text(txt):
    """Ensure text is a string"""
    if txt is None:
        return ""
    if isinstance(txt, list):
        return " ".join(str(x) for x in txt)
    return str(txt)

# --- 1. DEFINE CUSTOM COLLATOR ---
class MultiLabelDataCollator(DataCollatorWithPadding):
    """
    Standard DataCollatorWithPadding but forces labels to be float32
    to prevent BCEWithLogitsLoss from crashing with Long tensors.
    """
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            # Force labels to float32 (crucial for Multi-Label BCE Loss)
            batch["labels"] = batch["labels"].float()
        return batch


class ECTHRTrainer:
    def __init__(self, model_name="allenai/longformer-base-4096", max_length=4096):
        print(f"[v0] Initializing trainer with model: {model_name}")
        print(f"[v0] Max Sequence Length: {max_length}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlb = None
        self.num_labels = None
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_and_prepare_data(self):
        print("[v0] Loading ECTHR_B dataset...")
        dataset = load_dataset("lex_glue", "ecthr_b")

        # Fit MultiLabelBinarizer on all labels
        all_labels = [example['labels'] for split in dataset.values() for example in split]
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(all_labels)
        self.num_labels = len(self.mlb.classes_)

        print(f"Number of unique labels: {self.num_labels}")
        return dataset

    def preprocess_dataset(self, dataset):
        """Convert labels to binary vectors"""
        def encode_example(example):
            text = normalize_text(example['text'])
            labels = self.mlb.transform([example['labels']])[0]
            return {"text": text, "labels": labels.astype(np.float32)}

        processed = {}
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                processed[split] = dataset[split].map(encode_example, remove_columns=dataset[split].column_names)
        return processed

    def train_model(self, dataset, output_dir="./ecthr_longformer_finetuned", epochs=3):
        print("\n" + "="*80)
        print("TRAINING FINE-TUNED LONGFORMER MODEL")
        print("="*80)

        def tokenize_and_prepare(examples):
            texts = [normalize_text(t) for t in examples["text"]]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            tokenized["labels"] = examples["labels"]
            return tokenized

        print(f"[v0] Tokenizing with max_length={self.max_length}...")
        train_encoded = dataset["train"].map(tokenize_and_prepare, batched=True, remove_columns=dataset["train"].column_names)
        val_encoded = dataset["validation"].map(tokenize_and_prepare, batched=True, remove_columns=dataset["validation"].column_names)

        train_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        # Explicitly enable gradient checkpointing to be safe
        model.gradient_checkpointing_enable()

        # --- TRAINING ARGUMENTS OPTIMIZED FOR RTX 4050 (6GB) ---
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            
            # --- MEMORY SURVIVAL SETTINGS ---
            per_device_train_batch_size=1,   # MUST BE 1 for 6GB VRAM + 4096 tokens
            gradient_accumulation_steps=16,  # Simulates batch size 16
            per_device_eval_batch_size=1,    # Eval is also heavy, keep it 1
            
            # --- OPTIMIZER ---
            optim="adamw_bnb_8bit",          # REQUIRES bitsandbytes. Saves ~1.5GB VRAM.
            
            # --- HARDWARE ACCELERATION ---
            fp16=True,                       # Tensor Cores
            gradient_checkpointing=True,     # Recomputes gradients to save memory
            
            # --- CPU EFFICIENCY ---
            dataloader_num_workers=4,        # Good for your 8-core CPU
            dataloader_pin_memory=True,
            group_by_length=True,
            
            # --- LOGGING ---
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            learning_rate=2e-5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=1,              # Only keep best model to save disk space
            run_name="longformer-ecthr-b-4096",
            report_to=[]
        )

        data_collator = MultiLabelDataCollator(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=val_encoded,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print("[v0] Starting training...")
        trainer.train()

        print(f"[v0] Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        joblib.dump(self.mlb, f"{output_dir}/mlb.pkl")

        return model, trainer

    def evaluate_model(self, model, dataset, split='test', threshold=0.5):
        print("\n" + "="*80)
        print(f"EVALUATING ON {split.upper()} SET")
        print("="*80)

        # Careful evaluation mapping to avoid OOM
        def tokenize_eval(examples):
            texts = [normalize_text(t) for t in examples["text"]]
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        test_encoded = dataset[split].map(tokenize_eval, batched=True)
        test_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        from torch.utils.data import DataLoader
        # Use batch_size=2 for Eval. If it crashes, change to 1.
        loader = DataLoader(test_encoded, batch_size=2, shuffle=False)

        model.eval()
        all_preds, all_labels = [], []

        print("Running prediction loop...")
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).float()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds > threshold).astype(int)

        from sklearn.metrics import hamming_loss, f1_score, accuracy_score
        
        # Calculate key metrics
        acc = accuracy_score(all_labels, pred_binary)
        f1_mic = f1_score(all_labels, pred_binary, average='micro', zero_division=0)
        f1_mac = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
        
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - F1 Micro: {f1_mic:.4f}")
        print(f"  - F1 Macro: {f1_mac:.4f}")

        return {"accuracy": acc, "f1_micro": f1_mic, "f1_macro": f1_mac}


if __name__ == "__main__":
    # --- FIXED: Use 4096 to match your data ---
    trainer_obj = ECTHRTrainer(model_name="allenai/longformer-base-4096", max_length=4096)
    
    raw_dataset = trainer_obj.load_and_prepare_data()
    processed_dataset = trainer_obj.preprocess_dataset(raw_dataset)

    model, trainer = trainer_obj.train_model(processed_dataset, epochs=3)
    
    # Evaluate
    metrics = trainer_obj.evaluate_model(model, processed_dataset, split='test')

    print("\n" + "="*80)
    print("TRAINING & EVALUATION COMPLETE")
    print("="*80)