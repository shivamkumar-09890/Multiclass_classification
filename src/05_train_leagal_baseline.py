"""
Task 5a: Train Legal-BERT Baseline (Linear Probe)
FIX: Added MultiLabelDataCollator to force labels to Float32
"""

import os
import torch
import numpy as np
import joblib
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.preprocessing import MultiLabelBinarizer

# --- CONFIG ---
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512  # Legal-BERT limit
OUTPUT_DIR = "./legal_bert_baseline"

# --- THE MISSING PIECE: CUSTOM COLLATOR ---
class MultiLabelDataCollator(DataCollatorWithPadding):
    """Force labels to float32 to fix the RuntimeError"""
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()
        return batch

class LegalBaselineTrainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.mlb = MultiLabelBinarizer()
        
    def train(self):
        print(f"[Baseline] Loading Legal-BERT: {MODEL_NAME}...")
        dataset = load_dataset("lex_glue", "ecthr_b")
        
        # 1. Prepare Labels
        if os.path.exists("./ecthr_longformer_finetuned/mlb.pkl"):
            print("[Baseline] Loading Label Encoder from Longformer...")
            self.mlb = joblib.load("./ecthr_longformer_finetuned/mlb.pkl")
        else:
            print("[Baseline] Creating new Label Encoder...")
            all_labels = [ex['labels'] for split in dataset.values() for ex in split]
            self.mlb.fit(all_labels)

        # 2. Preprocess
        def encode(example):
            text = example['text']
            if isinstance(text, list): text = " ".join(str(x) for x in text)
            # We convert to float here, but the Collator is the safety net
            return {
                "text": text,
                "labels": self.mlb.transform([example['labels']])[0].astype(np.float32)
            }
            
        print("[Baseline] Tokenizing...")
        dataset = {k: v.map(encode) for k, v in dataset.items()}
        
        def tokenize(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
            )
            
        encoded = {k: v.map(tokenize, batched=True) for k, v in dataset.items()}
        
        # Set format strictly
        for k in encoded: 
            encoded[k].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # 3. Load Model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(self.mlb.classes_),
            problem_type="multi_label_classification"
        )
        
        # Freeze Body (Linear Probe)
        for param in model.bert.parameters():
            param.requires_grad = False
            
        print("[Baseline] Encoder Frozen. Training Classification Head only...")

        # 4. Training Args
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=1e-3,
            fp16=True,
            save_strategy="no",
            eval_strategy="epoch",
            logging_steps=50,
        )

        # --- FIX: USE THE CUSTOM COLLATOR ---
        data_collator = MultiLabelDataCollator(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model, args=args,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator  # <--- THIS WAS MISSING
        )
        
        trainer.train()
        
        print("[Baseline] Saving...")
        model.save_pretrained(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        print("[Baseline] Done!")

if __name__ == "__main__":
    LegalBaselineTrainer().train()