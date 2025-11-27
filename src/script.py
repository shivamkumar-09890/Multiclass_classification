import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
import bitsandbytes as bnb
import gc

# --- CONFIGURATION ---
MODEL_NAME = "allenai/longformer-base-4096"
SEQ_LEN = 4096       # The target length
BATCH_SIZE = 1       # The target batch size
NUM_LABELS = 10      # Dummy number of labels
USE_FP16 = True      # Enable Mixed Precision
USE_GRAD_CHECKPOINTING = True # Enable Memory Saving

def print_memory(stage):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{stage}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def run_stress_test():
    if not torch.cuda.is_available():
        print("❌ No GPU found! This test requires CUDA.")
        return

    print("="*40)
    print(f"⚡ VRAM STRESS TEST: Longformer-4096")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   FP16: {USE_FP16}")
    print(f"   Gradient Checkpointing: {USE_GRAD_CHECKPOINTING}")
    print("="*40)

    try:
        # 1. Load Model (Weights take VRAM)
        print("\n1. Loading Model...")
        config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        
        # We load directly to CUDA to measure impact immediately
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).cuda()
        
        if USE_GRAD_CHECKPOINTING:
            model.gradient_checkpointing_enable()
        
        print_memory("Model Loaded")

        # 2. Setup Optimizer (8-bit Adam)
        print("\n2. Initializing 8-bit Optimizer...")
        # Simulating the exact optimizer you'd use
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)
        print_memory("Optimizer Init")

        # 3. Create Dummy Batch (Sequence Length 4096)
        print(f"\n3. Creating Dummy Batch (Seq Len {SEQ_LEN})...")
        input_ids = torch.randint(0, 50265, (BATCH_SIZE, SEQ_LEN)).cuda()
        attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN)).cuda()
        labels = torch.randint(0, 1, (BATCH_SIZE, NUM_LABELS)).float().cuda()
        
        print_memory("Data Loaded")

        # 4. Forward Pass (Calculates Activations)
        print("\n4. Running Forward Pass...")
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        print_memory("After Forward")

        # 5. Backward Pass (Calculates Gradients - THE PEAK USAGE)
        print("\n5. Running Backward Pass (The Real Test)...")
        loss.backward()
        
        print_memory("After Backward")

        # 6. Optimizer Step
        print("\n6. Optimizer Step...")
        optimizer.step()
        optimizer.zero_grad()
        
        print_memory("After Step")

        print("\n" + "="*40)
        print(f"✅ SUCCESS! Peak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print("This configuration FITS on your GPU.")
        print("="*40)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "!"*40)
            print("❌ CRASH: CUDA OUT OF MEMORY")
            print("!"*40)
            print("Your GPU does not have enough VRAM for this configuration.")
            print(f"Crashed trying to allocate memory. Peak before crash: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        else:
            print(f"\n❌ Unexpected Error: {e}")
    finally:
        # Cleanup
        del model, optimizer, input_ids, labels
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_stress_test()