import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# --- Hyperparameters ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME_OR_PATH = os.path.join(SCRIPT_DIR, "..", "models", "gemma3_270m")
TRAIN_FILE = os.path.join(SCRIPT_DIR, "..", "datasets", "clinical_ner_text2text.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models_results", "gemma-wanda-finetuned")

SPARSITY_RATIO = 0.5
CALIBRATION_SAMPLES = 128
TOP_K_LAYERS_RATIO = 0.2
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5

# --- Utilities ---

def get_layers(model):
    """Get the Transformer layers from the model."""
    if hasattr(model, "model"):
        return model.model.layers
    return model.layers

def get_linear_layers(module):
    """Recursively find all linear layers in a module."""
    linears = {}
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            linears[name] = m
    return linears

# --- Wanda Pruning Implementation ---

@torch.no_grad()
def apply_wanda_pruning(model, tokenizer, calibration_dataset, sparsity=0.5):
    """
    Apply Wanda pruning: |W| * ||X||_2
    """
    print(f"Applying Wanda pruning with {sparsity} sparsity...")
    model.eval()
    layers = get_layers(model)
    device = next(model.parameters()).device
    
    # Hooks to collect activations
    ins = {}
    
    def get_hook(name):
        def hook(module, input, output):
            # We want the L2 norm of activations. 
            # X shape: [batch, sequence, hidden]
            x = input[0].float()
            if name not in ins:
                ins[name] = torch.zeros(x.shape[-1], device=device)
            # Update running sum of squares for L2 norm calculation
            # Formula: ||X||_2^2 over samples and sequence
            ins[name] += (x.reshape(-1, x.shape[-1])**2).sum(dim=0)
        return hook

    # Process calibration samples
    print("Collecting activations...")
    for i in tqdm(range(len(calibration_dataset))):
        batch = calibration_dataset[i]
        input_ids = tokenizer(batch["input"], return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
        
        # We hook layer by layer to avoid OOM or we can hook all linear layers
        # For simplicity, let's find all linear layers in all transformer blocks
        hooks = []
        for layer_idx, layer in enumerate(layers):
            linears = get_linear_layers(layer)
            for name, m in linears.items():
                full_name = f"layers.{layer_idx}.{name}"
                hooks.append(m.register_forward_hook(get_hook(full_name)))
        
        model(input_ids)
        
        for h in hooks:
            h.remove()

    # Apply pruning
    print("Masking weights...")
    masks = {}
    degradations = {} # To store degradation per layer
    
    for layer_idx, layer in enumerate(layers):
        linears = get_linear_layers(layer)
        layer_total_degradation = 0
        
        for name, m in linears.items():
            full_name = f"layers.{layer_idx}.{name}"
            if full_name not in ins:
                continue
                
            W = m.weight.data
            X_norm = torch.sqrt(ins[full_name] / (len(calibration_dataset) * 512)) # Approximate normalization
            
            # Wanda score: |W| * ||X||_2
            # Broadcase X_norm to W shape
            score = torch.abs(W) * X_norm.reshape(1, -1)
            
            # Find threshold for sparsity
            threshold = torch.quantile(score.float(), sparsity)
            mask = score >= threshold
            
            # Calculate degradation: norm of removed weights
            removed_weights = W * (~mask)
            layer_total_degradation += torch.norm(removed_weights).item()
            
            # Apply mask
            W.mul_(mask)
            masks[full_name] = mask
            
        degradations[layer_idx] = layer_total_degradation

    return masks, degradations

# --- Fine-Tuning Pipeline ---

class SparseFineTuningCallback(TrainerCallback):
    """Ensure pruned weights remain zero during training."""
    def __init__(self, masks):
        self.masks = masks

    def on_step_end(self, args, state, control, model=None, **kwargs):
        with torch.no_grad():
            layers = get_layers(model)
            for layer_idx, layer in enumerate(layers):
                linears = get_linear_layers(layer)
                for name, m in linears.items():
                    full_name = f"layers.{layer_idx}.{name}"
                    if full_name in self.masks:
                        m.weight.data.mul_(self.masks[full_name].to(m.weight.device))

def main():
    if not os.path.exists(MODEL_NAME_OR_PATH):
        print(f"Error: Model path {MODEL_NAME_OR_PATH} not found.")
        return

    print(f"Loading model and tokenizer from {MODEL_NAME_OR_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 1. Calibration Dataset
    print(f"Loading calibration dataset from {TRAIN_FILE}...")
    dataset_full = load_dataset("json", data_files=TRAIN_FILE, split="train")
    # Take a subset for calibration. If dataset is smaller than CALIBRATION_SAMPLES, use full.
    num_calib = min(len(dataset_full), CALIBRATION_SAMPLES)
    calib_subset = dataset_full.select(range(num_calib))

    # 2. Wanda Pruning
    masks, degradations = apply_wanda_pruning(model, tokenizer, calib_subset, sparsity=SPARSITY_RATIO)

    # 3. Identify Top-K affected layers
    sorted_layers = sorted(degradations.items(), key=lambda x: x[1], reverse=True)
    num_to_unfreeze = int(len(get_layers(model)) * TOP_K_LAYERS_RATIO)
    top_k_indices = [idx for idx, deg in sorted_layers[:num_to_unfreeze]]
    
    print(f"Ranking layers by degradation: {sorted_layers}")
    print(f"Unfreezing top {num_to_unfreeze} layers: {top_k_indices}")

    # 4. Selective Fine-Tuning
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze top-K
    layers = get_layers(model)
    for idx in top_k_indices:
        for param in layers[idx].parameters():
            param.requires_grad = True

    # Load target dataset
    print(f"Loading fine-tuning dataset from {TRAIN_FILE}...")
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    def tokenize_function(examples):
        texts = [f"Input: {i}\nOutput: {o}" for i, o in zip(examples["input"], examples["output"])]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[SparseFineTuningCallback(masks)]
    )

    print("Starting sparse selective fine-tuning...")
    trainer.train()

    print(f"Saving sparse fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
