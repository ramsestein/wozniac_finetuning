import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# --- Hyperparameters ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative to this script (wozniac/training_gemma_lora/)
MODEL_NAME_OR_PATH = os.path.join(SCRIPT_DIR, "..", "models", "gemma3_270m")
TRAIN_FILE = os.path.join(SCRIPT_DIR, "..", "datasets", "clinical_ner_text2text.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models_results", "gemma-lora-adapter")

# LoRA Config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"] # Common for attention layers

# Training Args
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

def formatting_func(example):
    # SFTTrainer maps this function over the dataset (not batched by default)
    return f"Input: {example['input']}\nOutput: {example['output']}"

def main():
    if not os.path.exists(MODEL_NAME_OR_PATH):
        print(f"Error: Model path {MODEL_NAME_OR_PATH} not found.")
        return

    print(f"Loading model from {MODEL_NAME_OR_PATH}...")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.padding_side = 'right' # for SFTTrainer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model (with BitsAndBytes if GPU memory is tight, but 270m is small)
    # We load in full precision or bf16 since it is small.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # Load Dataset
    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False, # Use bf16 if possible
        push_to_hub=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("Starting LoRA training...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("LoRA training complete.")

if __name__ == "__main__":
    main()
