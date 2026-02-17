import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative to this script (wozniac/training_gemma_lora/)
BASE_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "gemma3_270m")
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "..", "models_results", "gemma-lora-adapter")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models_results", "gemma-finetuned-merged")

def main():
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"Error: Base model path {BASE_MODEL_PATH} not found.")
        return
    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: Adapter path {ADAPTER_PATH} not found. Train first.")
        return

    print(f"Loading base model from {BASE_MODEL_PATH}...")
    # Load base model in full precision (or bf16) to merge.
    # Merging quantized models is trickier (needs dequantization).
    # Assuming we trained on bf16/fp32 or unquantized base model.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16, # Use float16 for merging to save memory if compatible
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    print(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {OUTPUT_DIR}...")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Merge complete.")

if __name__ == "__main__":
    main()
