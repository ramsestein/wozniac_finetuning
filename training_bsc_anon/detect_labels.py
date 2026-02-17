import os
from transformers import AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "bsc_ehr_anon_bin")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        print("\n--- Model Configuration ---")
        print(f"Num Labels: {model.config.num_labels}")
        print(f"ID to Label: {model.config.id2label}")
        print(f"Label to ID: {model.config.label2id}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
