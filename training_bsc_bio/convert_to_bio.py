import json
import os
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "datasets", "clinical_ner_100.jsonl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "train.conll")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "bsc-bio-ehr-es")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} not found.")
        # Fallback to huggingface hub if local not found? 
        # But user said "models folder". Let's assume it exists.
        # If not, we might fail.
        pass

    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Converting {INPUT_FILE} to BIO format using tokenizer...")
    
    count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                text = data.get("text", "")
                entities = data.get("entities", [])
                
                if not text:
                    continue

                # 1. Char-level labels
                char_labels = ["O"] * len(text)
                sorted_entities = sorted(entities, key=lambda x: len(x["text"]), reverse=True)
                
                for entity in sorted_entities:
                    entity_text = entity["text"]
                    label = "DIAGNOSIS"
                    
                    search_start = 0
                    while True:
                        start_idx = text.find(entity_text, search_start)
                        if start_idx == -1:
                            break
                        
                        end_idx = start_idx + len(entity_text)
                        
                        # Check availability
                        is_free = True
                        for i in range(start_idx, end_idx):
                            if char_labels[i] != "O":
                                is_free = False
                                break
                        
                        if is_free:
                            char_labels[start_idx] = f"B-{label}"
                            for i in range(start_idx + 1, end_idx):
                                char_labels[i] = f"I-{label}"
                        
                        search_start = start_idx + 1

                # 2. Tokenize with offsets
                # add_special_tokens=False to avoiding [CLS]/[SEP] in the middle of CoNLL file if we treat line by line as sentences
                # But usually for training we might want them? 
                # CoNLL format usually just lists tokens. The model training script will add special tokens.
                # So we export raw tokens.
                encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
                offsets = encoding["offset_mapping"]
                
                for token, (start, end) in zip(tokens, offsets):
                    # Skip special tokens if any remain (usually won't with add_special_tokens=False)
                    # Also skip zero-length tokens if any
                    if start == end:
                        continue
                    
                    # Label strategy:
                    # Look at the label of the start character
                    # If it is 'B-TAG', token is 'B-TAG'
                    # If it is 'I-TAG', token is 'I-TAG'
                    # If it is 'O', token is 'O'
                    
                    # Robustness: ensure start is within bounds
                    if start < len(char_labels):
                        token_label = char_labels[start]
                    else:
                        token_label = "O"
                        
                    fout.write(f"{token} {token_label}\n")
                
                fout.write("\n")
                count += 1
                
            except json.JSONDecodeError:
                print(f"Warning: JSON error on line {line_num}")
                continue

    print(f"Done. Converted {count} sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
