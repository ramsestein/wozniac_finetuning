import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)

# --- Hyperparameters ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative to this script (wozniac/training_bsc_anon/)
MODEL_NAME_OR_PATH = os.path.join(SCRIPT_DIR, "..", "models", "bsc_ehr_anon_bin")
TRAIN_FILE = os.path.join(SCRIPT_DIR, "train.conll")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models_results", "bsc-anon-finetuned")

NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WEIGHT_DECAY = 0.01

# Label Map
LABEL_LIST = ["O", "ANON-B", "ANON-I"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

class CoNLLDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_to_id, max_length):
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.sentences, self.labels = self._read_conll(file_path)

    def _read_conll(self, file_path):
        sentences = []
        labels = []
        
        current_sent = []
        current_labels = []
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return [], []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sent:
                        sentences.append(current_sent)
                        labels.append(current_labels)
                        current_sent = []
                        current_labels = []
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                token = parts[0]
                label = parts[-1] 
                
                current_sent.append(token)
                current_labels.append(label)
            
            # Flush last sentence if any
            if current_sent:
                sentences.append(current_sent)
                labels.append(current_labels)
                
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.labels[idx]
        
        # Convert tokens to IDs directly since they are subwords
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Convert labels to IDs
        label_ids = [self.label_to_id.get(l, 0) for l in labels] # Default to O if unknown
        
        # Add special tokens
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        
        input_ids = [cls_token_id] + input_ids + [sep_token_id]
        label_ids = [-100] + label_ids + [-100]
        
        # Truncate if necessary 
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length-1] + [sep_token_id]
            label_ids = label_ids[:self.max_length-1] + [-100]
            
        # Attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids
        }

def main():
    print(f"Loading model and tokenizer from {MODEL_NAME_OR_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, add_prefix_space=True)
    # Load model expecting existing head (3 labels)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME_OR_PATH, 
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
        # No ignore_mismatched_sizes=True needed as we expect match
    )

    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = CoNLLDataset(TRAIN_FILE, tokenizer, LABEL_TO_ID, MAX_LENGTH)
    
    print(f"Dataset size: {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_strategy="epoch",      # Save every epoch
        eval_strategy="no",         # No val set
        save_total_limit=2,         # Keep only last 2 checkpoints
        remove_unused_columns=False
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer 
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    main()
