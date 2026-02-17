# Lab: Clinical NER Fine-Tuning Experiments

Welcome to the Clinical NER (Named Entity Recognition) Lab! This repository contains a series of experiments designed to teach different methodologies for fine-tuning Large Language Models (LLMs) on specialized medical data.

## 🎯 Objectives
- Generate high-quality synthetic clinical data.
- Fine-tune Encoder models (RoBERTa) for token classification.
- Fine-tune Decoder models (Gemma 3) using Parameter-Efficient Fine-Tuning (LoRA).
- Explore advanced optimization: **Wanda Pruning** followed by **Selective Fine-Tuning**.

---

## 🧪 Experiment 1: Encoder Models (BSC-BIO)
High-performance Spanish biomedical models based on RoBERTa.

### 1.1 BIO Format Conversion
Medical NER datasets are often provided in JSON format. We convert them to **CoNLL/BIO** format (Beginning, Inside, Outside) so the model can learn token-level boundaries.
- **Script**: `training_bsc_bio/convert_to_bio.py`
- **Takeaway**: Subword tokenization (BPE) affects label alignment. We must use the model's tokenizer during conversion.

### 1.2 Full Fine-Tuning
Training the entire classification head for a specific task (Diagnosis detection).
- **Script**: `training_bsc_bio/train_model.py`
- **Key Hyperparameter**: `ignore_mismatched_sizes=True` allows loading weights when the number of output labels changes.

---

## 🚀 Experiment 2: Decoder Models (Gemma 3)
Generative models adapted for Text-to-Text extraction.

### 2.1 LoRA (Low-Rank Adaptation)
Instead of updating all weights (billions), we only train small "adapter" matrices.
- **Script**: `training_gemma_lora/train_lora.py`
- **Process**: Train adapter -> Merge with Base -> Deploy.
- **Benefit**: Significantly lower VRAM usage and storage.

### 2.2 Wanda Pruning + Selective Fine-Tuning
An advanced pipeline for efficient, sparse models.
- **Step 1 (Wanda)**: Prune 50% of weights without retraining by calculating importance: `|W| * ||X||₂`.
- **Step 2 (Ranking)**: Measure which layers were most "damaged" by pruning.
- **Step 3 (Selective FT)**: Freeze the model and only unfreeze the Top-K damaged layers to recover performance.
- **Script**: `training_gemma_wanda/wanda_pipeline.py`

---

## 🛠️ How to Run

1. **Environment Setup**:
   ```bash
   pip install transformers datasets peft trl torch accelerate bitsandbytes
   ```

2. **Data Generation**:
   ```bash
   python generate_dataset.py
   ```

3. **Gemma LoRA Training**:
   ```bash
   python training_gemma_lora/train_lora.py
   python training_gemma_lora/merge_model.py
   ```

4. **Wanda Pipeline**:
   ```bash
   python training_gemma_wanda/wanda_pipeline.py
   ```

---

## 📊 Results Summary
All trained models are stored in `models_results/`. Check the specific logs for loss curves and training metrics.

> [!TIP]
> **Students**: Compare the training time and model size between the merged LoRA model and the Wanda-pruned model. Which one is more efficient for deployment?
