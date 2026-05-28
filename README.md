# Building LLM from Scratch

> **Not vibe-coded. Every line was typed manually for the purpose of learning.**

A hands-on implementation of a Large Language Model built from the ground up using PyTorch, following the architecture of GPT-2. This project covers everything from tokenization to pretraining, fine-tuning, and evaluation.

---

## What's covered

| File | Description |
|---|---|
| `tokenizer.py` | Simple tokenizer implementation |
| `BPE.py` | Byte Pair Encoding tokenizer |
| `selfatten.py` | Self-attention mechanism |
| `llm_architecture.py` | Full GPT model architecture |
| `pretraining.py` | Pretraining loop and text generation utilities |
| `download_model.py` / `gpt_download.py` | Download pretrained GPT-2 weights |
| `loading_gpt2.py` / `loading_pre_trained_model.py` | Load pretrained GPT-2 weights into the model |
| `download_classification_data.py` | Download spam classification dataset |
| `classification_finetuning.py` | Fine-tune GPT-2 for spam/not-spam classification |
| `gpt2_classfier_model.py` | GPT-2 classifier model definition |
| `instruction_dataset_download.py` | Download instruction-following dataset |
| `instruction_finetuning.py` | Fine-tune GPT-2 on instruction-following data |
| `instruction_evaluation.py` | Generate responses and evaluate them using GPT-4o-mini |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key (required for evaluation only):

```
OPENAI_API_KEY=sk-...
```

---

## Workflow

### 1. Pretraining
```bash
python pretraining.py
```

### 2. Load pretrained GPT-2 weights
```bash
python loading_pre_trained_model.py
```

### 3. Classification fine-tuning (spam detection)
```bash
python download_classification_data.py
python classification_finetuning.py
```

### 4. Instruction fine-tuning
```bash
python instruction_dataset_download.py
python instruction_finetuning.py
```

### 5. Evaluate with GPT
```bash
python instruction_evaluation.py
```

This generates model responses, then scores each one (0–100) using GPT-4o-mini and prints a summary.

---

## Requirements

- Python 3.10+
- PyTorch
- See `requirements.txt` for full list
