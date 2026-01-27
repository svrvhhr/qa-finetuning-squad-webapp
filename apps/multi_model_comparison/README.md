---
title: SQuAD QA - Model Comparison
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.4.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# Question Answering: Model Comparison

Compare three transformer models (DistilBERT, RoBERTa, DeBERTa) fine-tuned on SQuAD v1.1.

## Features
- 📊 **Side-by-side comparison** of answers from all three models
- 🎯 **Confidence scores** for each prediction
- 🎨 **Visual highlighting** of answers in context (color intensity shows confidence)
- ⚡ **Performance metrics** (F1 Score, Exact Match)
- 🔄 **Real-time inference** on custom questions

## Models

All models were fine-tuned with identical hyperparameters on SQuAD v1.1:

| Model | Parameters | F1 Score | Exact Match |
|-------|-----------|----------|-------------|
| **DistilBERT** | 66M | 84.41% | 75.81% |
| **RoBERTa** | 125M | 91.96% | 85.65% |
| **DeBERTa** | 184M | 93.01% | 86.58% |

## Training Configuration

- **Dataset:** SQuAD v1.1 (87,599 training examples, 10,570 validation examples)
- **Epochs:** 3
- **Batch Size:** 64
- **Learning Rate:** 3e-5
- **Warmup Ratio:** 0.1
- **Max Sequence Length:** 384 tokens
- **Document Stride:** 128 tokens

## How to Use

1. **Enter a Context:** Paste a paragraph of text that contains the answer
2. **Ask a Question:** Type your question about the context
3. **Compare Answers:** See how each model extracts the answer with different confidence levels

The highlighted text shows where each model found its answer, with color intensity representing confidence.

## Example

**Context:** "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889."

**Question:** "When was the Eiffel Tower completed?"

**Expected Answer:** "1889"

## Technical Details

- **Base Models:**
  - DistilBERT: `distilbert-base-uncased`
  - RoBERTa: `roberta-base`
  - DeBERTa: `microsoft/deberta-v3-base`

- **Task:** Extractive Question Answering
- **Framework:** Hugging Face Transformers
- **Interface:** Gradio

## Links

- [DistilBERT Model Card](https://huggingface.co/khaledbouabdallah/distilbert-squad-finetuned)
- [RoBERTa Model Card](https://huggingface.co/khaledbouabdallah/roberta-squad-finetuned)
- [DeBERTa Model Card](https://huggingface.co/khaledbouabdallah/deberta-squad-finetuned)

## Credits

Fine-tuned models created as part of M2 Datascale - Fouille de Données project.
