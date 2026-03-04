# Patent Green Technology Classification

**A comprehensive deep learning system for classifying patent claims as green/environmentally beneficial technology using baseline models, LLM fine-tuning, and multi-agent debate.**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Project Overview

This project implements an end-to-end pipeline for classifying patent claims as **GREEN** (environmentally beneficial) or **NOT GREEN** technology. The system progresses from baseline models to advanced LLM fine-tuning and culminates in a sophisticated multi-agent debate system for robust predictions.

**Dataset**: 1.37M patent claims from Hugging Face with 650+ CPC code columns

### Key Achievements

| Method | Accuracy | F1 Score | Key Advantage |
|--------|----------|----------|---|
| Baseline (LogReg + Frozen) | 81.0% | 0.810 | Fast, lightweight |
| Fine-tuned PatentSBERTa | **83.5%** | **0.835** | Best accuracy-speed tradeoff |
| QLoRA Mistral | 80.5% | 0.805 | JSON output, memory efficient |
| Multi-Agent Debate | **84.5%** | **0.845** | Most interpretable, robust |

---

## 📋 Table of Contents

- [Installation](#-installation--requirements)
- [Quick Start](#-quick-start)
- [Project Structure](#-file-organization)
- [Workflow](#-step-by-step-workflow)
- [Models & Methods](#-models--methods)
- [Results](#-results--metrics)
- [Usage Guide](#-usage-guide)
- [Contributing](#-contributing)

---

## 🚀 Installation & Requirements

### Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (for GPU acceleration)
- **16GB+ RAM** (for model loading)
- **50GB+ disk space** (for model checkpoints)

### Setup

#### 1. Clone Repository
```bash
git clone https://github.com/KarolinaBohdan/Deep_Learning_Final.git
cd Deep_Learning_Final
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Download Pre-trained Models (First Run)
```bash
# PatentSBERTa auto-downloads
python -c "from transformers import AutoModel; AutoModel.from_pretrained('AI-Growth-Lab/PatentSBERTa')"

# Mistral-7B-Instruct (~14GB)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

### requirements.txt
```
# Core ML/DL
torch>=2.0.0
transformers>=4.41.0
peft>=0.11.0
trl>=0.8.0
datasets>=2.14.0

# Scientific Computing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Utilities
tqdm>=4.66.0
joblib>=1.3.0
bitsandbytes>=0.42.0
accelerate>=0.25.0
pyarrow>=13.0.0
```

---

## ⚡ Quick Start

### Classify New Patents
```python
import torch
import json
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned PatentSBERTa (fastest, best accuracy)
tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa")
model = AutoModel.from_pretrained(
    "partD_patentsberta_finetuned_evalsilver_only/",
    device_map=device
)

# Classify a claim
new_claim = "A method for converting wind energy into electrical power with 95% efficiency..."

inputs = tokenizer(new_claim, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)
# Process embeddings for classification...
```

### Batch Evaluation
```bash
python A3_qlora_inference_.py \
  --adapter_dir qlora_mistral_adapter_json \
  --eval_silver eval_silver.parquet \
  --output_csv predictions.csv
```

---

## 📁 File Organization
```
Deep_Learning_Final/
│
├── README.md (this file)
├── requirements.txt
│
├── ASSIGNMENT 2 - BASELINE & LLM INTEGRATION
│   ├── Assignment 2_Part A.ipynb              [Data prep & splitting]
│   ├── Assigment2_baseline_partA.py           [Frozen embeddings baseline]
│   ├── A2_partB_uncertainty.py                [Uncertainty estimation]
│   ├── A2_partC_llm_local.py                  [Mistral labeling + HITL]
│   ├── A2_partD_finetune.py                   [PatentSBERTa fine-tuning]
│   └── results/
│       ├── A2_LR_results_partA.out
│       ├── A2PartD_finetuned_results/
│       └── finetune_part3_final_metrics/
│
├── ASSIGNMENT 3 - QLORA FINE-TUNING
│   ├── A3_qlora_finetuning.py                 [QLoRA training]
│   ├── A3_qlora_inference_.py                 [QLoRA inference]
│   ├── qlora_mistral_adapter_json/            [LoRA adapter checkpoint]
│   └── Qlora_patentsberta_finetuning_final_metrics/
│
├── ASSIGNMENT 4 - MULTI-AGENT DEBATE
│   ├── mas_final.py                           [MAS implementation]
│   ├── MAS_train_patentsberta_human3/         [Training procedures]
│   └── results/
│       ├── mas_debate_results.csv
│       └── MAS evaluation reports
│
├── DATA FILES
│   ├── patents_50k_green.parquet              [50k balanced dataset]
│   ├── train_silver.parquet                   [30k training split]
│   ├── eval_silver.parquet                    [10k eval split]
│   ├── pool_unlabeled.parquet                 [10k pool split]
│   ├── hitl_green_100.csv                     [100 HITL samples]
│   ├── hitl_green_100_llm_corrected.csv       [100 HITL samples (verified)]
│   └── embeddings/
│       ├── X_train.npy
│       ├── X_eval.npy
│       └── X_pool.npy
│
├── MODEL CHECKPOINTS
│   ├── baseline_logreg.joblib
│   ├── partD_patentsberta_finetuned_evalsilver_only/
│   └── qlora_mistral_adapter_json/
│
└── M4_ASSIGNEMENT1_BDS_MFC.ipynb              [Portfolio assignment]
```

---

## 🔄 Step-by-Step Workflow

### Phase 1: Data Preparation
```
Load 1.37M raw patent claims
    ↓
Create is_green_silver labels from Y02* CPC codes
    ↓
Balance to 50k (25k green + 25k non-green)
    ↓
Split into train/eval/pool (60/20/20)
    ↓
Output: 4 parquet files
```

**Run**: `jupyter notebook "Assignment 2_Part A.ipynb"`

### Phase 2: Baseline Establishment
```
Generate PatentSBERTa embeddings (frozen, 768-dim)
    ↓
Train Logistic Regression on train_silver
    ↓
Evaluate on eval_silver
    ↓
Output: Baseline metrics + model checkpoint
```

**Run**: `python Assigment2_baseline_partA.py`  
**Results**: 81% accuracy, 0.810 F1 score

### Phase 3: Uncertainty & LLM Labeling
```
Calculate uncertainty scores on pool_unlabeled
    ↓
Select top-100 uncertain samples
    ↓
Use Mistral-7B to generate initial labels + rationales
    ↓
Human review and correction (HITL)
    ↓
Output: 100 gold labels with human validation
```

**Run**: `python A2_partC_llm_local.py`

### Phase 4: Fine-tuning PatentSBERTa
```
Load 30k train_silver + 100 gold labels
    ↓
Tokenize with PatentSBERTa tokenizer
    ↓
Fine-tune PatentSBERTa (unfrozen)
    ↓
Evaluate on eval_silver
```

**Run**: `python A2_partD_finetune.py`  
**Time**: ~30-45 minutes on A100 GPU  
**Results**: 83.5% accuracy, 0.835 F1 score (+2.5% improvement)

### Phase 5: QLoRA Fine-tuning
```
Quantize Mistral-7B to 4-bit
    ↓
Add LoRA adapters (trainable)
    ↓
Train on 30k train_silver + 100 gold
    ↓
Output JSON predictions with rationales
```

**Run Training**:
```bash
python A3_qlora_finetuning.py \
  --output_dir qlora_mistral_adapter_json \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4
```

**Time**: ~2-3 hours on A100 GPU  
**Memory**: ~3.5GB (vs 14GB for full model)

**Run Inference**: `python A3_qlora_inference_.py --adapter_dir qlora_mistral_adapter_json`

### Phase 6: Multi-Agent Debate System
```
Load fine-tuned Advocate (Mistral + LoRA)
    ↓
Load base Skeptic (Mistral base)
    ↓
Load base Judge (Mistral base)
    ↓
For each claim: Advocate argues GREEN, 
Skeptic argues NOT GREEN, Judge decides
    ↓
Output: Final predictions with explanations
```

**Run**:
```bash
python mas_final.py \
  --eval_silver eval_silver.parquet \
  --hitl_file hitl_green_100_llm_corrected.csv \
  --adapter_path qlora_mistral_adapter_json \
  --output mas_debate_results.csv
```

**Time**: ~4-6 hours | **Memory**: ~16GB | **Results**: 84.5% accuracy, 0.845 F1 score

---

## 🤖 Models & Methods

### PatentSBERTa

- **Model**: BERT fine-tuned on patent documents
- **Embedding Dimension**: 768
- **Max Sequence Length**: 256 tokens
- **Training Data**: 16M patent documents
- **Use Case**: Feature extraction, domain-specific understanding

### Mistral-7B-Instruct

- **Parameters**: 7 billion
- **Context Window**: 8,192 tokens
- **Instruction Tuning**: Yes (optimized for instruction following)
- **Fine-tuning Method**: QLoRA (efficient, memory-friendly)
- **Quantization**: 4-bit (14GB → 3.5GB)

### Multi-Agent Debate System

Three specialized agents provide robust predictions:
```
Patent Claim
    ├─→ ADVOCATE (Mistral + LoRA)  → "Argue for GREEN (1)"
    │
    ├─→ SKEPTIC (Mistral base)     → "Argue against NOT GREEN (0)"
    │
    └─→ JUDGE (Mistral base)       → Reviews both, decides
        Output: {label, confidence, reasoning}
```

**Why MAS Works**:
- ✅ Reduces single-model bias
- ✅ Better calibration through perspective diversity
- ✅ Explainable reasoning (debate arguments)
- ✅ Robust: unlikely all 3 agents make same mistake

---

## 📊 Results & Metrics

### Performance Comparison

| Method | Accuracy | F1 | Memory | Speed | Best For |
|--------|----------|-----|--------|-------|----------|
| Baseline (LogReg) | 81.0% | 0.810 | 0.5GB | ⚡⚡⚡ | Quick inference |
| Fine-tuned PatentSBERTa | **83.5%** | **0.835** | 1.2GB | ⚡⚡ | **Production** |
| QLoRA Mistral | 80.5% | 0.805 | 3.5GB | ⚡ | JSON output |
| Multi-Agent Debate | **84.5%** | **0.845** | 16GB | 🐢 | **Interpretability** |

### Baseline Performance
```
Dataset: eval_silver (10,000 samples)
Model: PatentSBERTa + Logistic Regression

              precision    recall  f1-score
    Non-Green       0.82      0.79      0.80
    Green           0.80      0.83      0.81
    Accuracy                          0.81
```

### Fine-tuned PatentSBERTa
```
Dataset: eval_silver (10,000 samples)

              precision    recall  f1-score
    Non-Green       0.84      0.82      0.83
    Green           0.83      0.85      0.84
    Accuracy                          0.835
```

### Multi-Agent Debate System
```
Dataset: eval_silver (10,000 samples)

              precision    recall  f1-score
    Non-Green       0.85      0.84      0.845
    Green           0.84      0.85      0.845
    Accuracy                          0.845
    F1 Improvement (vs baseline)      +3.5%
```

---

## 📖 Usage Guide

### Classify New Patents
```python
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned PatentSBERTa
tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa")
model = AutoModel.from_pretrained(
    "partD_patentsberta_finetuned_evalsilver_only/",
    device_map=device
)

# Classify
new_claim = "A method for converting wind energy..."
inputs = tokenizer(new_claim, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)
# Process embeddings...
```

### Batch Evaluation
```bash
python A3_qlora_inference_.py \
  --adapter_dir qlora_mistral_adapter_json \
  --eval_silver eval_silver.parquet \
  --output_csv predictions.csv
```

### Multi-Agent Debate on Custom Data
```bash
python mas_final.py \
  --eval_silver your_data.parquet \
  --adapter_path qlora_mistral_adapter_json \
  --output results.csv
```

---

## 📈 Green Technology Definition

The system classifies patents as GREEN if they describe:

- ✅ Emissions reduction, exhaust treatment, carbon capture
- ✅ Pollution control, filtration, removal
- ✅ Recycling, waste processing, circular economy
- ✅ Water treatment, purification, desalination
- ✅ Renewable energy, storage, grid integration
- ✅ Explicit energy-saving (not just "efficient")
- ✅ Environmental monitoring reducing waste

---

## 🤝 Contributing

### Contributors

- **Karolina Bohdan** 
- **Faraiba Farnan** 
- **Maleha Afzal** 

---

## 🙏 Acknowledgments

- **Hugging Face** for the patent dataset and transformers library
- **AI-Growth-Lab** for PatentSBERTa
- **Mistral AI** for Mistral-7B-Instruct
- **Meta** for PEFT and QLoRA implementations

---

## 📧 Questions?

For questions or issues, please open a GitHub issue or contact the project maintainers.

**Happy classifying! 🚀**
