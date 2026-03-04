# Green Patent Classification with QLoRA and Multi-Agent Reasoning

## Overview

This project builds a system that identifies whether a patent claim describes environmentally beneficial ("green") technology.

Patent texts are extremely dense and technical, which makes them difficult to classify using simple keyword methods. To address this, we combine:

- Domain adaptation using **QLoRA fine-tuning**
- **Human-in-the-loop labeling**
- A **Multi-Agent System (MAS)** that performs structured reasoning before producing the final label

The final system classifies patent claims as:

```
1 → GREEN technology  
0 → NOT_GREEN
```

and generates a short rationale explaining the decision.

---

# Project Overview

The pipeline consists of four main stages:

1. Baseline classifier + uncertainty sampling  
2. Human-in-the-loop correction of uncertain samples  
3. QLoRA fine-tuning of a large language model  
4. Multi-Agent reasoning for final classification  

The goal is to improve classification reliability and reduce false positives caused by vague sustainability claims.

---

# Dataset

The dataset contains **50,000 patent claims**.

Dataset splits:

| Split | Size | Purpose |
|------|------|------|
| train_silver | 30k | model training |
| eval_silver | 10k | evaluation |
| pool_unlabeled | 10k | uncertainty sampling |

---

# Step 1 — Baseline Model

We first train a baseline classifier using:

- **PatentSBERTa embeddings**
- **Logistic Regression**

The model predicts the probability that a patent claim is green.

From these probabilities we compute an uncertainty score:

```
u = 1 − 2|p − 0.5|
```

The **100 most uncertain patents** are selected for further inspection.

---

# Step 2 — Human-in-the-Loop Labeling

The selected **100 uncertain patents** are reviewed manually.

Initially, the claims are labeled by an LLM, but these predictions are verified and corrected by a human to ensure higher quality labels.

The resulting dataset:

```
hitl_green_100_llm_corrected.csv
```

These corrected labels help improve the final model.

---

# Step 3 — QLoRA Fine-Tuning

To better understand the dense linguistic structure of patent claims, we fine-tune a large language model using **QLoRA**.

Model:

```
Mistral-7B-Instruct
```

Training setup:

- Method: **QLoRA adapters**
- Training subset: **10,000 randomly sampled patents**
- GPU: **AAU AI Lab cluster**

The fine-tuned adapter allows the model to better understand:

- patent claim structure  
- environmental mechanisms  
- green technology terminology  

---

# Step 4 — Multi-Agent System (MAS)

Instead of relying on a single prediction, we use a **multi-agent reasoning system**.

Three agents collaborate to make the final decision.

### Advocate

Argues why the patent claim should be classified as green.

### Skeptic

Challenges the advocate and identifies possible greenwashing or weak environmental claims.

### Judge

Evaluates both arguments and produces the final output:

```json
{
 "label": 0 or 1,
 "rationale": "short explanation"
}
```

This structure forces the model to reason about the claim before classifying it, which helps reduce overly optimistic predictions.

---

# Prompt Engineering

Different prompting strategies were used across the agents.

### Advocate & Skeptic

We use **zero-shot role prompting**.

Each agent receives:

- a clear persona  
- explicit behavioral rules  
- a shared definition of GREEN technology  

This encourages adversarial reasoning.

### Judge

The Judge agent uses **few-shot prompting**.

The prompt contains example classifications with expected JSON outputs to ensure the final predictions follow the correct format.

---

# Definition of GREEN Technology

A patent claim is labeled **GREEN (1)** only if it clearly describes a technology with a direct environmental benefit, such as:

- emissions reduction  
- pollution control  
- recycling or waste processing  
- water purification  
- renewable energy generation  
- energy storage  
- explicit energy-saving mechanisms  

Claims that only use vague language such as *efficient*, *eco-friendly*, or *sustainable* without describing a mechanism are labeled **NOT_GREEN (0)**.

---

# Repository Structure

```
Deep_Learning_Final
│
├── A2_partB_uncertainty
├── A2_partC_llm_local.py
├── A2_partD_finetune.py
│
├── A3_qlora_finetuning.py
├── A3_qlora_inference_.py
│
├── mas_final.py
│
├── hitl_green_100.csv
├── MAS_final_labels.csv
│
└── README.md
```

---

# Running the Pipeline

### 1. Uncertainty sampling

Select the most uncertain patents.

```
python A2_partB_uncertainty
```

### 2. Fine-tune the QLoRA model

```
python A3_qlora_finetuning.py
```

This produces the **LoRA adapter weights**.

### 3. Run inference

```
python A3_qlora_inference_.py
```

The model outputs labels and rationales.

### 4. Run the Multi-Agent System

```
python mas_final.py
```

The pipeline runs the **Advocate → Skeptic → Judge** reasoning process and produces the final classification.

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- PEFT (QLoRA)
- PatentSBERTa
- scikit-learn
- Multi-Agent prompting
- AAU AI Lab GPU cluster
