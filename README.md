1. Project Overview
Title: Green Patent Detection (PatentSBERTa): Active Learning + LLM→Human HITL
Goal: Detect environmentally beneficial patents using deep learning and human-in-the-loop (HITL) feedback
2. Step-by-Step Workflow (organized into 4 Assignments)
Assignment 1: Data Preparation & Baseline (Part A)

Load 1.37M patent claims from the AI-Growth-Lab/patents_claims dataset
Create balanced 50k dataset (25k green + 25k non-green patents)
Split into train (60%), eval (20%), pool (20%)
Use PatentSBERTa embeddings with frozen parameters
Train Logistic Regression baseline
Assignment 2: Advanced Methods (Parts B-D)

Part B: Uncertainty estimation on the baseline model
Part C: LLM integration (Mistral-7B) for initial HITL labeling
Generate label suggestions with confidence + rationale
Create human review interface for annotation
Part D: Fine-tune PatentSBERTa with human-corrected labels
Combine silver (model) + gold (human) labels
Evaluate on eval_silver split
Assignment 3: LLM Fine-tuning with QLoRA

Fine-tune Mistral-7B-Instruct using QLoRA (4-bit quantization)
Train JSON output format: {"label": 0/1, "rationale": "..."}
Inference on eval_silver + HITL-100 samples
Compare with baseline performance
Assignment 4: Multi-Agent Debate System (MAS)

Implements 3 agents: Advocate, Skeptic, Judge
Advocate (with QLoRA adapter): argues claim is GREEN
Skeptic (base model): argues claim is NOT GREEN
Judge (base model): makes final decision based on debate
More robust than single-model predictions
3. Key Files & Their Purpose
Jupyter notebooks for data prep and exploration
Python scripts for baseline, fine-tuning, and inference
CSV data files with HITL annotations
Results and metrics files for each phase
4. Requirements
Core: torch, transformers, pandas, sklearn, datasets
Advanced: peft, trl, bitsandbytes (for QLoRA)
Optional: tqdm for progress tracking
5. Quick Start Commands
Data preparation and splitting
Running baselines
LLM labeling and fine-tuning
QLoRA training and inference
Multi-agent evaluation
The README now serves as a complete guide for anyone wanting to understand, reproduce, or extend this green patent detection project! 🚀

Make these code changes?
README.md

md
# Green Patent Detection

## Description
This project focuses on using deep learning techniques to detect green patents, which are patents that are related to environmentally friendly technologies. The goal is to build a model that can accurately classify these patents based on textual descriptions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KarolinaBohdan/Deep_Learning_Final.git
   cd Deep_Learning_Final
Set up a Python environment (recommended to use virtualenv or conda):
bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install required packages:
bash
pip install -r requirements.txt
Dataset
The dataset used for this project is obtained from [insert dataset source]. It contains [brief description of the contents]. Make sure to download and place the dataset in the data/ directory.

Model Architecture
The model architecture consists of [brief description of the architecture, e.g., number of layers, types of layers, etc.]. The model is built using [framework used, e.g., TensorFlow, PyTorch].

Training
To train the model, run the following command:

bash
python train.py --epochs 50 --batch_size 32
Make sure to adjust the parameters according to your compute resources.

Evaluation
After training the model, you can evaluate its performance with:

bash
python evaluate.py --model_path path/to/saved_model
This will output accuracy, precision, recall, and F1 score metrics.

Usage
To use the model for prediction on new data, run:

bash
python predict.py --input_file path/to/input_data
