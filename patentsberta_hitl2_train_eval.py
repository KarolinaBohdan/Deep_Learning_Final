import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


PARQUET_PATH = "patents_50k_green_1.parquet"
HITL_PATH = "qlora_final_labels.csv"
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"

MAX_LEN = 256
EPOCHS = 1
LR = 2e-5
BATCH_SIZE = 16
SEED = 42

OUT_DIR = "partD_patentsberta_finetuned_hitl2"
METRICS_PATH = os.path.join(OUT_DIR, "final_metrics_eval_silver.txt")

TRAIN_SPLIT = "train_silver"
EVAL_SPLIT = "eval_silver"

PARQUET_TEXT_COL = "text"
PARQUET_LABEL_COL = "is_green_silver"

HITL_TEXT_COL = "text"
HITL_LABEL_COL = "is_green_human_2"

set_seed(SEED)


def filter_binary(df, label_col):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[label_col].isin([0, 1])].copy()
    df[label_col] = df[label_col].astype(int)
    return df


def make_ds(texts, labels, tokenizer):
    ds = Dataset.from_dict({"text": list(texts), "labels": [int(x) for x in labels]})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    return ds.map(tok, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH).copy()

required = {"split", PARQUET_TEXT_COL, PARQUET_LABEL_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Parquet missing columns: {missing}. Found: {df.columns.tolist()}")

train_df = df[df["split"] == TRAIN_SPLIT][[PARQUET_TEXT_COL, PARQUET_LABEL_COL]].rename(
    columns={PARQUET_TEXT_COL: "text", PARQUET_LABEL_COL: "label"}
)
eval_df = df[df["split"] == EVAL_SPLIT][[PARQUET_TEXT_COL, PARQUET_LABEL_COL]].rename(
    columns={PARQUET_TEXT_COL: "text", PARQUET_LABEL_COL: "label"}
)

train_df = filter_binary(train_df, "label")
eval_df = filter_binary(eval_df, "label")

hitl = pd.read_csv(HITL_PATH).copy()
need = {HITL_TEXT_COL, HITL_LABEL_COL}
missing = need - set(hitl.columns)
if missing:
    raise ValueError(f"HITL CSV missing columns: {missing}. Found: {hitl.columns.tolist()}")

hitl = hitl[[HITL_TEXT_COL, HITL_LABEL_COL]].rename(columns={HITL_TEXT_COL: "text", HITL_LABEL_COL: "label"})
hitl = filter_binary(hitl, "label")

if len(hitl) == 0:
    raise RuntimeError("HITL file has 0 valid rows after filtering label to {0,1}.")

train_final = pd.concat([train_df, hitl], ignore_index=True)

print("train_silver rows:", len(train_df))
print("HITL rows appended:", len(hitl))
print("training rows total:", len(train_final))
print("eval_silver rows:", len(eval_df))

print("\nTraining label counts:\n", train_final["label"].value_counts().to_string())
print("\nEval label counts:\n", eval_df["label"].value_counts().to_string())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_ds = make_ds(train_final["text"].tolist(), train_final["label"].tolist(), tokenizer)
eval_ds = make_ds(eval_df["text"].tolist(), eval_df["label"].tolist(), tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# NOTE: compatible with older transformers (no evaluation_strategy / save_strategy / load_best_model_at_end)
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=50,
    report_to="none",
    seed=SEED,
    bf16=use_bf16,
    fp16=torch.cuda.is_available() and not use_bf16,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate + confusion matrix
eval_metrics = trainer.evaluate(eval_dataset=eval_ds)

pred = trainer.predict(eval_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(y_true, y_pred, digits=4)

with open(METRICS_PATH, "w") as f:
    f.write("EVAL_SILVER\n")
    f.write(f"TRAIN_SPLIT={TRAIN_SPLIT} (label={PARQUET_LABEL_COL}) + HITL(label={HITL_LABEL_COL})\n")
    f.write(f"EVAL_SPLIT={EVAL_SPLIT} (label={PARQUET_LABEL_COL})\n\n")
    for k, v in eval_metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\nCONFUSION_MATRIX (rows=true 0/1, cols=pred 0/1)\n")
    f.write(str(cm) + "\n")
    f.write("\nCLASSIFICATION_REPORT\n")
    f.write(report)

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("\nSaved model to:", OUT_DIR)
print("Saved metrics to:", METRICS_PATH)