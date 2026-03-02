import os
import re
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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report


PARQUET_PATH = "patents_50k_green_1.parquet"
HITL_PATH = "hitl_green_100_llm_corrected.csv"
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"

MAX_LEN = 256
EPOCHS = 1
LR = 2e-5
BATCH_SIZE = 16
SEED = 42

OUT_DIR = "partD_patentsberta_finetuned_evalsilver_only"
METRICS_PATH = os.path.join(OUT_DIR, "final_metrics_eval_silver_only.txt")

set_seed(SEED)


def clean_doc_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return str(int(round(float(s))))
    except Exception:
        digits = re.sub(r"\D", "", s)
        return digits if digits else None


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

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH).copy()

required = {"id", "split", "text", "is_green_silver"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Parquet missing columns: {missing}. Found: {df.columns.tolist()}")

df["id_str"] = df["id"].astype(str)
df["text"] = df["text"].fillna("").astype(str)
df["is_green_silver"] = pd.to_numeric(df["is_green_silver"], errors="coerce")

hitl = pd.read_csv(HITL_PATH).copy()

required_cols = {"doc_id", "is_green_human"}
missing = required_cols - set(hitl.columns)
if missing:
    raise ValueError(f"HITL CSV missing required columns: {missing}. Found: {hitl.columns.tolist()}")

hitl["doc_id_str"] = hitl["doc_id"].apply(clean_doc_id)
hitl["is_green_human"] = pd.to_numeric(hitl["is_green_human"], errors="coerce")

hitl = hitl[hitl["doc_id_str"].notna()].copy()
hitl = hitl[hitl["is_green_human"].isin([0, 1])].copy()
hitl["is_green_human"] = hitl["is_green_human"].astype(int)
hitl = hitl.drop_duplicates(subset=["doc_id_str"], keep="last")

df = df.merge(
    hitl[["doc_id_str", "is_green_human"]],
    left_on="id_str",
    right_on="doc_id_str",
    how="left",
)

gold_mask = df["is_green_human"].notna()
gold_100_df = df[gold_mask].copy()

print("Gold rows matched:", len(gold_100_df))
if len(gold_100_df) == 0:
    raise RuntimeError("0 gold rows matched. Check HITL doc_id matches parquet id.")

print("Gold split distribution:\n", gold_100_df["split"].value_counts(dropna=False).to_string())

df["is_green_gold"] = df["is_green_silver"]
df.loc[gold_mask, "is_green_gold"] = df.loc[gold_mask, "is_green_human"]

train_silver_df = df[df["split"] == "train_silver"].copy()
eval_silver_df = df[df["split"] == "eval_silver"].copy()

if len(train_silver_df) == 0 or len(eval_silver_df) == 0:
    raise RuntimeError("Missing train_silver or eval_silver rows. Check split values in parquet.")

train_silver_df = train_silver_df[~train_silver_df["id_str"].isin(gold_100_df["id_str"])].copy()

train_final_df = pd.concat([train_silver_df, gold_100_df], ignore_index=True)
train_final_df = train_final_df[["text", "is_green_gold"]].rename(columns={"is_green_gold": "label"})

eval_silver_df2 = eval_silver_df[["text", "is_green_silver"]].rename(columns={"is_green_silver": "label"})

train_final_df = filter_binary(train_final_df, "label")
eval_silver_df2 = filter_binary(eval_silver_df2, "label")

print("train_silver rows (after removing any overlapping gold ids):", len(train_silver_df))
print("training rows total (train_silver + gold_100):", len(train_final_df))
print("eval_silver rows:", len(eval_silver_df2))

print("\nTraining label counts:\n", train_final_df["label"].value_counts().to_string())
print("\nEval label counts:\n", eval_silver_df2["label"].value_counts().to_string())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_ds = make_ds(train_final_df["text"].tolist(), train_final_df["label"].tolist(), tokenizer)
eval_ds = make_ds(eval_silver_df2["text"].tolist(), eval_silver_df2["label"].tolist(), tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_metrics = trainer.evaluate(eval_dataset=eval_ds)

pred_out = trainer.predict(eval_ds)
eval_logits = pred_out.predictions
eval_preds = np.argmax(eval_logits, axis=-1)
eval_labels = np.array(eval_silver_df2["label"].tolist(), dtype=int)

cm = confusion_matrix(eval_labels, eval_preds, labels=[0, 1])
report = classification_report(eval_labels, eval_preds, digits=4)

with open(METRICS_PATH, "w") as f:
    f.write("EVAL_SILVER_ONLY\n")
    for k, v in eval_metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\nEVAL_SILVER_CONFUSION_MATRIX (rows=true 0/1, cols=pred 0/1)\n")
    f.write(str(cm) + "\n")
    f.write("\nEVAL_SILVER_CLASSIFICATION_REPORT\n")
    f.write(report)

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("\nSaved model to:", OUT_DIR)
print("Saved metrics to:", METRICS_PATH)
