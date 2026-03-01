import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib


# ----------------------------
# Load dataset
# ----------------------------

df = pd.read_parquet('patents_50k_green_1.parquet')

train_df = df[df["split"] == "train_silver"].reset_index(drop=True)
eval_df  = df[df["split"] == "eval_silver"].reset_index(drop=True)
pool_df  = df[df["split"] == "pool_unlabeled"].reset_index(drop=True)

X_train_text = train_df["text"].astype(str).tolist()
y_train = train_df["is_green_silver"].astype(int).values

X_eval_text = eval_df["text"].astype(str).tolist()
y_eval = eval_df["is_green_silver"].astype(int).values

X_pool_text = pool_df["text"].astype(str).tolist()


# ----------------------------
# Load PatentSBERTa (FROZEN)
# ----------------------------

MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
encoder.eval()


@torch.no_grad()
def embed_texts(texts, batch_size=32, max_length=256):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = encoder(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


# ----------------------------
# Generate embeddings
# ----------------------------

X_train = embed_texts(X_train_text)
X_eval  = embed_texts(X_eval_text)
X_pool  = embed_texts(X_pool_text)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_eval.npy", X_eval)
np.save("y_eval.npy", y_eval)
np.save("X_pool.npy", X_pool)


# ----------------------------
# Train Logistic Regression
# ----------------------------

clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(X_train, y_train)

joblib.dump(clf, "baseline_logreg.joblib")


# ----------------------------
# Evaluate
# ----------------------------

pred = clf.predict(X_eval)

print(classification_report(y_eval, pred, digits=4))

p, r, f1, _ = precision_recall_fscore_support(y_eval, pred, average="binary")
print(f"Precision={p:.4f}")
print(f"Recall={r:.4f}")
print(f"F1={f1:.4f}")