# qlora_inference_10k.py
# ------------------------------------------------------------
# QLoRA inference matching your training (JSON {label, rationale}).
# - OPTIONAL eval_silver (can skip with --skip_eval)
# - HITL-100 predictions + gold scoring + saves qlora_final.csv
# - Progress bars for eval_silver + HITL-100 via tqdm
#
# Run HITL only:
#   python3 qlora_inference_10k.py --adapter_dir qlora_mistral_adapter_json --skip_eval
#
# Install (once):
#   pip install -U tqdm scikit-learn transformers peft bitsandbytes accelerate pyarrow

import re
import json
import argparse
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# Match the FIRST JSON object in the GENERATED text
JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# Fallback: find a label even if JSON is broken
LABEL_RE = re.compile(r'("label"\s*:\s*([01]))|(\b([01])\b)')


def extract_json_from_generated(gen_text: str):
    """
    Robust JSON extraction:
    1) Try to parse first {...}
    2) If missing braces, wrap and try again
    """
    m = JSON_RE.search(gen_text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    candidate = gen_text.strip()
    if not candidate:
        return None

    if not candidate.startswith("{"):
        candidate = "{" + candidate
    if not candidate.endswith("}"):
        candidate = candidate + "}"

    try:
        return json.loads(candidate)
    except Exception:
        return None


def extract_label_fallback(gen_text: str):
    """If JSON parsing fails, try to find label=0/1 in the generated text."""
    m = LABEL_RE.search(gen_text)
    if not m:
        return None
    if m.group(2) in ("0", "1"):
        return int(m.group(2))
    if m.group(4) in ("0", "1"):
        return int(m.group(4))
    return None


def to_int01(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "green", "yes", "y"}:
            return 1
        if s in {"0", "false", "not_green", "no", "n"}:
            return 0
        try:
            return 1 if int(float(s)) == 1 else 0
        except Exception:
            return None
    try:
        return 1 if int(v) == 1 else 0
    except Exception:
        return None


# ✅ REPLACED with YOUR training prompt (exact)
def build_chat_prompt(tokenizer, claim_text: str) -> str:
    """
    Uses the model's chat template (better for Mistral-Instruct).
    Forces JSON output with label 0/1 + rationale.
    """
    system = (
        "You are a patent-claims classifier. "
        "Return ONLY valid JSON with exactly these keys: label, rationale. "
        "label must be 0 or 1. Do not output any other text."
    )
    user = (
        "Decide if the claim describes environmentally beneficial (green) technology.\n\n"
        'Return ONLY JSON like:\n{"label": 1, "rationale": "..."}\n\n'
        "Rules:\n"
        "- label: 1 = GREEN, 0 = NOT_GREEN\n"
        "- rationale: 1-2 short sentences\n\n"
        f"CLAIM:\n{claim_text}\n\n"
        "JSON:"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_and_tokenizer(base_model: str, adapter_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict_one(model, tokenizer, claim_text: str, max_new_tokens: int = 96):
    """
    Deterministic JSON generation.
    IMPORTANT: decode ONLY generated tokens, not the full prompt.
    """
    prompt = build_chat_prompt(tokenizer, claim_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # temperature is ignored when do_sample=False (warning is fine)
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # decode only the newly generated tokens
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    obj = extract_json_from_generated(gen_text)
    return gen_text, obj


def get_label_and_rationale(gen_text: str, obj):
    """
    Returns (label:int or None, rationale:str or None, parse_ok:bool)
    parse_ok=True means clean JSON parse with valid label.
    """
    if obj and isinstance(obj, dict) and "label" in obj:
        try:
            lab = int(obj["label"])
            if lab in (0, 1):
                rat = obj.get("rationale", None)
                return lab, (str(rat) if rat is not None else None), True
        except Exception:
            pass

    lab2 = extract_label_fallback(gen_text)
    if lab2 in (0, 1):
        return lab2, None, False

    return None, None, False


def evaluate_df(df: pd.DataFrame, model, tokenizer, text_col: str, ytrue_col: str,
                max_rows: int | None, title: str, max_new_tokens: int):
    df = df[[text_col, ytrue_col]].dropna().copy()
    df["y_true"] = df[ytrue_col].apply(to_int01)
    df = df.dropna(subset=["y_true"]).copy()
    df["y_true"] = df["y_true"].astype(int)

    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    print(f"\n=== {title} ===")
    print("Rows:", len(df))
    print("Label distribution:\n", df["y_true"].value_counts())

    y_true = df["y_true"].tolist()
    y_pred = []
    bad = 0
    fallback = 0

    texts = df[text_col].astype(str).tolist()
    for t in tqdm(texts, desc="eval_silver", total=len(texts)):
        gen_text, obj = predict_one(model, tokenizer, t, max_new_tokens=max_new_tokens)
        lab, _, ok = get_label_and_rationale(gen_text, obj)

        if lab is None:
            y_pred.append(0)
            bad += 1
        else:
            y_pred.append(lab)
            if not ok:
                fallback += 1

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)  # GREEN=1
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Metrics ---")
    print("Bad/unparsed (defaulted to 0):", bad, "out of", len(y_true))
    print("Fallback (label extracted without clean JSON):", fallback, "out of", len(y_true))
    print("Accuracy:", acc)
    print("F1 (GREEN=1):", f1)
    print("Precision:", prec)
    print("Recall:", rec)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))


def predict_hitl_100(df100: pd.DataFrame, model, tokenizer,
                     id_col: str, text_col: str, gold_col: str,
                     out_csv: str, max_new_tokens: int):
    """
    Saves predictions + compares to gold labels (gold_col).
    Includes progress bar for the 100 items.
    """
    if id_col not in df100.columns:
        raise ValueError(f"HITL-100 missing '{id_col}'. Available: {list(df100.columns)}")
    if text_col not in df100.columns:
        raise ValueError(f"HITL-100 missing '{text_col}'. Available: {list(df100.columns)}")
    if gold_col not in df100.columns:
        raise ValueError(f"HITL-100 missing gold column '{gold_col}'. Available: {list(df100.columns)}")

    out = df100.copy()
    out["gold01"] = out[gold_col].apply(to_int01)

    preds = []
    rationales = []
    raws = []
    parse_ok = []
    bad = 0
    fallback = 0

    texts = out[text_col].astype(str).tolist()
    for t in tqdm(texts, desc="HITL-100", total=len(texts)):
        gen_text, obj = predict_one(model, tokenizer, t, max_new_tokens=max_new_tokens)
        lab, rat, ok = get_label_and_rationale(gen_text, obj)

        raws.append(gen_text)
        parse_ok.append(ok)

        if lab is None:
            preds.append(None)
            rationales.append(None)
            bad += 1
        else:
            preds.append(lab)
            rationales.append(rat)
            if not ok:
                fallback += 1

    out["qlora_label"] = preds
    out["qlora_rationale"] = rationales
    out["qlora_raw"] = raws
    out["qlora_json_ok"] = parse_ok

    out.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)
    print("Bad (no label):", bad, "out of", len(out))
    print("Fallback (no clean JSON):", fallback, "out of", len(out))

    eval_rows = out.dropna(subset=["gold01", "qlora_label"]).copy()
    if len(eval_rows) == 0:
        print("No rows available to score (check gold_col / predictions).")
        return

    y_true = eval_rows["gold01"].astype(int).tolist()
    y_pred = eval_rows["qlora_label"].astype(int).tolist()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== HITL-100 Metrics (against gold) ===")
    print("Rows scored:", len(eval_rows), "/", len(out))
    print("Accuracy:", acc)
    print("F1 (GREEN=1):", f1)
    print("Precision:", prec)
    print("Recall:", rec)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--file_50k", type=str, default="patents_50k_green.parquet")
    ap.add_argument("--file_hitl100", type=str, default="hitl_green_100_llm_corrected.csv")

    ap.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--adapter_dir", type=str, required=True)

    # Optional eval on silver
    ap.add_argument("--skip_eval", action="store_true", help="Skip eval_silver evaluation (recommended for speed)")
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--eval_split_value", type=str, default="eval_silver")
    ap.add_argument("--text_col_50k", type=str, default="text")
    ap.add_argument("--label_col_50k", type=str, default="is_green_silver")
    ap.add_argument("--eval_max_rows", type=int, default=2000)

    # HITL
    ap.add_argument("--id_col_hitl", type=str, default="doc_id")
    ap.add_argument("--text_col_hitl", type=str, default="text")
    ap.add_argument("--gold_col_hitl", type=str, default="is_green_human")

    # Outputs / speed
    ap.add_argument("--out_csv_100", type=str, default="qlora_final.csv")
    ap.add_argument("--max_new_tokens_eval", type=int, default=64)
    ap.add_argument("--max_new_tokens_100", type=int, default=96)

    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_dir)

    if not args.skip_eval:
        df50 = pd.read_parquet(args.file_50k)
        split = df50[args.split_col].astype(str).str.lower().str.strip()
        df_eval = df50[split == args.eval_split_value.lower()].copy()

        evaluate_df(
            df=df_eval,
            model=model,
            tokenizer=tokenizer,
            text_col=args.text_col_50k,
            ytrue_col=args.label_col_50k,
            max_rows=args.eval_max_rows,
            title=f"Eval on {args.eval_split_value} (silver labels)",
            max_new_tokens=args.max_new_tokens_eval,
        )
    else:
        print("Skipping eval_silver (--skip_eval).")

    df100 = pd.read_csv(args.file_hitl100)
    predict_hitl_100(
        df100=df100,
        model=model,
        tokenizer=tokenizer,
        id_col=args.id_col_hitl,
        text_col=args.text_col_hitl,
        gold_col=args.gold_col_hitl,
        out_csv=args.out_csv_100,
        max_new_tokens=args.max_new_tokens_100,
    )


if __name__ == "__main__":
    main()
