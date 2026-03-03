# qlora_train_mistral_ai_lab.py
# ------------------------------------------------------------
# QLoRA training (Mistral-7B-Instruct) on AI-Lab GPU using:
#  - patents_50k_green.parquet (train_silver split + is_green_silver labels)
#  - hitl_green_100_llm_corrected.csv (exclude those 100 IDs from training)
#
# Output format trained: JSON ONLY -> {"label": 0/1, "rationale": "..."}
#
# Install (if needed):
#   pip install -U "transformers>=4.41" datasets peft trl bitsandbytes accelerate pyarrow

import os
import json
import argparse
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def to_int01(v):
    # Robust conversion to strict 0/1 (cleans silver labels)
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


def read_table_auto(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


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
def silver_target_json(label01: int) -> str:
    # templated rationale (silver set has labels but not real rationales)
    if int(label01) == 1:
        obj = {
            "label": 1,
            "rationale": "The claim indicates a technology that provides an environmental benefit such as reducing emissions, saving energy, or improving resource efficiency."
        }
    else:
        obj = {
            "label": 0,
            "rationale": "The claim describes general technology without a clear environmental benefit such as reducing emissions, saving energy, or improving resource efficiency."
        }
    return json.dumps(obj, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()

    # Files
    ap.add_argument("--file_50k", type=str, default="patents_50k_green.parquet")
    ap.add_argument("--file_hitl100", type=str, default="hitl_green_100_llm_corrected.csv")

    # Columns (YOUR dataset)
    ap.add_argument("--id_col_50k", type=str, default="id")
    ap.add_argument("--text_col_50k", type=str, default="text")
    ap.add_argument("--silver_col_50k", type=str, default="is_green_silver")
    ap.add_argument("--split_col_50k", type=str, default="split")
    ap.add_argument("--train_split_value", type=str, default="train_silver")

    # HITL file columns
    ap.add_argument("--id_col_hitl", type=str, default="doc_id")

    # Model
    ap.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--out_dir", type=str, default="qlora_mistral_adapter_json")

    # Training knobs (UPDATED: 10k)
    ap.add_argument("--max_rows", type=int, default=10000)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)

    # Safety/resume
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Speed boost on modern GPUs (safe if unsupported)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    print("[1/8] Loading data...")
    df50 = read_table_auto(args.file_50k)
    df100 = read_table_auto(args.file_hitl100)

    # Checks
    for c in [args.id_col_50k, args.text_col_50k, args.silver_col_50k, args.split_col_50k]:
        if c not in df50.columns:
            raise ValueError(f"50k missing '{c}'. Available: {list(df50.columns)}")
    if args.id_col_hitl not in df100.columns:
        raise ValueError(f"HITL missing '{args.id_col_hitl}'. Available: {list(df100.columns)}")

    print("[2/8] Filtering train split (train_silver)...")
    split = df50[args.split_col_50k].astype(str).str.lower().str.strip()
    df_train = df50[split == args.train_split_value.lower()].copy()
    print("Train rows before exclusion:", len(df_train))

    print("[3/8] Excluding HITL-100 IDs (avoid leakage)...")
    hitl_ids = set(df100[args.id_col_hitl].astype(str).str.strip().tolist())
    train_ids = df_train[args.id_col_50k].astype(str).str.strip()
    before = len(df_train)
    df_train = df_train[~train_ids.isin(hitl_ids)].copy()
    print("Removed:", before - len(df_train))
    print("Train rows after exclusion:", len(df_train))

    print("[4/8] Cleaning labels + sampling 10k...")
    df_train = df_train[[args.text_col_50k, args.silver_col_50k]].dropna()
    df_train["label01"] = df_train[args.silver_col_50k].apply(to_int01)
    df_train = df_train.dropna(subset=["label01"])
    df_train["label01"] = df_train["label01"].astype(int)

    if args.max_rows and len(df_train) > args.max_rows:
        df_train = df_train.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)

    print("Final train rows:", len(df_train))
    print("Label distribution:\n", df_train["label01"].value_counts())

    print("[5/8] Loading tokenizer + building SFT dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def to_sft(row):
        prompt = build_chat_prompt(tokenizer, str(row[args.text_col_50k]))
        target = silver_target_json(int(row["label01"]))
        return prompt + target + tokenizer.eos_token  # EOS helps stop cleanly

    df_train["text"] = df_train.apply(to_sft, axis=1)
    print("\nExample training item (truncated):\n", df_train["text"].iloc[0][:900], "\n")
    train_ds = Dataset.from_pandas(df_train[["text"]])

    print("[6/8] Loading model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model.config.use_cache = False

    print("[7/8] LoRA + trainer config...")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_cfg = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_len,
        logging_steps=20,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        report_to="none",
        dataset_text_field="text",

        packing=True,
        gradient_checkpointing=True,

        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        peft_config=lora,
        args=sft_cfg,
    )

    print("[8/8] Training QLoRA adapters...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving final adapter + tokenizer...")
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"\n✅ DONE. Adapter saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
