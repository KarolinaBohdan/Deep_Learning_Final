import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Config
# -----------------------------
INPUT = "hitl_green_100.csv"
OUTPUT = "hitl_green_100_llm.csv"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "You are labeling patent claims as GREEN technology (1) or NOT GREEN (0).\n"
    "Rule: Use ONLY the claim text. Do NOT use CPC codes or metadata.\n\n"
    "Return STRICTLY in this format:\n"
    "label: 0 or 1\n"
    "confidence: low or medium or high\n"
    "rationale: 1-3 sentences citing phrases from the claim text\n"
)

def build_prompt(text: str) -> str:
    return f"{SYSTEM_PROMPT}\nCLAIM TEXT:\n{text}\n"

def parse_output(generated_text: str):
    label = ""
    conf = ""
    rationale = ""

    for line in generated_text.splitlines():
        s = line.strip()
        if s.lower().startswith("label:"):
            label = s.split(":", 1)[1].strip()
        elif s.lower().startswith("confidence:"):
            conf = s.split(":", 1)[1].strip().lower()
        elif s.lower().startswith("rationale:"):
            rationale = s.split(":", 1)[1].strip()

    if label not in {"0", "1"}:
        label = "0"
    if conf not in {"low", "medium", "high"}:
        conf = "low"
    if not rationale:
        rationale = generated_text.strip()[:300]

    return label, conf, rationale

# -----------------------------
# Load CSV safely (all strings)
# -----------------------------
df = pd.read_csv(INPUT, dtype=str).fillna("")

# Force HITL columns to plain object dtype (most permissive)
for col in ["llm_green_suggested", "llm_confidence", "llm_rationale", "is_green_human", "human_notes"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(object)

# -----------------------------
# Label loop
# -----------------------------
for i in tqdm(range(len(df))):

    if df.loc[i, "llm_green_suggested"] in {"0", "1"} and df.loc[i, "llm_confidence"] != "" and df.loc[i, "llm_rationale"] != "":
        continue

    text = str(df.loc[i, "text"])
    prompt = build_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            num_beams=1
        )

    # ✅ Keep only newly generated tokens (remove prompt tokens)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    label, conf, rationale = parse_output(generated_text)

    df.loc[i, "llm_green_suggested"] = str(label)
    df.loc[i, "llm_confidence"] = str(conf)
    df.loc[i, "llm_rationale"] = str(rationale)

    if i % 5 == 0:
        df.to_csv(OUTPUT, index=False)

df.to_csv(OUTPUT, index=False)
print("Saved:", OUTPUT)