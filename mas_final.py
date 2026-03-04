import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =============================================================================
# ARCHITECTURE:
#   Advocate  → base Mistral-7B-Instruct + QLoRA adapter  (domain-adapted)
#   Skeptic   → base Mistral-7B-Instruct only             (clean generation)
#   Judge     → base Mistral-7B-Instruct only             (clean generation)
#
# WHY THE ADAPTER IS ONLY ON THE ADVOCATE:
# -----------------------------------------
# The QLoRA adapter was trained as a CLASSIFIER on PatentsBERTa-style data,
# meaning it learned to map patent claim text → a fixed output phrase.
# Loading it onto the Skeptic or Judge corrupts their generation, causing
# 100% parse failures. The Advocate can tolerate it since even a templated
# green argument is directionally correct — the Judge (base model) makes
# the final call.
# =============================================================================


# ----------------------------
# GREEN DEFINITION
# ----------------------------

GREEN_DEFINITION = """A claim is GREEN (label=1) if it explicitly describes a technology with a clear environmental benefit or energy-saving/waste-reducing mechanism:
- Emissions reduction, exhaust after-treatment, or carbon capture
- Pollution control, filtration, or pollutant removal
- Recycling, waste processing, or circular economy mechanisms
- Water treatment, purification, or desalination
- Renewable energy generation, energy storage, or grid integration
- Explicit energy-saving mechanisms (not just the word "efficient")
- Monitoring that explicitly reduces waste or spoilage

A claim is NOT GREEN (label=0) if:
- It is general engineering or transport optimization with no explicit environmental mechanism
- It uses vague terms like "eco-friendly" or "efficient" without describing a mechanism
- It involves generic electrification without an explicit environmental objective
- It is a military or weapon-related system"""


# ----------------------------
# Prompt Builders — Mistral [INST] format
# ----------------------------

def build_advocate_prompt(claim_text: str) -> str:
    instruction = f"""You are Agent 1 (The Advocate). Argue that this patent claim IS GREEN (label=1).

{GREEN_DEFINITION}

Rules:
- Use ONLY evidence explicitly stated in the claim text.
- Quote short phrases from the claim to support your argument.
- If the claim has a genuine explicit environmental mechanism, state it clearly.
- If the evidence is genuinely weak, say so honestly rather than fabricating arguments.
- Give 3-5 sentences. Do NOT output JSON.

CLAIM:
{claim_text}

ADVOCATE ARGUMENT:"""
    return f"<s>[INST] {instruction.strip()} [/INST]"


def build_skeptic_prompt(claim_text: str) -> str:
    instruction = f"""You are Agent 2 (The Skeptic). Argue that this patent claim is NOT GREEN (label=0).

{GREEN_DEFINITION}

Rules:
- Use ONLY evidence (or lack thereof) from the claim text.
- Challenge vague efficiency claims or electrification without explicit environmental purpose.
- If the claim DOES contain a genuinely clear environmental mechanism, acknowledge it honestly.
- Give 3-5 sentences. Do NOT output JSON.

CLAIM:
{claim_text}

SKEPTIC ARGUMENT:"""
    return f"<s>[INST] {instruction.strip()} [/INST]"


def build_judge_prompt(claim_text: str, advocate: str, skeptic: str) -> str:
    instruction = f"""You are Agent 3 (The Judge). Decide the final label for this patent claim.

{GREEN_DEFINITION}

Calibration examples:
Example 1 (NOT_GREEN): "A system using an armored vehicle and electrode to detect and ignite buried explosive devices."
Output: {{"label": 0, "confidence": "high", "rationale": "Military weapon system with no environmental mechanism."}}

Example 2 (GREEN): "A time-temperature indicator monitoring cumulative ambient temperature exposure during storage to prevent spoilage."
Output: {{"label": 1, "confidence": "high", "rationale": "Explicitly monitors temperature to prevent spoilage — a direct waste-reduction mechanism."}}

Example 3 (NOT_GREEN): "An optimized gear assembly providing enhanced torque and fuel efficiency."
Output: {{"label": 0, "confidence": "medium", "rationale": "Fuel efficiency is mentioned but no explicit environmental mechanism is described."}}

Judge rules:
- Weigh both arguments carefully and base your decision ONLY on the claim text.
- label=1 if the claim explicitly describes an environmental mechanism.
- label=0 if the claim is vague, generic, or lacks an explicit mechanism.
- Do NOT default to 0 — both outcomes are equally valid.
- Output ONLY valid JSON. No markdown, no explanation outside JSON.
- JSON must start with {{ and end with }}.
- Keys: label (integer 0 or 1), confidence ("low"/"medium"/"high"), rationale (1-3 sentences).

CLAIM:
{claim_text}

ADVOCATE ARGUMENT:
{advocate}

SKEPTIC ARGUMENT:
{skeptic}

JSON:"""
    return f"<s>[INST] {instruction.strip()} [/INST]"


# ----------------------------
# Utilities
# ----------------------------

def clean_json_block(text: str) -> str:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    label_m = re.search(r'"label"\s*:\s*([01])', text)
    conf_m  = re.search(r'"confidence"\s*:\s*"(low|medium|high)"', text)
    rat_m   = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
    if label_m:
        label = int(label_m.group(1))
        conf  = conf_m.group(1) if conf_m else "low"
        rat   = rat_m.group(1) if rat_m else "No rationale extracted."
        return json.dumps({"label": label, "confidence": conf, "rationale": rat})
    return text.strip()


def safe_json_load(s: str):
    s = clean_json_block(s)
    try:
        return json.loads(s)
    except Exception:
        return None


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 350):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return text[len(prompt_decoded):].strip()


# ----------------------------
# Main
# ----------------------------

def main():
    BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
    ADAPTER_DIR = "/ceph/home/student.aau.dk/gs62rg"
    INPUT_CSV   = "/ceph/home/student.aau.dk/gs62rg/hitl_green_100.csv"
    OUTPUT_CSV  = "/ceph/home/student.aau.dk/gs62rg/hitl_green_100_mas.csv"
    TEXT_COL    = "text"

    # ----------------------------------------------------------------
    # SET THIS TO None TO RUN ALL 100 ROWS
    # Set to 10 or 20 for a quick sanity check
    # ----------------------------------------------------------------
    MAX_ROWS = None

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.eval()
    print("Base model ready (Skeptic + Judge).")

    print(f"Loading QLoRA adapter from: {ADAPTER_DIR}")
    advocate_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    advocate_model.eval()
    print("Advocate model ready (base + QLoRA adapter).\n")

    df = pd.read_csv(INPUT_CSV)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Available: {df.columns.tolist()}")

    # Cap rows if MAX_ROWS is set
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()
        print(f"TEST MODE: Running on first {MAX_ROWS} rows only.\n")
    else:
        print(f"FULL MODE: Running on all {len(df)} rows.\n")

    df["agent1_advocate"]     = ""
    df["agent2_skeptic"]      = ""
    df["judge_json"]          = ""
    df["llm_green_suggested"] = pd.NA
    df["llm_confidence"]      = ""
    df["llm_rationale"]       = ""
    df["parse_failed"]        = False

    parse_fail_count = 0

    for i in range(len(df)):
        claim = str(df.loc[i, TEXT_COL])

        # Agent 1: Advocate — domain-adapted (base + QLoRA adapter)
        advocate  = generate(advocate_model, tokenizer, build_advocate_prompt(claim), max_new_tokens=300)

        # Agent 2: Skeptic — clean base model
        skeptic   = generate(base_model, tokenizer, build_skeptic_prompt(claim), max_new_tokens=300)

        # Agent 3: Judge — clean base model (critical for JSON integrity)
        judge_raw = generate(base_model, tokenizer, build_judge_prompt(claim, advocate, skeptic), max_new_tokens=350)

        df.loc[i, "agent1_advocate"] = advocate
        df.loc[i, "agent2_skeptic"]  = skeptic
        df.loc[i, "judge_json"]      = judge_raw

        # Always print all claims in test mode, first 3 in full mode
        if MAX_ROWS is not None or i < 3:
            print(f"\n{'='*60}")
            print(f"CLAIM {i+1}:\n{claim[:300]}\n")
            print(f"--- ADVOCATE (adapter) ---\n{advocate}\n")
            print(f"--- SKEPTIC (base) ---\n{skeptic}\n")
            print(f"--- JUDGE RAW (base) ---\n{judge_raw}\n")

        j = safe_json_load(judge_raw)

        if j is None:
            parse_fail_count += 1
            print(f"[WARNING] Claim {i+1}: Could not parse JSON. Raw: {judge_raw[:200]}")
            df.loc[i, "llm_green_suggested"] = pd.NA
            df.loc[i, "llm_confidence"]      = "low"
            df.loc[i, "llm_rationale"]       = f"PARSE FAILED. Raw: {judge_raw[:300]}"
            df.loc[i, "parse_failed"]        = True
        else:
            label = j.get("label", None)
            conf  = j.get("confidence", "low")
            rat   = j.get("rationale", "")

            try:
                label = int(label)
            except Exception:
                label = None

            if label not in [0, 1]:
                print(f"[WARNING] Claim {i+1}: Invalid label '{label}'. Marking for review.")
                df.loc[i, "llm_green_suggested"] = pd.NA
                df.loc[i, "parse_failed"]        = True
            else:
                df.loc[i, "llm_green_suggested"] = label

            df.loc[i, "llm_confidence"] = conf if conf in ["low", "medium", "high"] else "low"
            df.loc[i, "llm_rationale"]  = rat if isinstance(rat, str) else str(rat)

        if (i + 1) % 10 == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            green     = (df["llm_green_suggested"] == 1).sum()
            not_green = (df["llm_green_suggested"] == 0).sum()
            print(f"Progress: {i+1}/{len(df)} | GREEN: {green} | NOT GREEN: {not_green} | Parse failures: {parse_fail_count}")

    df.to_csv(OUTPUT_CSV, index=False)

    total     = len(df)
    green     = (df["llm_green_suggested"] == 1).sum()
    not_green = (df["llm_green_suggested"] == 0).sum()
    failed    = df["parse_failed"].sum()

    print(f"\n{'='*60}")
    print(f"DONE. Saved: {OUTPUT_CSV}")
    print(f"Total: {total} | GREEN: {green} | NOT GREEN: {not_green} | Parse failed: {failed}")
    if failed > 10:
        print("NOTE: Some parse failures — check judge_json column for raw outputs.")


if __name__ == "__main__":
    main()
