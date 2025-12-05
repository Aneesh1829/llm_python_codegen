#!/usr/bin/env python
import argparse
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

# ------------------------------------------------------------
# Helpers: build prompt variants
# ------------------------------------------------------------
def build_prompt_from_fields(instruction, input_text):
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return (
            f"Below is a coding task and some additional context.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"Below is a coding task.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

# ------------------------------------------------------------
# Tokenize one batch (works for both dataset formats)
# ------------------------------------------------------------
def tokenize_batch(examples, tokenizer, max_length):
    # cases:
    # 1) dataset columns are 'prompt' and 'target' (already pre-built)
    # 2) dataset columns are 'instruction', 'input', 'output' (we must build)
    if "prompt" in examples and "target" in examples:
        prompts = [ (p or "").strip() for p in examples.get("prompt", []) ]
        targets = [ (t or "").strip() for t in examples.get("target", []) ]
    else:
        instrs = examples.get("instruction", [""] * len(next(iter(examples.values()), [])))
        inps = examples.get("input", [""] * len(instrs))
        outs = examples.get("output", [""] * len(instrs))
        prompts = [ build_prompt_from_fields(i, j) for i, j in zip(instrs, inps) ]
        targets = [ (o or "").strip() for o in outs ]

    # Build final texts: prompt + target
    full_texts = [p + t for p, t in zip(prompts, targets)]

    # Safety: if no examples in this batch, return empty structure expected by map
    if len(full_texts) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    toks = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    # labels = input_ids (causal LM)
    toks["labels"] = toks["input_ids"].copy()
    return toks

# ------------------------------------------------------------
# Safe tokenizer loader
# ------------------------------------------------------------
def load_tokenizer_safe(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        warnings.warn(f"Fast tokenizer load failed ({e}). Falling back to slow tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# ------------------------------------------------------------
def main(model_name, data_file, max_length, batch_size, out_dir):
    print("Loading tokenizer:", model_name)
    tokenizer = load_tokenizer_safe(model_name)

    print("Loading dataset:", data_file)
    ds = load_dataset("json", data_files=data_file)["train"]
    print("Dataset loaded:", len(ds), "samples")
    print("Columns:", ds.column_names)

    tokenized = ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        batch_size=batch_size,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    print("Tokenized keys:", tokenized.column_names)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenized_save = out_path / "tokenized"
    tokenizer_save = out_path / "tokenizer"
    tokenized.save_to_disk(str(tokenized_save))
    tokenizer.save_pretrained(str(tokenizer_save))
    print("Saved tokenized dataset to:", tokenized_save)
    print("Saved tokenizer to:", tokenizer_save)

# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset (supports prompt/target or instruction/input/output)")
    parser.add_argument("--model_name", default="microsoft/phi-2")
    parser.add_argument("--data_file", default="data/clean_train.jsonl")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()
    main(args.model_name, args.data_file, args.max_length, args.batch_size, args.out_dir)
