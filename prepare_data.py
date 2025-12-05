#!/usr/bin/env python

"""
prepare_data.py

Converts a JSONL dataset with instruction / input / output
into a clean training file:
{
    "prompt": "...",
    "target": "..."
}

Works for code generation & debugging datasets.
"""

import json
import argparse
from pathlib import Path


def build_prompt(ex):
    """
    Builds the instruction-format prompt.
    If input is empty, we skip it.
    """

    instruction = ex.get("instruction", "").strip()
    input_text = ex.get("input", "").strip()
    output_text = ex.get("output", "").strip()

    if input_text:
        prompt = f"""Below is a coding task along with additional context.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is a coding task.

### Instruction:
{instruction}

### Response:
"""

    return prompt, output_text


def process_file(in_file, out_file):
    in_path = Path(in_file)
    out_path = Path(out_file)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    print(f"Reading: {in_file}")

    processed = []

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prompt, target = build_prompt(ex)

            processed.append({
                "prompt": prompt,
                "target": target
            })

    print(f"Total processed examples: {len(processed)}")

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in processed:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved processed dataset to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for instruction tuning.")
    parser.add_argument("--input_file", default="data/raw/train.jsonl")
    parser.add_argument("--output_file", default="data/train.jsonl")

    args = parser.parse_args()

    process_file(args.input_file, args.output_file)
