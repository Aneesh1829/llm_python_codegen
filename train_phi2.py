#!/usr/bin/env python
"""
Train Phi-2 with QLoRA (4-bit) + PEFT
Optimized for small GPUs and clean training.
"""

import argparse
import os
from pathlib import Path
import torch
import gc
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["WANDB_DISABLED"] = "true"


# ------------------------------------------------------------
# Load tokenized dataset
# ------------------------------------------------------------
def load_tokenized(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tokenized dataset not found at {path}. Run prepare_tokenize.py first.")
    return load_from_disk(str(p))


# ------------------------------------------------------------
# Build and quantize Phi-2 model
# ------------------------------------------------------------
def load_model(model_name, trust_remote_code=False):
    print("Loading Phi-2 model:", model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code
    )

    print("Model loaded successfully!")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    print("Loading tokenized dataset...")
    tokenized = load_tokenized(args.tokenized_dir)

    print(f"Dataset size: {len(tokenized)} samples")

    model = load_model(args.model_name)

    # LoRA config (Phi-2 is small → use small LoRA)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("\n===== START TRAINING =====\n")

    try:
        trainer.train()
        print("\nTraining finished. Saving model...\n")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

    except torch.cuda.OutOfMemoryError:
        print("\n❌ CUDA Out of Memory! Reduce batch size or accumulation steps.")
        raise

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phi-2 using QLoRA")
    parser.add_argument("--model_name", default="microsoft/phi-2")
    parser.add_argument("--tokenizer_dir", default="data/tokenizer")
    parser.add_argument("--tokenized_dir", default="data/tokenized")
    parser.add_argument("--output_dir", default="phi2_lora_out")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
