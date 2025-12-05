# llm_python_codegen
fine tuned llm (phi2) to generate and debug python codes 
Python LLM Finetuning (QLoRA + LoRA)

This repository contains a complete pipeline for finetuning Large Language Models (LLMs) on Python code using PEFT, LoRA, and QLoRA.
It includes scripts for:

Dataset preparation & subsampling

Tokenization

QLoRA/LoRA training using HuggingFace Trainer

Saving adapters or merging weights

Inference & evaluation

This project works on GPU systems, including Windows (WSL), Linux, and containers.

🚀 Quickstart
1. Create & activate virtual environment
python -m venv llm_env
# Linux/macOS:
source llm_env/bin/activate
# Windows PowerShell:
llm_env\Scripts\activate.ps1

2. Install dependencies
pip install -r requirements.txt

3. Tokenize dataset
python scripts/tokenize.py \
  --input data/raw/python.jsonl \
  --output data/tokenized_python \
  --tokenizer <BASE_MODEL_NAME_OR_PATH>

4. Train (QLoRA example)
accelerate launch --config_file accelerate_config.yaml \
scripts/train_qlora.py \
  --model_name_or_path <PATH_TO_BASE_OR_FINETUNED_MODEL> \
  --data_dir data/tokenized_python \
  --output_dir outputs/python_adapter \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --lora_r 16

5. Run inference
python scripts/generate.py \
  --model outputs/python_adapter \
  --prompt_file examples/prompts.txt

📁 Project Structure
.
├── README.md
├── requirements.txt
├── accelerate_config.yaml
├── scripts/
│   ├── tokenize.py
│   ├── train_qlora.py
│   ├── train_lora_trainer.py
│   ├── generate.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── tokenized/
├── outputs/
└── examples/

📦 Dependencies

Include these in your requirements.txt:

transformers>=4.35
datasets
accelerate
bitsandbytes
peft
torch
safetensors
huggingface_hub

🗂 Dataset Preparation Workflow

Load source dataset (JSONL, parquet, or HuggingFace dataset)

Convert to instruction or completion format
Example JSONL structure:

{"prompt": "### Instruction:\n<description>\n### Response:\n", "completion": "<python code>"}


Tokenize

Save tokenized dataset using save_to_disk()

✂️ Tokenization

Uses the same tokenizer as the model

Supports long sequences (1024–2048 tokens)

Optionally masks prompt tokens during training

Example label masking inside tokenize.py:

labels = [-100] * prompt_length + input_ids[prompt_length:]

🧠 Training (QLoRA / LoRA)

Key features:

load_in_8bit=True for QLoRA

LoraConfig applied to attention layers

HuggingFace Trainer used for training loop

Supports multi-GPU via accelerate

Recommended settings:

learning_rate: 2e-4
lora_rank (r): 16
lora_alpha: 32
batch_size: 1–4 (with gradient accumulation)
max_length: 1024
epochs: 1–3

📝 Saving & Merging Models
Save LoRA adapter
model.save_pretrained("outputs/python_adapter")

Merge LoRA → full model
model = model.merge_and_unload()
model.save_pretrained("outputs/merged_python_model")

🔍 Inference
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("outputs/python_adapter")
model = AutoModelForCausalLM.from_pretrained("outputs/python_adapter")

prompt = "Write a Python function to check if a number is prime."
inputs = tok(prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(output[0], skip_special_tokens=True))

⚙️ accelerate_config.yaml (example)
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: fp16
num_processes: 1
num_machines: 1
use_cpu: false

🧹 Git Tips

Ignoring your virtual environment:

llm_env/


If you get “no upstream branch” error:

git push --set-upstream origin master

🧪 Hyperparameter Recommendations
Component	Suggested value
Learning Rate	1e-4 to 2e-4
LoRA Rank (r)	8–32
LoRA Alpha	16–32
Batch Size	1–4
Max Length	1024 or 2048
Epochs	1–3
🛠 Common Issues
Issue	Fix
CUDA OOM	Lower batch size, reduce sequence length
Tokenizer mismatch	Use same tokenizer as model
Slow training	Use fp16, QLoRA, gradient accumulation
ImportError: DownloadConfig	Remove legacy imports; not needed
📌 License
MIT License
