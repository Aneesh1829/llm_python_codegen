from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "phi2_lora_out_small"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"   # automatically uses GPU
)

while True:
    prompt = input("\nYour prompt: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    print("\n=== Model Output ===")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
