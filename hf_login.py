from getpass import getpass
from huggingface_hub import login

print("Hugging Face Login (optional)")
token = getpass("Enter HuggingFace token (or press Enter to skip): ")

if token.strip():
    login(token)
    print("✅ Logged in to Hugging Face Hub.")
else:
    print("⏩ Skipped login. You can still use local models.")
