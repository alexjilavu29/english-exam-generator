import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os


def load_gemma_model():
    load_dotenv()

    token = os.getenv("TOKEN")
    login(token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "google/gemma-7b"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model... (this may take a while)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    return tokenizer, model, device


def generate_response(prompt, tokenizer, model, device):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **input_ids,
        max_new_tokens=500,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


tokenizer, model, device = load_gemma_model()

print("\nGemma 7B is ready! Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")

    if prompt.lower() == 'quit':
        break

    print("\nGemma is thinking...\n")
    response = generate_response(prompt, tokenizer, model, device)
    print(f"Gemma: {response}\n")
