'''
Test models, with MultiModelWithScalarHeads.generate() from architecture.py
'''
import torch
from architecture import MultiModelWithScalarHeads
from model_loader import ModelLoader
from models import MODELS  # must point to a list of ModelInfo objects

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the architecture with random projection heads
    model = MultiModelWithScalarHeads(
        base_models=MODELS,
        model_loader=ModelLoader(device=device),
        device=device
    )

    prompt = """You will receive a question and you'll have to provide a sensible answer for it, but reason a bit beforehand. All results should be explicit. You prompt should start with "Ok, let's reason about that:
    Question: A kilogram of pork costs $6 while a kilogram of chicken costs $2 less. How much will a 3-kilogram of chicken and a kilogram of pork cost?"""
    result = model.generate(prompt, max_new_tokens=100)

    print("\n=== Prompt ===")
    print(prompt)

    print("\n=== Generated Output ===")
    print(result["generated_text"])

    print("\n=== Generation Trace ===")
    for i, step in enumerate(result["steps"]):
        print(f"Step {i+1}: Token = '{step['token']}', Model = {step['model_id']}")

if __name__ == "__main__":
    main()
