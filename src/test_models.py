'''
Test models, with MultiModelWithScalarHeads.generate() from architecture.py
Using pretrained model weights
'''
import torch
from architecture import MultiModelWithScalarHeads
from model_loader import ModelLoader
from models import MODELS  # must point to a list of ModelInfo objects
import os

# Get the absolute path to the weights file
# Assuming your script is in the src directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to nlp/
weights_path = os.path.join(project_root, "weights", "best_model_heads.pt")

print(f"Looking for weights at: {weights_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Load the pretrained model using the from_pretrained class method
    model = MultiModelWithScalarHeads.from_pretrained(
        base_models_info=MODELS,
        weights_path=weights_path,
        head_hidden_dim=256,  # Match the training configuration
        head_dropout=0.1,     # Match the training configuration
        model_loader=ModelLoader(device=device),
        device=device
    )

    prompt = """You will receive a question and you'll have to provide a sensible answer for it, but reason a bit beforehand. All results should be explicit. You prompt should start with "Ok, let's reason about that:
    Question: A kilogram of pork costs $6 while a kilogram of chicken costs $2 less. How much will a 3-kilogram of chicken and a kilogram of pork cost?"""
    
    # Call generate with return_decisions=True to get the generation trace
    result = model.generate(
        prompt=prompt, 
        max_new_tokens=200, 
        temperature=0.7,  # Adjust temperature as needed
        return_decisions=True,
        return_token_ids=False  # Return decoded text instead of token IDs
    )

    print("\n=== Prompt ===")
    print(prompt)

    print("\n=== Generated Output ===")
    print(result["generated_text"])

    print("\n=== Generation Trace ===")
    for i, step in enumerate(result["steps"]):
        print(f"Step {i+1}: Token = '{step['token']}', Model = {step['model_id']}")

if __name__ == "__main__":
    main()