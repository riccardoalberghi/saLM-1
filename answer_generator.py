#!/usr/bin/env python3
"""
Answer Generator Script

This script takes a question as a CLI argument, generates an answer using a multi-model ensemble,
and saves the confidence scores for each token of the answer for each model in a JSON file.
"""

import argparse
import json
import os
import torch
import sys
from typing import Dict, List

# Add the src directory to the path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if not script_dir in sys.path:
    sys.path.append(script_dir)

from src.architecture import MultiModelWithScalarHeads
from src.model_loader import ModelLoader
from src.models import MODELS

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate answer to a question with model confidence scores.')
    parser.add_argument('question', help='The question to answer')
    parser.add_argument('--output', '-o', default='answer_confidence.json', 
                        help='Output JSON file path (default: answer_confidence.json)')
    parser.add_argument('--max-tokens', '-m', type=int, default=200,
                        help='Maximum number of tokens to generate (default: 200)')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                        help='Temperature for text generation (default: 0.7)')
    parser.add_argument('--weights-path', '-w', 
                        default=os.path.join(script_dir, "weights", "best_model_heads.pt"),
                        help='Path to model weights file')
    return parser.parse_args()

def construct_prompt(question: str) -> str:
    """Construct a prompt for the model based on the question."""
    return f"""You will receive a question and you'll have to provide a sensible answer for it, but reason a bit beforehand. All results should be explicit. 

Question: {question}

Let me reason about this:"""

def process_generation_results(model, generation_result, prompt):
    """
    Process generation results and extract confidence scores.
    
    Args:
        model: The MultiModelWithScalarHeads model
        generation_result: Result from model.generate with return_decisions=True
        prompt: The original prompt
        
    Returns:
        Dict with question, answer, and token data with confidence scores
    """
    # Extract the generated text and steps
    generated_text = generation_result["generated_text"]
    steps = generation_result["steps"]
    
    # Get model IDs for reference
    model_ids = model.model_ids
    
    # Process each token step to add confidence scores
    tokens_with_confidence = []
    
    for step in steps:
        token = step["token"]
        token_id = step["token_id"]
        best_model_id = step["model_id"]
        
        # Create token data with the information we already have
        token_data = {
            "token": token,
            "token_id": token_id,
            "best_model": best_model_id,
            # We'll add confidence scores later
        }
        
        tokens_with_confidence.append(token_data)
    
    # Return the complete result
    return {
        "question": prompt,
        "answer": generated_text,
        "tokens": tokens_with_confidence
    }

def main():
    """Main function to generate answer and save confidence scores."""
    args = parse_arguments()
    
    # Check if weights file exists
    if not os.path.exists(args.weights_path):
        print(f"Warning: Weights file not found at {args.weights_path}")
        print("Will continue with default initialization.")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Construct prompt from question
    prompt = construct_prompt(args.question)
    print(f"Processing question: {args.question}")
    
    try:
        # Initialize the model
        model = MultiModelWithScalarHeads.from_pretrained(
            base_models_info=MODELS,
            weights_path=args.weights_path,
            head_hidden_dim=256,
            head_dropout=0.1,
            model_loader=ModelLoader(device=device),
            device=device
        )
        
        # Generate answer using the built-in generate method
        generation_result = model.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            return_decisions=True,
            return_token_ids=False  # Return decoded text instead of token IDs
        )
        
        # Process the generation results to extract and add confidence scores
        result = process_generation_results(model, generation_result, prompt)
        
        # Now we need to process the tokens to add confidence scores
        # For simplicity, we'll re-process the prompt and each token to get the model confidences
        
        # First, prepare the tokenizer
        tokenizer = model.common_tokenizer
        
        # Process the initial prompt
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(device)
        original_input_length = tokenized_input.input_ids.shape[1]
        
        # The full text includes the prompt plus all generated tokens
        full_text = prompt + "".join([t["token"] for t in result["tokens"]])
        
        # Process the full text to get confidence scores for all positions
        with torch.no_grad():
            model.eval()
            outputs = model([full_text], use_cache=False)
            all_raw_scores = outputs["all_raw_scores"]
            model_confidence = model.get_model_probs(all_raw_scores)
            
            # Extract confidence scores for each generated token
            for i, token_data in enumerate(result["tokens"]):
                # The position in the sequence is original_input_length + i
                pos = original_input_length + i
                if pos < model_confidence.shape[1]:  # Check if position is within bounds
                    # Get confidence scores for all models at this position
                    position_confidences = {
                        model_id: model_confidence[0, pos, idx].item()
                        for idx, model_id in enumerate(model.model_ids)
                    }
                    # Add to token data
                    token_data["confidence_scores"] = position_confidences
        
        # Save results to JSON file
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nGenerated answer:\n{result['answer']}")
        print(f"\nConfidence scores saved to: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
