import sys
import os
import re
import json
import argparse
from itertools import cycle
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm

# Model identifiers
MODELS = [
    "Qwen/Qwen2.5-0.5B",  # Small model
    "likewendy/Qwen2.5-3B-sex-GPRO-float16",  # Larger model
]

# Number of tokens to generate per model before switching
CHUNK_SIZE = 10

# Maximum total tokens for an answer
MAX_TOKENS = 1000

# Store tokenizers and models
MODEL_TOKENIZERS = {}
MODEL_INSTANCES = {}
EOS_TOKEN_IDS = {}

def initialize_tokenizers():
    """Initialize tokenizers and get EOS token IDs for all models"""
    for model_name in MODELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            MODEL_TOKENIZERS[model_name] = tokenizer
            EOS_TOKEN_IDS[model_name] = tokenizer.eos_token_id
            print(f"Model {model_name} EOS token ID: {tokenizer.eos_token_id}")
        except Exception as e:
            print(f"Error loading tokenizer for {model_name}: {e}")

def initialize_models():
    """Initialize models for all specified model names"""
    for model_name in MODELS:
        print(f"Loading model {model_name}...")
        try:
            # Fix - specify CPU explicitly instead of "auto" to avoid SafeTensors error
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,  # Use explicit device instead of "auto"
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            MODEL_INSTANCES[model_name] = model
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

def generate_answer_alternating(question: str, chunk_size: int = CHUNK_SIZE, max_tokens: int = MAX_TOKENS) -> str:
    """
    Generate an answer for `question`, switching between models every `chunk_size` tokens.
    """
    # Make the prompt more explicit to encourage generation
    prompt = question + "\nAnswer:"
    generated = ""
    total_tokens = 0
    current_model_idx = 0
    
    while total_tokens < max_tokens:
        model_name = MODELS[current_model_idx]
        model = MODEL_INSTANCES[model_name]
        tokenizer = MODEL_TOKENIZERS[model_name]
        
        print(f"Using {model_name}...")
        
        try:
            # Tokenize the full prompt + generated text so far
            inputs = tokenizer(prompt + generated, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate next chunk
            outputs = model.generate(
                **inputs,
                max_new_tokens=chunk_size,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Get only the newly generated tokens
            new_tokens = outputs[0, inputs.input_ids.shape[1]:]
            
            # Check for EOS token
            if tokenizer.eos_token_id in new_tokens:
                eos_idx = (new_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                new_tokens = new_tokens[:eos_idx]
                
                new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated += new_text
                print(f"EOS token encountered, stopping generation")
                break
            
            # Decode and add to generated text
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            if not new_text:
                break
                
            generated += new_text
            total_tokens += len(new_tokens)
            
            # Switch to the next model
            current_model_idx = (current_model_idx + 1) % len(MODELS)
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            # Switch to the next model
            current_model_idx = (current_model_idx + 1) % len(MODELS)
    
    if not generated:
        return "Could not generate an answer with any model."
    
    return generated.strip()

def generate_answer_single_model(question: str, model_name: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Generate an answer using a single model.
    """
    # Make the prompt more explicit to encourage generation
    prompt = question + "\nAnswer:"
    
    model = MODEL_INSTANCES[model_name]
    tokenizer = MODEL_TOKENIZERS[model_name]
    
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate answer in one go
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the generated portion (skip the prompt)
        generated = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return generated.strip()
        
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return f"Error generating answer with {model_name}."

def extract_answer(text: str):
    """Extract the numerical answer from text."""
    if not text:
        return None
        
    # First look for "the answer is X" pattern
    answer_match = re.search(r"the answer is \$?(\d+(?:\.\d+)?)\.?", text.lower())
    if answer_match:
        return float(answer_match.group(1))
    
    # Look for the last numerical value in the text
    number_matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    if number_matches and len(number_matches) > 0:
        return float(number_matches[-1])
    
    return None

def evaluate_answer(generated_text: str, ground_truth: str):
    """Compare the generated answer with the ground truth."""
    generated_answer = extract_answer(generated_text)
    correct_answer = extract_answer(ground_truth)
    
    if generated_answer is None or correct_answer is None:
        return False
    
    # Exact match comparison
    return abs(generated_answer - correct_answer) < 1e-6

def evaluate_models(dataset, num_samples, output_dir="results"):
    """Evaluate all models and combinations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Limiting to specified number of samples
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    # Evaluation results for each approach
    results = {
        "model_0": {"name": MODELS[0], "results": {"correct": 0, "total": len(dataset), "details": []}},
        "model_1": {"name": MODELS[1], "results": {"correct": 0, "total": len(dataset), "details": []}},
        "alternating": {"name": "Alternating", "results": {"correct": 0, "total": len(dataset), "details": []}}
    }
    
    # Generate and evaluate for each example
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question = item.get("question", "").strip()
        ground_truth = item.get("answer", "").strip()
        
        if not question:
            continue
            
        print(f"\n[{idx+1}/{len(dataset)}] Question:")
        print(question)
        
        # Generate answers with each approach
        answers = {
            "model_0": generate_answer_single_model(question, MODELS[0]),
            "model_1": generate_answer_single_model(question, MODELS[1]),
            "alternating": generate_answer_alternating(question)
        }
        
        # Evaluate each answer
        for approach, answer in answers.items():
            is_correct = evaluate_answer(answer, ground_truth)
            if is_correct:
                results[approach]["results"]["correct"] += 1
            
            # Store detailed results
            results[approach]["results"]["details"].append({
                "id": idx,
                "question": question,
                "ground_truth": ground_truth,
                "extracted_ground_truth": extract_answer(ground_truth),
                "generated": answer,
                "extracted_generated": extract_answer(answer),
                "correct": is_correct
            })
            
            print(f"{results[approach]['name']} Answer:")
            print(answer)
            print(f"Correct: {is_correct}")
            print("-" * 30)
    
    # Calculate accuracy for each approach
    for approach in results.keys():
        if results[approach]["results"]["total"] > 0:
            results[approach]["results"]["accuracy"] = (
                results[approach]["results"]["correct"] / results[approach]["results"]["total"]
            )
    
    # Save results
    for approach, data in results.items():
        result_file = os.path.join(output_dir, f"{approach}_results.json")
        with open(result_file, "w") as f:
            json.dump(data["results"], f, indent=2)
    
    # Save a summary
    summary = {approach: {
        "name": data["name"],
        "accuracy": data["results"]["accuracy"],
        "correct": data["results"]["correct"],
        "total": data["results"]["total"]
    } for approach, data in results.items()}
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    for approach, data in summary.items():
        print(f"{data['name']}: {data['correct']}/{data['total']} = {data['accuracy']:.2%}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GSM8K math tasks.")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--config", type=str, default="main", help="Dataset configuration")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    print(f"Loaded {len(dataset)} examples.")
    
    # Initialize tokenizers and models
    print("Initializing tokenizers...")
    initialize_tokenizers()
    
    print("Initializing models...")
    initialize_models()
    
    # Run evaluation
    print(f"Running evaluation on {args.samples} examples...")
    results = evaluate_models(dataset, args.samples, args.output)
    
    print(f"Evaluation complete. Results saved to {args.output} directory.")

if __name__ == "__main__":
    main()