import sys
from itertools import cycle

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Two Hugging Face model identifiers
MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "likewendy/Qwen2.5-3B-sex-GPRO-float16",
]

# Number of tokens to generate per model before switching
CHUNK_SIZE = 10

# Maximum total tokens for an answer
MAX_TOKENS = 1000

# Store tokenizers and EOS token IDs
MODEL_TOKENIZERS = {}
EOS_TOKEN_IDS = {}

def initialize_tokenizers():
    """Initialize tokenizers and get EOS token IDs for all models"""
    for model_name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        MODEL_TOKENIZERS[model_name] = tokenizer
        EOS_TOKEN_IDS[model_name] = tokenizer.eos_token_id
        print(f"Model {model_name} EOS token ID: {tokenizer.eos_token_id}")

def generate_answer(question: str, engines, chunk_size: int = CHUNK_SIZE, max_tokens: int = MAX_TOKENS) -> str:
    """
    Generate an answer for `question`, switching between engines every `chunk_size` tokens.
    Stops generation when EOS token is encountered.
    """
    # Make the prompt more explicit to encourage generation
    prompt = question + "\nAnswer:"
    generated = ""
    total_tokens = 0
    current_model_idx = 0
    
    while total_tokens < max_tokens:
        current_engine = engines[current_model_idx]
        model_name = MODELS[current_model_idx]
        
        print(f"Using {model_name}...")
        
        # Set parameters for this chunk with EOS token as a stop condition
        params = SamplingParams(
            temperature=0.6,
            max_tokens=chunk_size,
            stop_token_ids=[EOS_TOKEN_IDS[model_name]]
            skip_special_tokens=False,
        )
        
        # Generate with current model
        try:
            result = current_engine.generate(
                prompt + generated,
                sampling_params=params,
            )
            
            # Extract the generated text
            output = result[0].outputs[0]
            text = output.text
            
            # Check if the model produced an EOS token
            if len(output.token_ids) > 0 and output.token_ids[-1] == EOS_TOKEN_IDS[model_name]:
                print(f"EOS token encountered, stopping generation")
                generated += text
                break
            
            generated += text
            total_tokens += len(output.token_ids)

            print(generated)
            
            # If no tokens were generated, it likely means we've reached the end
            if len(output.token_ids) == 0:
                break
            
            # Switch to the next model for next chunk
            current_model_idx = (current_model_idx + 1) % len(engines)
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            # Switch to the next model
            current_model_idx = (current_model_idx + 1) % len(engines)
    
    if not generated:
        return "Could not generate an answer with any model."
    
    return generated.strip()

def main():
    # Load a small slice of GSM8K for testing; remove slicing for full dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train[:100]")

    print("Initializing tokenizers...")
    initialize_tokenizers()

    print("Initializing vLLM engines...")
    engines = []
    for model_name in MODELS:
        engine = LLM(model=model_name, tensor_parallel_size=1)
        engines.append(engine)

    # Iterate and generate
    for idx, item in enumerate(dataset):
        question = item.get("question", "").strip()
        if not question:
            continue
        print(f"\n[{idx+1}/{len(dataset)}] Question:")
        print(question)
        answer = generate_answer(question, engines)
        print("Answer:")
        print(answer)
        print("-" * 60)

if __name__ == "__main__":
    main()
