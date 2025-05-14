import torch
from transformers import AutoTokenizer
from model_loader import ModelInfo
from models import MODELS

def compare_tokenizer_vocabularies(models):
    """
    Load the tokenizers for each model and analyze vocabulary differences.
    """
    print("Loading tokenizers...")
    tokenizers = {}
    
    for model_info in models:
        model_id = model_info.id
        model_path = model_info.path
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizers[model_id] = tokenizer
            print(f"Loaded tokenizer for {model_id} ({model_path}) with vocab size: {len(tokenizer.vocab)}")
        except Exception as e:
            print(f"Failed to load tokenizer for {model_id}: {e}")
    
    # If any tokenizer failed to load, exit
    if len(tokenizers) != len(models):
        print("Not all tokenizers loaded successfully. Exiting.")
        return
    
    # Get all tokenizer vocabularies
    vocabs = {}
    for model_id, tokenizer in tokenizers.items():
        vocab = tokenizer.get_vocab()
        vocabs[model_id] = vocab
        
    # Basic vocabulary statistics
    print("\n===== Basic Vocabulary Statistics =====")
    for model_id, vocab in vocabs.items():
        print(f"Model {model_id}: {len(vocab)} tokens")
    
    # Find common tokens across all vocabularies
    all_tokens = set()
    for vocab in vocabs.values():
        all_tokens.update(vocab.keys())
    
    common_tokens = set.intersection(*[set(vocab.keys()) for vocab in vocabs.values()])
    
    print(f"\nTotal unique tokens across all models: {len(all_tokens)}")
    print(f"Tokens common to all models: {len(common_tokens)} ({len(common_tokens)/len(all_tokens)*100:.2f}%)")
    
    # Analyze pairwise overlap
    print("\n===== Pairwise Vocabulary Overlap =====")
    model_ids = list(vocabs.keys())
    for i in range(len(model_ids)):
        for j in range(i+1, len(model_ids)):
            model_a = model_ids[i]
            model_b = model_ids[j]
            
            tokens_a = set(vocabs[model_a].keys())
            tokens_b = set(vocabs[model_b].keys())
            
            common = tokens_a.intersection(tokens_b)
            only_in_a = tokens_a - tokens_b
            only_in_b = tokens_b - tokens_a
            
            print(f"{model_a} vs {model_b}:")
            print(f"  Common tokens: {len(common)} ({len(common)/len(tokens_a.union(tokens_b))*100:.2f}%)")
            print(f"  Only in {model_a}: {len(only_in_a)} ({len(only_in_a)/len(tokens_a)*100:.2f}%)")
            print(f"  Only in {model_b}: {len(only_in_b)} ({len(only_in_b)/len(tokens_b)*100:.2f}%)")
    
    # Analyze token ID consistency for common tokens
    print("\n===== Token ID Consistency =====")
    id_mismatches = 0
    total_common = len(common_tokens)
    
    for token in common_tokens:
        ids = [vocab[token] for vocab in vocabs.values()]
        if len(set(ids)) > 1:
            id_mismatches += 1
    
    print(f"Common tokens with different IDs: {id_mismatches} out of {total_common} ({id_mismatches/total_common*100:.2f}%)")
    
    # Analyze a few example texts to see tokenization differences
    example_texts = [
        "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
        "The integral of x^2 from 0 to 1 equals 1/3.",
        "Have by lemma auto using rfl"  # Sample Lean tactic text
    ]
    
    print("\n===== Example Tokenization Differences =====")
    for text in example_texts:
        print(f"\nText: {text}")
        for model_id, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            print(f"{model_id}: {tokens} (length: {len(tokens)})")
    
    # Look for special tokens differences
    print("\n===== Special Tokens Comparison =====")
    special_token_attrs = ['unk_token', 'pad_token', 'bos_token', 'eos_token', 'mask_token']
    
    for model_id, tokenizer in tokenizers.items():
        print(f"\nModel {model_id} special tokens:")
        for attr in special_token_attrs:
            if hasattr(tokenizer, attr):
                token = getattr(tokenizer, attr)
                token_str = token if token is not None else "None"
                print(f"  {attr}: {token_str}")
    
    # Save vocabulary mismatches to a file
    print("\n===== Generating Vocabulary Mapping =====")
    # Choose the general model as the reference
    reference_model = f"{MODELS[0].id}"
    reference_vocab = vocabs[reference_model]
    
    # Create mappings between vocabularies
    mappings = {}
    for model_id, vocab in vocabs.items():
        if model_id == reference_model:
            continue
        
        # Map from model's tokens to reference model
        token_mapping = {}
        token_id_mapping = {}
        
        # First, map identical tokens
        for token, idx in vocab.items():
            if token in reference_vocab:
                token_mapping[token] = token
                token_id_mapping[idx] = reference_vocab[token]
        
        mappings[model_id] = {
            "token_mapping": token_mapping,
            "token_id_mapping": token_id_mapping,
            "coverage": len(token_mapping) / len(vocab) * 100
        }
    
    for model_id, mapping in mappings.items():
        print(f"Mapping from {model_id} to {reference_model}:")
        print(f"  Covered tokens: {len(mapping['token_mapping'])} out of {len(vocabs[model_id])} ({mapping['coverage']:.2f}%)")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    compare_tokenizer_vocabularies(MODELS)