'''
this is models.py with additionally the test to generate answers for each model
'''

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import gc

# Define the set of domain-expert models with parameter count, task, and specialization
MODELS = [
    {
        # General Purpose
        "name": "Qwen2.5-0.5B-Instruct",
        "parameters": 500_000_000,
        "task": "General Purpose (Instruction Following)",
        "path": "Qwen/Qwen2.5-0.5B-Instruct",
        "type": "causal",
        "prompt": "Explain the concept of climate change in simple terms."
    },

    {
        # Mathematics
        "name": "MWP-T5",
        "parameters": 220_000_000,
        "task": "Mathematical Reasoning",
        "path": "mwpt5/t5-mawps-pen",
        "type": "seq2seq",
        "prompt": "Solve this math problem: If x + 3 = 7, what is the value of x?"
    },
    {
        # Biomedical (general purpose but good for specialized domains)
        "name": "SmolLM-360M",
        "parameters": 360_000_000,
        "task": "General Language Tasks (with biomedical capabilities)",
        "path": "HuggingFaceTB/SmolLM-360M",
        "type": "causal",
        "prompt": "Explain the function of antibodies in the immune system."
    },
        # Chemistry/Extraction
    {
        "name": "NuExtract-tiny",
        "parameters": 500_000_000,
        "task": "Chemical Information Extraction",
        "path": "numind/NuExtract-tiny"
        "prompt": "Explain me the difference between a reduction and oxidation reaction"
    }, 

    {
        # Physics
        "name": "smolLM2-360M-physics",
        "parameters": 360_000_000,
        "task": "Physics Problem Solving",
        "path": "akhilfau/fine-tuned-smolLM2-360M-with-on-combined_Instruction_dataset",
        "type": "causal",
        "prompt": "Calculate the force needed to accelerate a 2kg object at 5 m/s²."
    },
    {
        # Computer_Science
        "name": "diff-codegen-350m-v2",
        "parameters": 350_000_000,
        "task": "Code Generation",
        "path": "CarperAI/diff-codegen-350m-v2",
        "type": "causal",
        "prompt": "Write a Python function to check if a number is prime."
    },

        # Biomedical 
    {
        "name": "SmolLM-360M",
        "parameters": 360_000_000,
        "task": "General Language Tasks (with good performance for specialized domains)",
        "path": "HuggingFaceTB/SmolLM-360M"
        "prompt": "What is the difference between RNA and DNA?"
    }, 

    {
        # History
        "name": "TinyLlama-History-Chat",
        "parameters": 200_000_000,
        "task": "History Q&A",
        "path": "ambrosfitz/tinyllama-history-chat_v0.3",
        "type": "causal",
        "prompt": "What were the main causes of World War I?"
    },
    {
        # Philosophy
        "name": "T5-Philosophy-Model",
        "parameters": 60_000_000,
        "task": "Philosophical Text Summarization",
        "path": "tvganesh/philosophy_model",
        "type": "seq2seq",
        "prompt": "Summarize Plato's theory of forms."
    },
    {
        # Literature
        "name": "LED-BookSum",
        "parameters": 500_000_000,
        "task": "Long-form Literature Summarization",
        "path": "pszemraj/led-base-book-summary",
        "type": "seq2seq",
        "prompt": "Summarize the main themes in Shakespeare's Hamlet."
    }
    # economics
    # engineering
]

def test_generation():
    results = []

    # Detect MPS support
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if use_mps else "cpu")
    print(f"\n🖥️ Using device: {device}")

    for m in MODELS:
        print(f"\n--- Testing generation for {m['name']} ({m['task']}, {m['parameters']:,} params) ---")
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(m["path"])
            print(f"Tokenizer loaded for {m['name']}")

            if "prompt" not in m:
                print(f"⚠️ No prompt defined for {m['name']}, skipping...")
                continue

            # Prepare the input
            inputs = tokenizer(m["prompt"], return_tensors="pt").to(device)
            print(f"Input prepared: {m['prompt'][:30]}...")

            # Load and run model
            if m.get("type") == "causal":
                model = AutoModelForCausalLM.from_pretrained(m["path"]).to(device)

                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif m.get("type") == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(m["path"]).to(device)

                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif m.get("type") == "encoder":
                model = AutoModel.from_pretrained(m["path"]).to(device)

                outputs = model(**inputs)
                generated_text = f"[Encoder model processed input - shape: {outputs.last_hidden_state.shape}]"

            else:
                raise ValueError(f"Unknown model type for {m['name']}")

            print(f"✅ Generation successful for {m['name']}")
            print(f"Generated output (truncated): {generated_text[:100]}...")

            results.append({
                "model": m["name"],
                "success": True,
                "output": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
            })

        except Exception as e:
            print(f"❌ Failed to generate with {m['name']}: {e}")
            results.append({
                "model": m["name"],
                "success": False,
                "error": str(e)
            })

        # Cleanup
        del tokenizer, model, inputs
        if "outputs" in locals():
            del outputs
        gc.collect()

    print("\n\n=== GENERATION TEST SUMMARY ===")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{status} {r['model']}: {'Generation successful' if r['success'] else 'Failed - ' + r['error']}")

    return results


if __name__ == "__main__":
    test_generation()