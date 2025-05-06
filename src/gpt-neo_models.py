import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import sys

MODELS = [
    # === GPT‑Neo‑125M Domain Variants ===
    {
        "name": "GPT‑Neo‑125M (General)",
        "parameters": 125_000_000,
        "task": "General‑purpose text generation",
        "path": "EleutherAI/gpt-neo-125m",
        "type": "causal",
        "prompt": "Once upon a time in a distant galaxy,"
    },
    {
        "name": "GPT‑Neo‑125M‑Code‑Clippy",
        "parameters": 125_000_000,
        "task": "Code completion",
        "path": "flax-community/gpt-neo-125M-code-clippy",
        "type": "causal",
        "prompt": "```python\ndef quicksort(arr):"
    },
    {
        "name": "GPT‑Neo‑125M‑APPS",
        "parameters": 125_000_000,
        "task": "Programming problem solving",
        "path": "flax-community/gpt-neo-125M-apps",
        "type": "causal",
        "prompt": "Write a function to solve the Tower of Hanoi."
    },
    {
        "name": "GPT‑Neo‑Math‑Small",
        "parameters": 125_000_000,
        "task": "Proof‑step generation (Lean tactics)",
        "path": "Saisam/gpt-neo-math-small",
        "type": "causal",
        "prompt": "<GOAL> Prove that the sum of two even numbers is even. <PROOFSTEP>"
    },
    
    {
        "name": "GPT‑Neo‑CS‑Finetuned",
        "parameters": 125_000_000,
        "task": "Computer science (code & tutorials)",
        "path": "KimByeongSu/gpt-neo-125m-cs-finetuning-20000",
        "type": "causal",
        "prompt": "Write a Python function to check if a number is prime."
    },
    {
        "name": "GPT‑Neo‑Literature",
        "parameters": 125_000_000,
        "task": "Fictional storytelling",
        "path": "hakurei/lit-125M",
        "type": "causal",
        "prompt": "[ Title: The Dunwich Horror; Author: H. P. Lovecraft; Genre: Horror ] *** When a traveler"
    },
    {
        "name": "GPT‑Neo‑History",
        "parameters": 125_000_000,
        "task": "Historical analysis",
        "path": "shubhamgantayat/EleutherAI-gpt-neo-125m-brief-history-of-time-model",
        "type": "causal",
        "prompt": "Analyze the causes and consequences of the Industrial Revolution."
    },
]


def test_generation():
    results = []
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if use_mps else "cpu")
    print(f"\n🖥️ Using device: {device}")

    # List of models requiring explicit GPTNeoForCausalLM
    gpt_neo_model_paths = [
        "jtatman/gpt-neo-125m-chem-physics",
        "bhaskartripathi/GPT_Neo_Market_Analysis",
        "faclorxin152/gpt-neo-125m-medical-qa"
    ]

    for m in MODELS:
        print(f"\n--- Testing generation for {m['name']} ({m['task']}) ---")
        tokenizer = model = inputs = outputs = None
        try:
            # Handle tokenizer for Medical QA model
            if m["path"] == "faclorxin152/gpt-neo-125m-medical-qa":
                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
            else:
                tokenizer = AutoTokenizer.from_pretrained(m["path"])

            # Handle model loading for specific paths
            if m["path"] in gpt_neo_model_paths:
                from transformers import GPTNeoForCausalLM
                model = GPTNeoForCausalLM.from_pretrained(m["path"]).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(m["path"]).to(device)

            inputs = tokenizer(m["prompt"], return_tensors="pt").to(device)
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"model": m["name"], "success": True, "output": text})
            print(f"✅ {m['name']} → {text[:80]}…")

        except Exception as e:
            print(f"❌ {m['name']}: {e}")
            results.append({"model": m["name"], "success": False, "error": str(e)})

        finally:
            # safe cleanup only if variables were set
            for var in (tokenizer, model, inputs, outputs):
                try:
                    del var
                except NameError:
                    pass
            gc.collect()
            if use_mps:
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

    # summary
    print("\n=== GENERATION TEST SUMMARY ===")
    failures = [r for r in results if not r["success"]]
    for r in results:
        status = "✅" if r["success"] else "❌"
        note   = r.get("error", "Success")
        print(f"{status} {r['model']}: {note}")

    # final check: abort if any model failed
    if failures:
        failed_models = [f["model"] for f in failures]
        raise RuntimeError(f"Generation failed for: {failed_models}")
    else:
        print("🎉 All models generated successfully!")

    return results

if __name__ == "__main__":
    try:
        results = test_generation()
        
        # New: Print successful models list
        successes = [r['model'] for r in results if r['success']]
        print("\n🧠 Successfully loaded models:")
        for model in successes:
            print(f"  - {model}")
            
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)