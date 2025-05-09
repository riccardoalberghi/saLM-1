from datasets import load_dataset
from evaluate import load as load_metric

TASKS = {
    "math_gsm8k":        ("openai/gsm8k",                     "main",           "accuracy"),
    "code_to_text":      ("google/code_x_glue_ct_code_to_text","python",          "bleu"),
    "protein_stability": ("SaProtHub/Dataset-Stability-TAPE",  None,              "spearmanr"),
    "protein_fluorescence": ("SaProtHub/Dataset-Fluorescence-TAPE", None,          "spearmanr"),
    "science_qa":        ("lmms-lab/ScienceQA",               "ScienceQA-FULL",  "accuracy"),
    "med_qa":            ("bigbio/med_qa",                    None,              "accuracy"),
    "econ_logic_qa":     ("yinzhu-quan/econ_logic_qa",        None,              "accuracy"),
    "qm9":               ("yairschiff/qm9",                   None,              "mse"),
    # Additional math dataset: ASDiv
    "math_asdiv":        ("EleutherAI/asdiv",               None,              "accuracy"),
}

loaded = []
failed = []

for name, (hf_id, config, metric_name) in TASKS.items():
    print(f"\n=== {name} ===")
    # Load dataset
    try:
        ds = load_dataset(hf_id) if config is None else load_dataset(hf_id, config)
        print(f"✔ Loaded splits: {list(ds.keys())}")
        loaded.append(name)
    except Exception as e:
        print(f"✗ Failed to load {hf_id!r}: {e}")
        failed.append(name)
        continue

    # Load metric
    try:
        metric = load_metric(metric_name)
        print(f"✔ Metric '{metric_name}' loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load metric '{metric_name}': {e}")

# Final summary
print("\n=== Summary of dataset loading ===")
print(f"Successfully loaded datasets: {loaded}")
print(f"Failed to load datasets: {failed}")
