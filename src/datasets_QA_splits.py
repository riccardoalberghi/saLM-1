from datasets import load_dataset, DatasetDict, concatenate_datasets
from evaluate import load as load_metric
import gc

# TASKS: mapping name → (HF id, config, metric, trust_remote_code)
TASKS = {
    # General English QA: extractive QA format (context, question, answers)
    "squad_v2": ("squad_v2", None, "f1", True),  # ~100MB

    # Medical QA: USMLE style multiple-choice
    "med_qa": ("bigbio/med_qa", None, "accuracy", True),  # ~200MB

    # Open-domain QA: Natural Questions
    "natural_questions": ("sentence-transformers/natural-questions", None, "f1", True),  # ~1GB

    # Physics/Science QA: ScienceQA
    "science_qa": ("lmms-lab/ScienceQA", "ScienceQA-FULL", "accuracy", True),  # ~100MB

    # Economics QA: FinQA (numerical reasoning)
    "finqa": ("ibm-research/finqa", None, "accuracy", True),  # ~300MB

    # Chemistry QA: ChemProt QA (relation extraction QA)
    "chemprot_qa": ("zapsdcn/chemprot", None, "accuracy", True),  # ~50MB

    # Mathematics QA: MathQA
    "math_qa": ("allenai/math_qa", None, "accuracy", True),  # large-scale math QA (~500MB)

    # Literature QA: TriviaQA
    "trivia_qa": ("sentence-transformers/trivia-qa", None, "f1", True)  # ~100MB
}


loaded = []
failed = []
split_counts = {}

for name, (hf_id, config, metric_name, trust_code) in TASKS.items():
    print(f"\n=== {name} ===")
    try:
        # Load dataset with remote code handling
        load_args = {}
        if config:
            load_args["name"] = config
        ds = load_dataset(hf_id, **load_args, trust_remote_code=trust_code)
        print(f"✔ Initial splits: {list(ds.keys())}")

        # Determine existing splits
        splits = set(ds.keys())

        # If only 'train' exists
        if splits == {"train"}:
            print("- Only 'train' split found. Splitting into 80% train, 10% validation, 10% test.")
            tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
            test_val = tmp["test"].train_test_split(test_size=0.5, seed=42)
            ds = DatasetDict({
                "train": tmp["train"],
                "validation": test_val["train"],
                "test": test_val["test"]
            })
        # If 'train' and 'validation' exist but no 'test'
        elif "train" in splits and "validation" in splits and "test" not in splits:
            print("- 'train' and 'validation' splits found. Creating 'test' split from 'train'.")
            tmp = ds["train"].train_test_split(test_size=0.1111, seed=42)  # Approx 10% of original
            ds = DatasetDict({
                "train": tmp["train"],
                "validation": ds["validation"],
                "test": tmp["test"]
            })
        # If 'train' and 'test' exist but no 'validation'
        elif "train" in splits and "test" in splits and "validation" not in splits:
            print("- 'train' and 'test' splits found. Creating 'validation' split from 'train'.")
            tmp = ds["train"].train_test_split(test_size=0.1111, seed=42)  # Approx 10% of original
            ds = DatasetDict({
                "train": tmp["train"],
                "validation": tmp["test"],
                "test": ds["test"]
            })
        # If 'validation' and 'test' exist but no 'train'
        elif "train" not in splits and "validation" in splits and "test" in splits:
            print("- 'validation' and 'test' splits found. Creating 'train' split from them.")
            combined = concatenate_datasets([ds["validation"], ds["test"]])
            tmp = combined.train_test_split(test_size=0.2, seed=42)
            val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
            ds = DatasetDict({
                "train": tmp["train"],
                "validation": val_test["train"],
                "test": val_test["test"]
            })
        # If all three splits exist
        elif {"train", "validation", "test"}.issubset(splits):
            print("- All required splits are present. No action needed.")
        else:
            print(f"- Unexpected split configuration: {splits}. Attempting to create standard splits.")
            all_splits = []
            for split in ds.values():
                all_splits.append(split)
            combined = all_splits[0]
            for split in all_splits[1:]:
                combined = concatenate_datasets([combined, split])
            tmp = combined.train_test_split(test_size=0.2, seed=42)
            test_val = tmp["test"].train_test_split(test_size=0.5, seed=42)
            ds = DatasetDict({
                "train": tmp["train"],
                "validation": test_val["train"],
                "test": test_val["test"]
            })

        # Print dataset statistics and record split counts
        print(f"✔ Final splits: {list(ds.keys())}")
        counts = {}
        for split in ds.keys():
            count = len(ds[split])
            counts[split] = count
            try:
                size_bytes = ds[split].info.size_in_bytes
                size_mb = size_bytes / (1024 ** 2)
                print(f"   - {split}: {count} examples, {size_mb:.1f} MB")
            except Exception:
                print(f"   - {split}: {count} examples")
        split_counts[name] = counts

        # Load metric
        metric = load_metric(metric_name)
        print(f"✔ Metric '{metric_name}' loaded successfully")

        loaded.append(name)
        # Memory cleanup
        del ds
        gc.collect()

    except Exception as e:
        print(f"✗ Failed to load {name}: {str(e)}")
        failed.append(name)

# Final summary
print("\n=== Final Summary ===")
print(f"Successfully loaded: {loaded}")
print(f"Failed to load: {failed}")
print("\n=== Split Counts ===")
for name, counts in split_counts.items():
    print(f"{name}: train={counts.get('train',0)}, validation={counts.get('validation',0)}, test={counts.get('test',0)}")
