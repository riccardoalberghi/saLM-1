from datasets import load_dataset, DatasetDict, concatenate_datasets
from evaluate import load as load_metric
import gc

# TASKS: mapping name → (HF id, config, metric, trust_remote_code)

TASKS = {
    # General English QA: extractive QA format (context, question, answers)
    "squad_v2": ("squad_v2", None, "f1", True),  # ~100MB

    # Medical QA: MedMCQA (~183K training examples)
    "med_qa": ("openlifescienceai/medmcqa", None, "accuracy", True),  # ~200MB

    # Open-domain QA: Natural Questions
    "natural_questions": ("sentence-transformers/natural-questions", None, "f1", True),  # ~1GB

    # Science QA: SciQAG (~188K QA pairs)
    "science_qa": ("MasterAI-EAM/SciQAG", None, "accuracy", True),  # ~200MB

    # Economics QA: Sujet-Finance-QA-Vision-100k (~100K QA pairs)
    #"finqa": ("sujet-ai/Sujet-Finance-QA-Vision-100k", None, "accuracy", True),  # ~300MB

    # Chemistry QA: ChemRxivQuest (~100K QA pairs)
    #"chemprot_qa": ("chemrxiv_quest", None, "accuracy", True),  # ~150MB

    # Mathematics QA: AQuA-RAT (~100K algebraic word problems)
    "math_qa": ("deepmind/aqua_rat", None, "accuracy", True),  # ~500MB

    # Literature QA: TriviaQA
    "trivia_qa": ("sentence-transformers/trivia-qa", None, "f1", True)  # ~100MB
}


loaded = []
failed = []
split_counts = {}

for name, (hf_id, config, metric_name, trust_code) in TASKS.items():
    print(f"\n=== {name} ===")
    try:
        load_args = {}
        if config:
            load_args["name"] = config
        ds = load_dataset(hf_id, **load_args, trust_remote_code=trust_code)
        print(f"✔ Initial splits: {list(ds.keys())}")

        splits = set(ds.keys())

        # Split normalization logic
        if splits == {"train"}:
            print("- Only 'train' split found. Splitting into 80% train, 10% validation, 10% test.")
            tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
            test_val = tmp["test"].train_test_split(test_size=0.5, seed=42)
            ds = DatasetDict({
                "train": tmp["train"],
                "validation": test_val["train"],
                "test": test_val["test"]
            })
        elif "train" in splits and "validation" in splits and "test" not in splits:
            val_size = len(ds["validation"])
            train_size = len(ds["train"])
            desired_val_size = int(0.1 * train_size)
            if val_size < desired_val_size:
                print(f"- 'validation' split too small ({val_size} < {desired_val_size}). Resplitting 'train'.")
                tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
                test_val = tmp["test"].train_test_split(test_size=0.5, seed=42)
                ds = DatasetDict({
                    "train": tmp["train"],
                    "validation": test_val["train"],
                    "test": test_val["test"]
                })
            else:
                print("- 'train' and 'validation' splits found. Creating 'test' split from 'train'.")
                tmp = ds["train"].train_test_split(test_size=0.1111, seed=42)
                ds = DatasetDict({
                    "train": tmp["train"],
                    "validation": ds["validation"],
                    "test": tmp["test"]
                })
        elif "train" in splits and "test" in splits and "validation" not in splits:
            test_size = len(ds["test"])
            train_size = len(ds["train"])
            desired_test_size = int(0.1 * train_size)
            if test_size < desired_test_size:
                print(f"- 'test' split too small ({test_size} < {desired_test_size}). Resplitting 'train'.")
                tmp = ds["train"].train_test_split(test_size=0.2, seed=42)
                test_val = tmp["test"].train_test_split(test_size=0.5, seed=42)
                ds = DatasetDict({
                    "train": tmp["train"],
                    "validation": test_val["train"],
                    "test": test_val["test"]
                })
            else:
                print("- 'train' and 'test' splits found. Creating 'validation' split from 'train'.")
                tmp = ds["train"].train_test_split(test_size=0.1111, seed=42)
                ds = DatasetDict({
                    "train": tmp["train"],
                    "validation": tmp["test"],
                    "test": ds["test"]
                })
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
        elif {"train", "validation", "test"}.issubset(splits):
            print("- All required splits are present. No action needed.")
        else:
            print(f"- Unexpected split configuration: {splits}. Attempting to create standard splits.")
            all_splits = [split for split in ds.values()]
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

        # Print dataset statistics
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

        # Print dataset schema and head of each split
        for split in ds.keys():
            print(f"\n📊 Schema of '{split}' split:")
            try:
                print(ds[split].features)
            except Exception as e:
                print(f"   - Failed to print schema for '{split}': {e}")

            print(f"\n🔍 Preview of '{split}' split:")
            try:
                print(ds[split].select(range(min(3, len(ds[split])))))
            except Exception as e:
                print(f"   - Failed to preview split '{split}': {e}")

        split_counts[name] = counts

        # Load metric
        metric = load_metric(metric_name)
        print(f"✔ Metric '{metric_name}' loaded successfully")

        loaded.append(name)
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
    print(f"{name}: train={counts.get('train', 0)}, validation={counts.get('validation', 0)}, test={counts.get('test', 0)}")
