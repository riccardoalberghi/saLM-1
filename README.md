# saLM-1: Self-Aware Language Model Ensemble

## Overview

The architecture uses scalar projection heads on top of pre-trained language models to determine which model should handle each token in a sequence. The current implementation includes:

- General-purpose text generation (Qwen 2.5 1.5B)
- Medical question answering (Qwen 2.5 1.5B Medical Finetuned)
- Mathematical reasoning (Qwen 2.5 1.5B Math)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nlp
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Training

The training process fine-tunes the projection heads while keeping the base models frozen. This approach allows the system to learn when to use each specialized model without modifying the underlying language models.

### Training Command

```bash
python src/train.py \
  --datasets cola13/gsm8k_formatted cola13/medqa_formatted cola13/boolq_formatted \
  --output_dir ./model_outputs \
  --num_epochs 5 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --warmup_steps 50 \
  --entropy_weight 0.1 \
  --head_hidden_dim 256 \
  --head_dropout 0.1 \
  --device cuda
```

### Training Parameters

- `--datasets`: List of HuggingFace dataset repositories for finetuning. The default datasets are:
  - `cola13/gsm8k_formatted` (Mathematical reasoning)
  - `cola13/medqa_formatted` (Medical knowledge)
  - `cola13/boolq_formatted` (General knowledge)
- `--output_dir`: Directory to save model checkpoints (default: `./model_outputs`)
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_steps`: Number of warmup steps for scheduler (default: 50)
- `--entropy_weight`: Weight for the entropy term in loss (default: 0.1)
- `--head_hidden_dim`: Hidden dimension for projection heads (default: 256)
- `--head_dropout`: Dropout rate for projection heads (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (cuda, cpu, mps). If not specified, will use best available.
- `--log_every_n_steps`: Log metrics every N steps (default: 5)

During training, the script:
1. Loads the specified datasets and balances them to have equal representation
2. Initializes the base models and projection heads
3. Freezes the base models' parameters, only training the projection heads
4. Uses the perfect alignment loss function to train the system
5. Evaluates on validation data and saves the best model

The training progress and metrics are logged to Weights & Biases.

## Evaluation

The evaluation script assesses performance on mathematical reasoning (GSM8K) and medical question answering (MedQA) tasks.

### Evaluation Command

```bash
python evaluate.py \
  --datasets all \
  --sample_size 50 \
  --model_weights ./model_outputs/best_model_heads.pt \
  --output_dir ./evaluation_results \
  --save_examples \
  --max_new_tokens 512 \
  --temperature 0.3
```

### Evaluation Parameters

- `--datasets`: Datasets to evaluate on. Options: "gsm8k", "medqa", "all" (default: "all")
- `--sample_size`: Number of examples to evaluate (default: 50)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--model_weights`: Path to MultiModelWithScalarHeads weights file
- `--only_ensemble`: Only evaluate the ensemble model, not individual models
- `--max_new_tokens`: Maximum number of tokens to generate (default: 512)
- `--temperature`: Temperature for text generation (default: 0.3)
- `--output_dir`: Directory to save results (default: "evaluation_results")
- `--save_examples`: Save detailed examples for analysis
- `--num_examples`: Number of examples to save when --save_examples is used (default: 5)
- `--no_plots`: Disable generation of result plots
- `--device`: Device to run on (default: auto-detect)
- `--debug`: Enable debug mode with minimal samples and verbose output

The evaluation provides:
1. Accuracy metrics for both ensemble and individual models
2. Detailed analysis of model decision patterns
3. Visualizations comparing model performance
4. Example outputs for qualitative analysis

## Using the Trained Model

To generate answers using a trained model:

```python
from src.architecture import MultiModelWithScalarHeads
from src.models import MODELS

# Load the model with trained weights
model = MultiModelWithScalarHeads.from_pretrained(
    base_models=MODELS,
    weights_path="./model_outputs/best_model_heads.pt"
)

# Generate text
prompt = "Solve the following problem: If I have 5 apples and give 2 to my friend, how many do I have left?"
response = model.generate(
    prompt=prompt,
    max_new_tokens=200,
    temperature=0.3
)

print(response)
```

For more advanced use, the `answer_generator.py` script can be used to process questions from different domains.

## Repository Structure

- `src/`: Source code directory
  - `architecture.py`: Implementation of the multi-model architecture
  - `models.py`: Configuration for base models
  - `train.py`: Training script
  - `model_loader.py`: Utility for loading models
  - `loss_functions.py`: Implementation of loss functions
  - `test_models.py`: Testing script for models
  - `globals.py`: Global constants
- `evaluate.py`: Script for evaluating model performance
- `answer_generator.py`: Script for generating answers to questions
- `requirements.txt`: Required dependencies
