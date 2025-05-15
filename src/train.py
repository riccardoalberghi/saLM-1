import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Optional, Tuple
import os
import random
import numpy as np
import wandb

from architecture import MultiModelWithScalarHeads
from model_loader import ModelLoader, ModelInfo
from models import MODELS
from globals import *
from loss_functions import perfect_alignment_loss
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BalancedFinetuningDataset(Dataset):
    """Dataset that combines multiple HuggingFace finetuning datasets and balances them."""

    def __init__(
        self,
        repo_names: List[str],
        tokenizer,
        split: str = "train",
        max_length: int = MAX_LENGTH,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Tuple[str, str]] = []
        random.seed(seed)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        datasets_list = []
        for repo in repo_names:
            try:
                ds = load_dataset(repo, split=split)
                datasets_list.append(ds)
                logger.info(f"Loaded dataset {repo} with {len(ds)} {split} samples")
            except Exception as e:
                logger.warning(f"Failed to load dataset {repo}: {e}")

        if not datasets_list:
            raise ValueError("No datasets could be loaded. Please check dataset names.")

        # Balance by trimming each dataset to the smallest size
        min_size = min(len(ds) for ds in datasets_list)
        logger.info(f"Balancing datasets to smallest size: {min_size} samples each")

        for ds in datasets_list:
            indices = list(range(len(ds)))
            if shuffle:
                random.shuffle(indices)
            for i in indices[:min_size]:
                item = ds[i]
                # Common schema handling
                if "input" in item and "target" in item:
                    self.examples.append((item["input"], item["target"]))
                elif "question" in item and "answer" in item:
                    self.examples.append((item["question"], item["answer"]))
                elif "text" in item and "label" in item:
                    self.examples.append((item["text"], str(item["label"])))
                else:
                    # Skip unsupported schema items
                    continue

        if shuffle:
            random.shuffle(self.examples)
            
        # Print a few examples
        if len(self.examples) > 0:
            logger.info(f"Created {split} dataset with {len(self.examples)} examples")
            for i in range(min(3, len(self.examples))):
                logger.info(f"Example {i+1}: Input: '{self.examples[i][0][:50]}...', Target: '{self.examples[i][1][:50]}...'")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text = self.examples[idx]
        
        # Combine input and target for next token prediction
        combined_text = input_text + target_text
        
        # Just return the raw text - tokenization will happen during training
        return {
            "phrase": combined_text
        }

def train_epoch(
    model: MultiModelWithScalarHeads,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: str,
    entropy_weight: float = 0.1,
    global_step: int = 0,
    log_every_n_steps: int = 10
) -> int:
    """Training loop with perfect alignment between model outputs and labels."""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        phrases = batch["phrase"]
        
        # Forward pass through model
        outputs = model(phrases)
        
        # Get model outputs
        token_logits = outputs["token_logits"]
        all_raw_scores = outputs["all_raw_scores"]
        model_ids = outputs["model_ids"]
        
        # Calculate model probabilities
        model_probs = model.get_model_probs(all_raw_scores)
        
        # Create perfectly aligned labels using SAME tokenizer and parameters as model
        with torch.no_grad():
            # Get tokenizer from model
            tokenizer = model.common_tokenizer
            
            # Tokenize phrases exactly as model does
            encodings = tokenizer(
                phrases, 
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Get input_ids for creating labels
            input_ids = encodings["input_ids"]
            
            # Create next-token prediction labels (shifted by 1)
            labels = torch.full_like(input_ids, fill_value=-100)  # Default to ignore_index
            labels[:, :-1] = input_ids[:, 1:]  # Token at position i predicts token at position i+1
            
            # Mark padding tokens to be ignored in loss calculation
            padding_mask = (input_ids == tokenizer.pad_token_id)
            labels[padding_mask] = -100
        
        # For debugging in first batch
        if batch_idx == 0:
            logger.info(f"Example phrase: '{phrases[0]}'")
            non_pad_input = input_ids[0][input_ids[0] != tokenizer.pad_token_id]
            non_pad_labels = labels[0][labels[0] != -100]
            logger.info(f"Input tokens: {tokenizer.decode(non_pad_input)}")
            logger.info(f"Label tokens: {tokenizer.decode(non_pad_labels)}")
            logger.info(f"Input shape: {input_ids.shape}, Labels shape: {labels.shape}")
            logger.info(f"Model probs shape: {model_probs.shape}")
            
            # Check alignment
            seq_len = min(model_probs.size(1), labels.size(1))
            logger.info(f"Using sequence length: {seq_len} for alignment")
        
        # Ensure all tensors have matching sequence lengths
        seq_len = min(model_probs.size(1), labels.size(1))
        model_probs_aligned = model_probs[:, :seq_len, :]
        labels_aligned = labels[:, :seq_len]
        
        token_logits_aligned = {}
        for model_id in model_ids:
            token_logits_aligned[model_id] = token_logits[model_id][:, :seq_len, :]
        
        # Calculate loss using imported function
        loss = perfect_alignment_loss(
            token_logits=token_logits_aligned,
            model_probs=model_probs_aligned,
            labels=labels_aligned,
            ignore_index=-100,
            entropy_weight=entropy_weight
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Logging
        global_step += 1
        if global_step % log_every_n_steps == 0:
            # Calculate and log perplexity
            perplexity = torch.exp(torch.tensor(loss.item())).item()
            
            # Basic training stats
            log_dict = {
                "train/loss": loss.item(),
                "train/perplexity": perplexity,
                "train/global_step": global_step,
                "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            }
            
            # Create mask for non-ignored positions
            non_ignore_mask = (labels_aligned != -100).float()
            
            # Model usage stats
            _, max_model_indices = model_probs_aligned.max(dim=2)
            total_valid_positions = non_ignore_mask.sum().item()
            
            # Count activations for each model
            activations_count = {}
            for i, model_id in enumerate(model_ids):
                model_activated = (max_model_indices == i).float() * non_ignore_mask
                activations = model_activated.sum().item()
                activation_percentage = activations / (total_valid_positions + 1e-10) * 100
                
                log_dict[f"train/model_activation_count/{model_id}"] = int(activations)
                log_dict[f"train/model_activation_percent/{model_id}"] = activation_percentage
                
                # Mean probability
                model_prob = model_probs_aligned[:, :, i]
                valid_probs = model_prob * non_ignore_mask
                mean_prob = valid_probs.sum().item() / (total_valid_positions + 1e-10)
                log_dict[f"train/model_prob_mean/{model_id}"] = mean_prob
            
            # Log to wandb
            wandb.log(log_dict, step=global_step)
    
    return global_step

def evaluate(
    model: MultiModelWithScalarHeads,
    dataloader: DataLoader,
    device: str,
    entropy_weight: float = 0.1,
) -> float:
    """Evaluation loop with perfect alignment."""
    model.eval()
    total_loss = 0.0
    
    # Tracking metrics
    model_activations = {model_id: 0 for model_id in model.model_ids}
    model_prob_sums = {model_id: 0.0 for model_id in model.model_ids}
    total_valid_positions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            phrases = batch["phrase"]
            
            # Forward pass
            outputs = model(phrases)
            
            # Get model outputs
            token_logits = outputs["token_logits"]
            all_raw_scores = outputs["all_raw_scores"]
            model_ids = outputs["model_ids"]
            
            # Calculate model probabilities
            model_probs = model.get_model_probs(all_raw_scores)
            
            # Create perfectly aligned labels using SAME tokenizer as model
            tokenizer = model.common_tokenizer
            
            # Tokenize phrases
            encodings = tokenizer(
                phrases, 
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Get input_ids
            input_ids = encodings["input_ids"]
            
            # Create next-token prediction labels
            labels = torch.full_like(input_ids, fill_value=-100)
            labels[:, :-1] = input_ids[:, 1:]
            
            # Mask padding
            padding_mask = (input_ids == tokenizer.pad_token_id)
            labels[padding_mask] = -100
            
            # Ensure all tensors have matching sequence lengths
            seq_len = min(model_probs.size(1), labels.size(1))
            model_probs_aligned = model_probs[:, :seq_len, :]
            labels_aligned = labels[:, :seq_len]
            
            token_logits_aligned = {}
            for model_id in model_ids:
                token_logits_aligned[model_id] = token_logits[model_id][:, :seq_len, :]
            
            # Calculate loss using imported function
            loss = perfect_alignment_loss(
                token_logits=token_logits_aligned,
                model_probs=model_probs_aligned,
                labels=labels_aligned,
                ignore_index=-100,
                entropy_weight=entropy_weight
            )
            
            total_loss += loss.item()
            
            # Get mask for non-ignored positions
            non_ignore_mask = (labels_aligned != -100).float()
            batch_valid_positions = non_ignore_mask.sum().item()
            
            # Track model usage
            _, max_model_indices = model_probs_aligned.max(dim=2)
            
            for i, model_id in enumerate(model_ids):
                # Count activations
                model_activated = (max_model_indices == i).float() * non_ignore_mask
                model_activations[model_id] += model_activated.sum().item()
                
                # Track mean probabilities
                model_prob = model_probs_aligned[:, :, i]
                valid_probs = model_prob * non_ignore_mask
                model_prob_sums[model_id] += valid_probs.sum().item()
            
            total_valid_positions += batch_valid_positions
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / max(len(dataloader), 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Log validation metrics
    val_log_dict = {
        "val/loss": avg_loss,
        "val/perplexity": perplexity,
        "val/total_valid_positions": total_valid_positions
    }
    
    # Log model-specific metrics
    for model_id in model.model_ids:
        # Mean probability
        mean_prob = model_prob_sums[model_id] / (total_valid_positions + 1e-10)
        val_log_dict[f"val/model_prob_mean/{model_id}"] = mean_prob
        
        # Activation percentage
        activation_percentage = (model_activations[model_id] / (total_valid_positions + 1e-10)) * 100
        val_log_dict[f"val/model_activation_percent/{model_id}"] = activation_percentage
        val_log_dict[f"val/model_activation_count/{model_id}"] = model_activations[model_id]
    
    # Log to wandb
    wandb.log(val_log_dict)
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train MultiModelWithScalarHeads with Perfect Alignment")
    
    parser.add_argument("--output_dir", type=str, default="./model_outputs", help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps for scheduler")
    parser.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for the entropy term in loss")
    parser.add_argument("--head_hidden_dim", type=int, default=256, help="Hidden dimension for projection heads")
    parser.add_argument("--head_dropout", type=float, default=0.1, help="Dropout rate for projection heads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps). If None, will use best available.")
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="Log metrics every N steps")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        required=True,
        help="List of HuggingFace dataset repository names for finetuning."
    )
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="multimodel-perfect-alignment", config=vars(args))
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize model loader
    model_loader = ModelLoader(device=args.device)
    device = model_loader.device
    logger.info(f"Using device: {device}")
    
    # Initialize MultiModelWithScalarHeads
    model = MultiModelWithScalarHeads(
        base_models=MODELS,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        model_loader=model_loader,
        device=device 
    )
    
    # Standardize special tokens across all tokenizers
    logger.info("Standardizing special tokens across all tokenizers")
    
    # Get available model IDs from the model
    available_model_ids = list(model.tokenizers.keys())
    if not available_model_ids:
        raise ValueError("No models loaded. Check your model configuration.")
    
    # Use the first model as the standard model
    standard_model_id = available_model_ids[0]
    logger.info(f"Using {standard_model_id} as the standard model for tokenization")
    
    standard_tokenizer = model.tokenizers[standard_model_id]
    
    for model_id, tokenizer in model.tokenizers.items():
        if model_id != standard_model_id:
            # Set special tokens to match standard tokenizer
            tokenizer.pad_token = standard_tokenizer.pad_token
            tokenizer.eos_token = standard_tokenizer.eos_token
            tokenizer.bos_token = standard_tokenizer.bos_token if hasattr(standard_tokenizer, 'bos_token') else tokenizer.bos_token
            logger.info(f"Standardized special tokens for {model_id}")
    
    # Use the standard tokenizer for dataset
    tokenizer = model.tokenizers[standard_model_id]
    logger.info(f"Using tokenizer from {standard_model_id}")
    logger.info(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    logger.info(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # Freeze the base models' parameters
    for model_id, base_model in model.base_models.items():
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info(f"Frozen parameters for base model: {model_id}")
    
    # Count trainable parameters
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"Number of trainable parameters: {trainable_params}")
    
    # Create datasets from HuggingFace repositories
    train_dataset = BalancedFinetuningDataset(
        repo_names=args.datasets,
        tokenizer=tokenizer,
        split="train",
        max_length=MAX_LENGTH,
        shuffle=True,
        seed=args.seed
    )
    
    val_dataset = BalancedFinetuningDataset(
        repo_names=args.datasets,
        tokenizer=tokenizer,
        split="validation",
        max_length=MAX_LENGTH,
        shuffle=False,
        seed=args.seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01  # Add weight decay for regularization
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training with perfect alignment")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        global_step = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            entropy_weight=args.entropy_weight,
            global_step=global_step,
            log_every_n_steps=args.log_every_n_steps
        )
        
        # Evaluate
        val_loss = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            entropy_weight=args.entropy_weight
        )
        
        logger.info(f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}")
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "val/loss": val_loss,
            "val/best_loss": best_val_loss if val_loss >= best_val_loss else val_loss
        }, step=global_step)
        
        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save checkpoint for this epoch
        checkpoint = {
            "epoch": epoch + 1,
            "projection_heads_state_dict": {k: v for k, v in model.state_dict().items() if "projection_heads" in k},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": val_loss
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save best projection heads
            best_heads_state = {k: v for k, v in model.state_dict().items() if "projection_heads" in k}
            torch.save(best_heads_state, os.path.join(args.output_dir, "best_model_heads.pt"))
            
            logger.info(f"Saved new best model heads with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
