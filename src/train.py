import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import argparse
from typing import List, Dict, Optional, Tuple
import os
import random
import numpy as np
from datasets import load_dataset
import wandb

from architecture import MultiModelWithScalarHeads
from loss_functions import weighted_token_cross_entropy
from model_loader import ModelLoader, ModelInfo
from models import MODELS
from globals import *

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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text = self.examples[idx]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "target_ids": target_encoding["input_ids"].squeeze(),
            "input_text": input_text,
            "target_text": target_text,
        }

def train_epoch(
    model: MultiModelWithScalarHeads,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: str,
    entropy_weight: float = 0.1,
    pad_token_id: int = 0,
    global_step: int = 0,
    log_every_n_steps: int = 10
) -> int:
    model.train()
    total_loss = 0.0
    truncation_count = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        target_ids = batch["target_ids"].to(device)

        # Retrieve input and target texts directly from the batch
        input_texts = batch["input_text"]
        target_texts = batch["target_text"]

        # Get the tokenizer
        tokenizer = model.tokenizers[model.model_ids[0]]
        
        # Build teacher-forcing inputs by concatenating input and target texts
        # Handle max sequence length gracefully
        combined_texts = []
        for inp, tgt in zip(input_texts, target_texts):
            # Combine input and target with EOS token
            combined = inp + tokenizer.eos_token + tgt
            
            # Tokenize to check length
            tokens = tokenizer.encode(combined, add_special_tokens=False)
            
            # If combined text is too long, truncate from the beginning
            # This keeps the most recent part of the input and all of the target
            if len(tokens) > MAX_LENGTH:
                truncation_count += 1
                # Take the last MAX_LENGTH tokens
                tokens = tokens[-MAX_LENGTH:]
                # Convert back to text
                combined = tokenizer.decode(tokens)
            
            combined_texts.append(combined)

        # Forward pass through the model with teacher forcing
        outputs = model(combined_texts)

        # Get token logits and model probabilities
        token_logits = outputs["token_logits"]
        all_raw_scores = outputs["all_raw_scores"]
        model_ids = outputs["model_ids"]

        # Calculate model probabilities
        model_probs = model.get_model_probs(all_raw_scores)

        # Calculate token probabilities
        token_probs_full = model.get_token_probs(token_logits)

        # ---------------------------
        # Align predictions with targets (teacher forcing)
        # ---------------------------
        seq_len_target = target_ids.size(1)

        # Select predictions that correspond to each target token.
        # For an autoregressive LM, the logits at position *i* predict token *i+1*.
        # Because we appended the full target sequence to the input, the first
        # prediction for the target sequence is located at the EOS position that
        # precedes the first target token and the final logits position predicts
        # the token after the last target token (which we discard).
        model_probs_aligned = model_probs[:, -(seq_len_target + 1):-1, :]
        token_probs = {
            mid: probs[:, -(seq_len_target + 1):-1, :]
            for mid, probs in token_probs_full.items()
        }

        # Calculate loss
        loss = weighted_token_cross_entropy(
            token_probs_dict=token_probs,
            model_probs=model_probs_aligned,
            target_tokens=target_ids,
            model_ids=model_ids,
            entropy_weight=entropy_weight,
            pad_token_id=pad_token_id
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        # wandb logging every n steps
        global_step += 1
        if global_step % log_every_n_steps == 0:
            # Prepare logging dictionary
            log_dict = {
                "train/loss": loss.item(),
                "train/global_step": global_step,
                "train/truncated_sequences": truncation_count
            }
            
            # Get the mask for non-padding tokens (to ignore padding in activation counts)
            non_padding_mask = (target_ids != pad_token_id).float()
            
            # Track model activations (which model has highest probability at each position)
            # Get model with highest probability at each position [batch_size, seq_len]
            _, max_model_indices = model_probs_aligned.max(dim=2)
            
            # Initialize counter for model activations
            activations_count = {model_id: 0 for model_id in model_ids}
            total_valid_positions = non_padding_mask.sum().item()
            
            # Count activations across the batch for each model
            for i, model_id in enumerate(model_ids):
                # Count positions where this model has highest probability (excluding padding)
                model_activated = (max_model_indices == i).float() * non_padding_mask
                activations = model_activated.sum().item()
                activations_count[model_id] = int(activations)
            
            # Log mean model probabilities and activation metrics for each model
            for i, model_id in enumerate(model_ids):
                # Extract the probability for this model at each position [batch_size, seq_len]
                model_prob = model_probs_aligned[:, :, i]
                
                # Calculate mean across batch and sequence dimensions (for non-padding tokens)
                valid_probs = model_prob * non_padding_mask
                sum_probs = valid_probs.sum().item()
                mean_prob = sum_probs / (total_valid_positions + 1e-10)  # Avoid division by zero
                
                # Log the mean probability
                log_dict[f"train/model_prob_mean/{model_id}"] = mean_prob
                
                # Log the activation percentage (how often this model is chosen)
                activation_percentage = activations_count[model_id] / (total_valid_positions + 1e-10) * 100
                log_dict[f"train/model_activation_percent/{model_id}"] = activation_percentage
                
                # Log the raw activation count
                log_dict[f"train/model_activation_count/{model_id}"] = activations_count[model_id]
            
            # Log total number of valid (non-padding) positions
            log_dict["train/total_valid_positions"] = total_valid_positions
            
            wandb.log(log_dict, step=global_step)
    
    # Log final truncation count
    if truncation_count > 0:
        logger.info(f"Truncated {truncation_count} sequences during training due to exceeding max length of {MAX_LENGTH}")

    return global_step

def evaluate(
    model: MultiModelWithScalarHeads,
    dataloader: DataLoader,
    device: str,
    entropy_weight: float = 0.1,
    pad_token_id: int = 0
) -> float:
    model.eval()
    total_loss = 0.0
    truncation_count = 0
    
    # Counters for validation statistics
    model_prob_sums = {model_id: 0.0 for model_id in model.model_ids}
    model_activations = {model_id: 0 for model_id in model.model_ids}
    total_valid_positions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            target_ids = batch["target_ids"].to(device)
            
            # Retrieve input and target texts directly from the batch
            input_texts = batch["input_text"]
            target_texts = batch["target_text"]
            
            # Get the tokenizer
            tokenizer = model.tokenizers[model.model_ids[0]]
            
            # Build teacher-forcing inputs with length handling
            combined_texts = []
            for inp, tgt in zip(input_texts, target_texts):
                # Combine input and target with EOS token
                combined = inp + tokenizer.eos_token + tgt
                
                # Tokenize to check length
                tokens = tokenizer.encode(combined, add_special_tokens=False)
                
                # If combined text is too long, truncate from the beginning
                if len(tokens) > MAX_LENGTH:
                    truncation_count += 1
                    # Take the last MAX_LENGTH tokens
                    tokens = tokens[-MAX_LENGTH:]
                    # Convert back to text
                    combined = tokenizer.decode(tokens)
                
                combined_texts.append(combined)
            
            # Forward pass
            outputs = model(combined_texts)
            
            # Get token logits and model probabilities
            token_logits = outputs["token_logits"]
            all_raw_scores = outputs["all_raw_scores"]
            model_ids = outputs["model_ids"]
            
            # Calculate model probabilities
            model_probs = model.get_model_probs(all_raw_scores)
            
            # Calculate token probabilities
            token_probs_full = model.get_token_probs(token_logits)
            
            # Align predictions with targets (teacher forcing)
            seq_len_target = target_ids.size(1)
            # Align predictions with targets as in training
            model_probs_aligned = model_probs[:, -(seq_len_target + 1):-1, :]
            token_probs = {
                mid: probs[:, -(seq_len_target + 1):-1, :]
                for mid, probs in token_probs_full.items()
            }
            
            # Calculate loss
            loss = weighted_token_cross_entropy(
                token_probs_dict=token_probs,
                model_probs=model_probs_aligned,
                target_tokens=target_ids,
                model_ids=model_ids,
                entropy_weight=entropy_weight,
                pad_token_id=pad_token_id
            )
            
            total_loss += loss.item()
            
            # Get the mask for non-padding tokens
            non_padding_mask = (target_ids != pad_token_id).float()
            
            # Get model with highest probability at each position
            _, max_model_indices = model_probs_aligned.max(dim=2)
            
            # Calculate batch statistics
            batch_valid_positions = non_padding_mask.sum().item()
            
            # Gather statistics
            for i, model_id in enumerate(model_ids):
                # Aggregate probabilities for this model (excluding padding)
                model_prob = model_probs_aligned[:, :, i]
                valid_probs = model_prob * non_padding_mask
                model_prob_sums[model_id] += valid_probs.sum().item()
                
                # Count positions where this model has highest probability
                model_activated = (max_model_indices == i).float() * non_padding_mask
                model_activations[model_id] += model_activated.sum().item()
            
            # Accumulate total non-padding positions
            total_valid_positions += batch_valid_positions
    
    # Log validation statistics to wandb
    val_log_dict = {
        "val/loss": total_loss / len(dataloader),
        "val/truncated_sequences": truncation_count
    }
    
    # Log mean probabilities and activation percentages
    for model_id in model.model_ids:
        # Calculate mean probability
        mean_prob = model_prob_sums[model_id] / (total_valid_positions + 1e-10)
        val_log_dict[f"val/model_prob_mean/{model_id}"] = mean_prob
        
        # Calculate activation percentage
        activation_percentage = (model_activations[model_id] / (total_valid_positions + 1e-10)) * 100
        val_log_dict[f"val/model_activation_percent/{model_id}"] = activation_percentage
        val_log_dict[f"val/model_activation_count/{model_id}"] = model_activations[model_id]
    
    # Log total valid positions
    val_log_dict["val/total_valid_positions"] = total_valid_positions
    
    # Log to wandb
    wandb.log(val_log_dict)
    
    # Log truncation count
    if truncation_count > 0:
        logger.info(f"Truncated {truncation_count} sequences during evaluation due to exceeding max length of {MAX_LENGTH}")
    
    return total_loss / len(dataloader)

# Save the model checkpoint
def save_checkpoint(
    model: MultiModelWithScalarHeads,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    epoch: int,
    loss: float,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only the projection heads' state dictionary
    projection_heads_state = {}
    model_state_dict = model.state_dict()
    
    for key in model_state_dict:
        # Only save parameters for the projection heads
        if 'projection_heads' in key:
            projection_heads_state[key] = model_state_dict[key]
    
    checkpoint = {
        "epoch": epoch,
        "projection_heads_state_dict": projection_heads_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))
    logger.info(f"Saved checkpoint for epoch {epoch}")

def main():
    parser = argparse.ArgumentParser(description="Train MultiModelWithScalarHeads")
    
    parser.add_argument("--output_dir", type=str, default="./model_outputs", help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for scheduler")
    parser.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for the entropy term in loss")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--head_hidden_dim", type=int, default=256, help="Hidden dimension for projection heads")
    parser.add_argument("--head_dropout", type=float, default=0.1, help="Dropout rate for projection heads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps). If None, will use best available.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        required=True,
        help="List of HuggingFace dataset repository names for finetuning."
    )
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log metrics every N steps")
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="multimodel-finetuning", config=vars(args))
    
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
    
    # Freeze the base models' parameters
    for model_id, base_model in model.base_models.items():
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info(f"Frozen parameters for base model: {model_id}")
    
    # Only train the projection heads
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"Number of trainable parameters: {trainable_params}")
    
    # Get a tokenizer from the first model to use for the dataset
    first_model_id = MODELS[0].id
    tokenizer = model.tokenizers[first_model_id]
    
    # Create dataset and dataloaders
    # Build balanced finetuning dataset
    train_dataset = BalancedFinetuningDataset(
        args.datasets,
        tokenizer,
        max_length=args.max_length,
        split="train",
    )
    val_dataset = BalancedFinetuningDataset(
        args.datasets,
        tokenizer,
        max_length=args.max_length,
        split="validation",
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
    
    # Initialize optimizer - only for the projection heads parameters
    optimizer = optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    global_step = 0

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Train for one epoch, logging every few steps
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

        # Log validation loss and best validation loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "val/loss": val_loss,
            "val/best_loss": best_val_loss if val_loss >= best_val_loss else val_loss
        }, step=global_step)

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            loss=val_loss,
            output_dir=args.output_dir
        )

        # Save best model (only projection heads)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Extract projection heads state dict
            projection_heads_state = {}
            model_state_dict = model.state_dict()

            for key in model_state_dict:
                # Only save parameters for the projection heads
                if 'projection_heads' in key:
                    projection_heads_state[key] = model_state_dict[key]

            torch.save(projection_heads_state, os.path.join(args.output_dir, "best_model_heads.pt"))
            logger.info(f"Saved new best model heads with validation loss: {val_loss:.4f}")

    logger.info("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()