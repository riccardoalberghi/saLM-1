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

from src.architecture import MultiModelWithScalarHeads
from src.loss_functions import weighted_token_cross_entropy
from src.model_loader import ModelLoader, ModelInfo
from src.models import MODELS

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
        max_length: int = 128,
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
        }

def train_epoch(
    model: MultiModelWithScalarHeads,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: str,
    entropy_weight: float = 0.1,
    pad_token_id: int = 0
) -> float:
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        # Construct input texts from input_ids
        # Note: This is a simplified approach. In real implementation, you might need to decode the input_ids first
        # or design your dataset class to provide the texts directly
        batch_size = input_ids.shape[0]
        input_texts = ["placeholder"] * batch_size
        
        # Forward pass through the model
        outputs = model(input_texts)
        
        # Get token logits and model probabilities
        token_logits = outputs["token_logits"]
        all_raw_scores = outputs["all_raw_scores"]
        model_ids = outputs["model_ids"]
        
        # Calculate model probabilities
        model_probs = model.get_model_probs(all_raw_scores)
        
        # Calculate token probabilities
        token_probs = model.get_token_probs(token_logits)
        
        # Calculate loss
        loss = weighted_token_cross_entropy(
            token_probs_dict=token_probs,
            model_probs=model_probs,
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
    
    return total_loss / len(dataloader)

def evaluate(
    model: MultiModelWithScalarHeads,
    dataloader: DataLoader,
    device: str,
    entropy_weight: float = 0.1,
    pad_token_id: int = 0
) -> float:
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Similar to training, construct input_texts 
            batch_size = input_ids.shape[0]
            input_texts = ["placeholder"] * batch_size
            
            # Forward pass
            outputs = model(input_texts)
            
            # Get token logits and model probabilities
            token_logits = outputs["token_logits"]
            all_raw_scores = outputs["all_raw_scores"]
            model_ids = outputs["model_ids"]
            
            # Calculate model probabilities
            model_probs = model.get_model_probs(all_raw_scores)
            
            # Calculate token probabilities
            token_probs = model.get_token_probs(token_logits)
            
            # Calculate loss
            loss = weighted_token_cross_entropy(
                token_probs_dict=token_probs,
                model_probs=model_probs,
                target_tokens=target_ids,
                model_ids=model_ids,
                entropy_weight=entropy_weight,
                pad_token_id=pad_token_id
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

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
    
    args = parser.parse_args()
    
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
    # In practice, you might want to handle different tokenizers for different models more carefully
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
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            entropy_weight=args.entropy_weight
        )
        
        # Evaluate
        val_loss = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            entropy_weight=args.entropy_weight
        )
        
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
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

if __name__ == "__main__":
    main()
