import torch
import os
import argparse
import logging
from typing import Dict, Optional
from src.architecture import MultiModelWithScalarHeads
from src.model_loader import ModelLoader
from src.models import MODELS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_finetuned_model(
    checkpoint_path: str,
    head_hidden_dim: int = 256,
    head_dropout: float = 0.1,
    device: Optional[str] = None
) -> MultiModelWithScalarHeads:
    """
    Load a MultiModelWithScalarHeads model with finetuned projection heads
    
    Args:
        checkpoint_path: Path to the saved projection heads checkpoint
        head_hidden_dim: Hidden dimension for projection heads
        head_dropout: Dropout rate for projection heads
        device: Device to load the model on (cuda, cpu, mps)
        
    Returns:
        MultiModelWithScalarHeads model with loaded projection heads
    """
    # Initialize model loader and determine device
    model_loader = ModelLoader(device=device)
    device = model_loader.device
    logger.info(f"Using device: {device}")
    
    # Initialize fresh model
    model = MultiModelWithScalarHeads(
        base_models=MODELS,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        model_loader=model_loader,
        device=device
    )
    
    # Load saved projection heads state dict
    logger.info(f"Loading finetuned projection heads from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "projection_heads_state_dict" in checkpoint:
        # If the checkpoint contains a projection_heads_state_dict key
        projection_heads_state = checkpoint["projection_heads_state_dict"]
    else:
        # If the checkpoint is just the projection heads state dict itself
        projection_heads_state = checkpoint
    
    # Create a new state dict for the full model
    model_state_dict = model.state_dict()
    
    # Update only the projection heads part of the model
    for key in projection_heads_state:
        if key in model_state_dict:
            model_state_dict[key] = projection_heads_state[key]
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict)
    logger.info("Successfully loaded finetuned projection heads")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Load a finetuned MultiModelWithScalarHeads model")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--head_hidden_dim", type=int, default=256, help="Hidden dimension for projection heads")
    parser.add_argument("--head_dropout", type=float, default=0.1, help="Dropout rate for projection heads")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    # Load the model
    model = load_finetuned_model(
        checkpoint_path=args.checkpoint_path,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        device=args.device
    )
    
    logger.info("Model loaded successfully!")
    logger.info(f"Model has {len(model.model_ids)} base models with the following IDs:")
    for model_id in model.model_ids:
        logger.info(f"  - {model_id}")

if __name__ == "__main__":
    main()