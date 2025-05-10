import torch
import torch.nn.functional as F
from typing import List, Dict

def weighted_token_cross_entropy(
    token_probs_dict: Dict[str, torch.Tensor],  # Dict of model token probabilities
    model_probs: torch.Tensor,                  # Probability distribution over models
    target_tokens: torch.Tensor,                # Target token indices
    model_ids: List[str],                       # Model IDs in order   
    entropy_weight: float = 0.1,                # Weight for model probabilities entropy                               
    pad_token_id: int = 0                       # ID to ignore (padding)
) -> torch.Tensor:
    """
    Compute cross entropy between weighted token probabilities and targets.
    
    This function:
    1. Computes a weighted average of token probabilities across models
    2. Calculates cross entropy between this weighted distribution and target tokens
    
    Args:
        token_probs_dict: Dictionary mapping model_id to token probabilities
                         Each tensor has shape [batch_size, seq_len, vocab_size]
        model_probs: Probability distribution over models [batch_size, num_models]
        target_tokens: Target token indices [batch_size, seq_len]
        model_ids: List of model IDs corresponding to order in model_probs
        pad_token_id: Token ID to ignore in loss calculation (usually padding)
        
    Returns:
        Cross entropy loss averaged over non-padding positions
    """
    batch_size = model_probs.shape[0]
    
    # Initialize the weighted token probability distribution
    first_model_id = model_ids[0]
    seq_len = token_probs_dict[first_model_id].shape[1]
    vocab_size = token_probs_dict[first_model_id].shape[2]
    device = token_probs_dict[first_model_id].device
    
    # Initialize weighted token probabilities
    weighted_probs = torch.zeros(batch_size, seq_len, vocab_size, device=device)
    
    # Compute weighted average of token probabilities
    for i, model_id in enumerate(model_ids):
        # Get model weight [batch_size, 1, 1]
        model_weight = model_probs[:, i:i+1, None]
        
        # Get token probabilities for this model [batch_size, seq_len, vocab_size]
        token_probs = token_probs_dict[model_id]
        
        # Weight and add to combined probabilities
        weighted_probs += model_weight * token_probs
    
    # Reshape for cross entropy
    flat_probs = weighted_probs.view(-1, vocab_size)  # [batch_size*seq_len, vocab_size]
    flat_targets = target_tokens.view(-1)             # [batch_size*seq_len]
    
    # Create mask for non-padding positions
    mask = (flat_targets != pad_token_id)
    
    per_token_loss = F.cross_entropy(
        flat_probs, 
        flat_targets,
        reduction='none'  
    )
    
    masked_loss = per_token_loss * mask.float()
    num_tokens = mask.sum().float().clamp(min=1.0) 

    model_probs_entropy = -(model_probs * torch.log(model_probs + 1e-10)).sum(dim=-1).mean()
    
    return masked_loss.sum() / num_tokens + entropy_weight * model_probs_entropy

