import torch
import torch.nn.functional as F
from typing import List, Dict

def weighted_token_cross_entropy(
    token_probs_dict: Dict[str, torch.Tensor],
    model_probs: torch.Tensor,  # [batch_size, seq_len, num_models]
    target_tokens: torch.Tensor,
    pad_token_id: int,
    model_ids: List[str],
    entropy_weight: float = 0.1
) -> torch.Tensor:
    """
    Calculate weighted token cross-entropy loss with token-level routing.
    
    Args:
        token_probs_dict: Dictionary of token probabilities for each model
                         [batch_size, seq_len, vocab_size]
        model_probs: Probabilities for each model at each token position
                    [batch_size, seq_len, num_models]
        target_tokens: Target token IDs [batch_size, seq_len]
        model_ids: List of model IDs
        entropy_weight: Weight for entropy regularization
        pad_token_id: Token ID for padding
        
    Returns:
        Weighted cross-entropy loss
    """
    # Verify dimensions
    assert model_probs.dim() == 3, "model_probs must be 3D: [batch_size, seq_len, num_models]"
    
    batch_size, seq_len, num_models = model_probs.shape
    vocab_size = next(iter(token_probs_dict.values())).size(-1)
    
    # ------------------------------------------------------------------
    # Align sequence lengths across all inputs by using the minimum length
    # available. This avoids assertion errors when some tensors are shorter
    # than others (e.g., target sequence shorter than model_probs length).
    # ------------------------------------------------------------------
    # effective_seq_len = min(
    #     seq_len,
    #     target_tokens.size(1),
    #     *[token_probs_dict[model_id].size(1) for model_id in model_ids]
    # )
    
    # Trim tensors to the effective sequence length
    # model_probs = model_probs[:, :effective_seq_len]
    # target_tokens = target_tokens[:, :effective_seq_len]
    # for model_id in model_ids:
    #     token_probs_dict[model_id] = token_probs_dict[model_id][:, :effective_seq_len]
    
    # Update seq_len to reflect the aligned length
    # seq_len = effective_seq_len
    
    # Initialize tensor for combined probabilities
    combined_probs = torch.zeros(batch_size, seq_len, vocab_size, device=target_tokens.device, requires_grad=True)
    
    # Combine token probabilities with position-specific model weights
    for i, model_id in enumerate(model_ids):
        # Get weights for this model at each position [batch_size, seq_len, 1]
        model_weight = model_probs[:, :, i].unsqueeze(-1)
        
        # Only use required sequence length
        token_probs = token_probs_dict[model_id][:, :seq_len]
        
        # Weight each model's token probabilities by its position-specific weight
        combined_probs = combined_probs + model_weight * token_probs
    
    # Reshape for cross entropy calculation
    # From [batch_size, seq_len, vocab_size] to [batch_size * seq_len, vocab_size]
    combined_probs_flat = combined_probs.reshape(-1, vocab_size)
    
    # Reshape target tokens from [batch_size, seq_len] to [batch_size * seq_len]
    target_flat = target_tokens.reshape(-1)
    
    # Create mask for non-padding tokens
    non_pad_mask = (target_flat != pad_token_id)
    
    # Only calculate loss on non-padding tokens
    if non_pad_mask.sum() > 0:
        # Filter out padding tokens
        # Use indexing that preserves gradients
        valid_indices = torch.nonzero(non_pad_mask).squeeze()
        combined_probs_non_pad = combined_probs_flat.index_select(0, valid_indices)
        target_non_pad = target_flat.index_select(0, valid_indices)
        
        # Ensure target indices are valid
        max_index = vocab_size - 1
        target_non_pad = torch.clamp(target_non_pad, 0, max_index)
        
        # Calculate negative log-likelihood loss directly from predicted probabilities
        # Gather the probability assigned to the correct target token at each position
        selected_probs = combined_probs_non_pad.gather(1, target_non_pad.unsqueeze(1)).squeeze(1)
        
        # Use ‑log(p) as the loss (add small epsilon for numerical stability)
        token_loss = -torch.log(selected_probs + 1e-10).mean()
    else:
        # If all tokens are padding, return zero loss
        # Use a tensor connected to the graph
        token_loss = torch.sum(combined_probs) * 0.0
    
    # Add entropy regularization - encourage diversity in model selection
    # Calculate entropy at each token position and average
    entropy = -torch.sum(model_probs * torch.log(model_probs + 1e-10), dim=2).mean()
    
    # Subtract entropy (with weight) from loss to encourage diverse model usage
    total_loss = token_loss - entropy_weight * entropy
    
    return total_loss
