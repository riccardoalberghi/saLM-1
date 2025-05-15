import torch
import torch.nn as nn
from typing import List, Dict, Optional
from model_loader import ModelLoader, ModelInfo
from models import MODELS

class ScalarProjectionHead(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32  # Added dtype parameter

    ):
        super().__init__()
        
        intermediate_dim = hidden_dim // 2
        
        self.non_linear_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(intermediate_dim, 1)  
        )
        self.to(dtype)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Convert the input tensor to the same dtype as the projection head
        if hidden_state.dtype != next(self.parameters()).dtype:
            hidden_state = hidden_state.to(next(self.parameters()).dtype)
        return self.non_linear_projection(hidden_state)


class MultiModelWithScalarHeads(nn.Module):
    
    def __init__(
        self, 
        base_models: List[ModelInfo],
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        model_loader: Optional[ModelLoader] = None,
        device: str = None,
        dtype: torch.dtype = torch.float32  # Added dtype parameter

    ):
        super().__init__()
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.model_loader = model_loader if model_loader else ModelLoader(device)
        
        self.base_models = {}
        self.tokenizers = {}
        self.model_ids = []

            # Store the device
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_loader = model_loader if model_loader else ModelLoader(self.device)

        
        # Load all models 
        for model_info in base_models:
            model, tokenizer = self.model_loader.get_model(model_info)
            model_id = model_info.id
            self.base_models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            self.model_ids.append(model_id)

            #MATI FA GLI SGAMBETTI AI CIECHI

            # Set padding token for each tokenizer if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Check that all models have the same hidden size
        hidden_sizes = [model.config.hidden_size for model in self.base_models.values()]
        assert all(size == hidden_sizes[0] for size in hidden_sizes), \
            f"All models must have the same hidden size. Found sizes: {hidden_sizes}"
        
        self.common_tokenizer = self.tokenizers[self.model_ids[0]]

        self.hidden_size = hidden_sizes[0]
        
        # Create a projection head for each model
        self.projection_heads = nn.ModuleDict()
        
        for model_id in self.model_ids:
            self.projection_heads[model_id] = ScalarProjectionHead(
                input_dim=self.hidden_size,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout
            ).to(self.device)

        # Move the entire model to the device
        self.to(self.device)    
    
    def forward(self, input_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process input texts through all models and their projection heads.

        Args:
            input_texts: Input texts to process as a batch

        Returns:
            Dictionary containing hidden states, raw scores, and model IDs
        """
        if not isinstance(input_texts, list):
            input_texts = [input_texts]

        hidden_states      = {}
        raw_model_scores   = {}
        token_logits       = {}
        seq_lens_per_model = {}

        #tokenize with the common tokenizer
        inputs = self.common_tokenizer(input_texts,
                    padding=True,
                    return_tensors="pt").to(self.device)
        # -------------------------------------------------
        # 1. Run every base model and collect its scores
        # -------------------------------------------------
        for model_id, model in self.base_models.items():

            # Freeze base model parameters
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Save logits (detached)
            token_logits[model_id] = outputs.logits.detach()

            # Last‑layer hidden states: [B, L, H]
            last_hidden_state = outputs.hidden_states[-1].to(self.device).detach()

            batch_size, seq_len, _ = last_hidden_state.shape
            seq_lens_per_model[model_id] = seq_len            # keep track

            # Compute projection‑head scores
            position_scores = torch.zeros(batch_size, seq_len, device=self.device)

            for pos in range(seq_len):
                token_representation = last_hidden_state[:, pos, :]
                position_scores[:, pos] = (
                    self.projection_heads[model_id](token_representation)
                    .squeeze(-1)
                )

            raw_model_scores[model_id] = position_scores     # [B, L]

        # -------------------------------------------------
        # 2. Right‑pad each tensor with zeros to the max length
        # -------------------------------------------------
        max_seq_len = max(seq_lens_per_model.values())

        for model_id, scores in raw_model_scores.items():
            pad_len = max_seq_len - scores.size(1)
            if pad_len > 0:
                # pad on the right (dim=1) with zeros
                raw_model_scores[model_id] = torch.nn.functional.pad(
                    scores, (0, pad_len), value=0.0
                )

        # -------------------------------------------------
        # 3. Stack into shape [B, max_seq_len, num_models]
        # -------------------------------------------------
        all_raw_model_scores = torch.stack(
            [raw_model_scores[model_id] for model_id in self.model_ids],
            dim=-1
        )
        
        # Return results
        return {
            "hidden_states": hidden_states,
            "raw_scores": raw_model_scores,
            "all_raw_scores": all_raw_model_scores,
            "token_logits": token_logits,
            "model_ids": self.model_ids
        }
    
    def get_model_probs(self, all_raw_scores: torch.Tensor) -> torch.Tensor:
        """Convert raw scores to model probabilities using softmax."""
        return torch.softmax(all_raw_scores, dim=-1)
  
    def get_token_probs(self, token_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert token logits to token probabilities using softmax, ensuring unified vocab size."""
        # First, find the maximum vocabulary size across all models
        max_vocab_size = max(logits.size(-1) for logits in token_logits.values())
        
        # Convert logits to probabilities and pad if necessary
        token_probs = {}
        for model_id, logits in token_logits.items():
            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Check if we need to pad
            current_vocab_size = probs.size(-1)
            if current_vocab_size < max_vocab_size:
                # Create zero padding tensor
                padding = torch.zeros(
                    probs.size(0),  # batch size
                    probs.size(1),  # sequence length
                    max_vocab_size - current_vocab_size,  # padding width
                    device=probs.device
                )
                
                # Concatenate the padding with the probabilities
                probs = torch.cat([probs, padding], dim=-1)
                
            token_probs[model_id] = probs
        
        return token_probs 
    
    def get_combined_token_probs(self, token_logits: Dict[str, torch.Tensor], model_probs: torch.Tensor) -> torch.Tensor:
        """Get token probabilities weighted by model probabilities"""
        # Convert logits to probabilities
        token_probs = self.get_token_probs(token_logits)
        
        # Get dimensions from the first model's token probabilities
        first_model_id = self.model_ids[0]
        batch_size, seq_len, vocab_size = token_probs[first_model_id].shape
        
        # Initialize combined probabilities
        combined_probs = torch.zeros(batch_size, seq_len, vocab_size, device=self.device)
        
        # Check if we have token-level routing
        if model_probs.dim() == 3:  # [batch_size, seq_len, num_models]
            # Token-level routing
            for i, model_id in enumerate(self.model_ids):
                # Get weight for this model at each position [batch_size, seq_len, 1]
                weight = model_probs[:, :, i].unsqueeze(-1)
                
                # Weight the token probabilities
                weighted_probs = weight * token_probs[model_id]
                
                # Add to combined probabilities
                combined_probs += weighted_probs
        else:  # [batch_size, num_models]
            # Original sequence-level routing behavior
            for i, model_id in enumerate(self.model_ids):
                # Get weight for this model [batch_size, 1, 1]
                weight = model_probs[:, i].unsqueeze(-1).unsqueeze(-1)
                
                # Weight the token probabilities
                weighted_probs = weight * token_probs[model_id]
                
                # Add to combined probabilities
                combined_probs += weighted_probs
        
        return combined_probs
        
    def __del__(self):
        """Clean up resources when the model is deleted."""
        try:
            self.model_loader.unload_all_models()
        except:
            pass
