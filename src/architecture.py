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
        dropout: float = 0.1
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
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.non_linear_projection(hidden_state)


class MultiModelWithScalarHeads(nn.Module):
    
    def __init__(
        self, 
        base_models: List[ModelInfo],
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        model_loader: Optional[ModelLoader] = None,
        device: str = None
    ):
        super().__init__()
        
        self.model_loader = model_loader if model_loader else ModelLoader(device)
        
        self.base_models = {}
        self.tokenizers = {}
        self.model_ids = []
        
        # Load all models 
        for model_info in base_models:
            model, tokenizer = self.model_loader.get_model(model_info)
            model_id = model_info.id
            self.base_models[model_id] = model
            self.tokenizers[model_id] = tokenizer
            self.model_ids.append(model_id)
        
        # Check that all models have the same hidden size
        hidden_sizes = [model.config.hidden_size for model in self.base_models.values()]
        assert all(size == hidden_sizes[0] for size in hidden_sizes), \
            f"All models must have the same hidden size. Found sizes: {hidden_sizes}"
        
        self.hidden_size = hidden_sizes[0]
        
        # Create a projection head for each model
        self.projection_heads = nn.ModuleDict()
        
        for model_id in self.model_ids:
            self.projection_heads[model_id] = ScalarProjectionHead(
                input_dim=self.hidden_size,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout
            )
    
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
        
        hidden_states = {}
        raw_model_scores = {}
        token_logits = {}
        
        for model_id, model in self.base_models.items():
            tokenizer = self.tokenizers[model_id]
            
            # Tokenize input
            inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(model.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get token logits for future use in training
                logits = outputs.logits
                token_logits[model_id] = logits
                
                # Get the last token's representation from the last layer
                last_hidden_state = outputs.hidden_states[-1]
                last_token_representation = last_hidden_state[:, -1, :]
                
                # Store hidden state
                hidden_states[model_id] = last_token_representation
                
                # Apply the model-specific projection head
                raw_model_scores[model_id] = self.projection_heads[model_id](last_token_representation)
        
        # Stack all raw scores into a single tensor
        all_raw_model_scores = torch.cat([raw_model_scores[model_id] for model_id in self.model_ids], dim=1)
        
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
        return torch.softmax(all_raw_scores, dim=1)
    
    def get_token_probs(self, token_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert token logits to token probabilities using softmax."""
        return {model_id: torch.softmax(logits, dim=-1) 
                for model_id, logits in token_logits.items()}
    
    def get_combined_token_probs(self, token_logits: Dict[str, torch.Tensor], model_probs: torch.Tensor) -> torch.Tensor:
        """Get token probabilities weighted by model probabilities"""
        # ASSUMING SAME VOCAB FOR ALL MODELS, THIS MIGHT NOT BE TRUE

        # Convert logits to probabilities
        token_probs = self.get_token_probs(token_logits)
        
        # Initialize combined probabilities
        combined_probs = None
        
        # Weight and combine token probabilities
        for i, model_id in enumerate(self.model_ids):
            # Get weight for this model [batch_size, 1, 1]
            weight = model_probs[:, i:i+1, None]
            
            # Weight the token probabilities
            weighted_probs = weight * token_probs[model_id]
            
            # Add to combined probabilities
            if combined_probs is None:
                combined_probs = weighted_probs
            else:
                combined_probs += weighted_probs
                
        return combined_probs
    
    def __del__(self):
        """Clean up resources when the model is deleted."""
        try:
            self.model_loader.unload_all_models()
        except:
            pass


