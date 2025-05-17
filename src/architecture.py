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

    def train(self, mode:bool = True):
        super().train(mode)

        self.non_linear_projection.train(mode)
        return self

    def eval(self, mode:bool = True):
        return self.train(not mode)


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
        
    def train(self, mode:bool = True):
        super().train(mode)

        for model in self.base_models.values():
            model.train(mode)
        for head in self.projection_heads.values():
            head.train(mode)

        return self

    def eval(self, mode:bool = True):
        return self.train(not mode)

    def generate(
        self,
        *args,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        return_decisions: bool = False,
        return_token_ids: bool = True,  # New parameter to control output format
        **kwargs  # Accept all standard generation parameters
    ):
        """
        Generate text using model-wise confidence to select next token.
        Supports both string prompts and tokenized inputs.
        Optionally returns model decisions at each step and/or raw token IDs.

        Args:
            max_new_tokens (int): Max number of tokens to generate.
            temperature (float): Softmax temperature (1.0 = no scaling).
            return_decisions (bool): Whether to return a log of token + model at each step.
            return_token_ids (bool): Whether to return token IDs instead of decoded text.
            **kwargs: Additional generation parameters including:
                - input_ids: Pre-tokenized input IDs
                - attention_mask: Attention mask for the input
                - prompt: Text prompt as an alternative to input_ids

        Returns:
            Depending on parameters:
            - If return_token_ids=True, return_decisions=False: Returns the tensor of token IDs
            - If return_token_ids=True, return_decisions=True: Returns dict with token IDs and generation steps
            - If return_token_ids=False, return_decisions=False: Returns decoded string
            - If return_token_ids=False, return_decisions=True: Returns dict with decoded text and generation steps
        """

        self.eval()
        tokenizer = self.common_tokenizer
        
        # Extract inputs from kwargs
        input_ids = kwargs.get('input_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        prompt = kwargs.get('prompt', None)
        
        # Handle already tokenized inputs
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        # If no input_ids provided, we need a prompt string
        elif prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            raise ValueError("Either 'input_ids' or 'prompt' keyword argument must be provided")
        
        stop_token = tokenizer.eos_token
        generation_trace = []
        batch_size = input_ids.shape[0]
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Decode current sequence to text
            input_texts = [tokenizer.decode(input_ids[i], skip_special_tokens=False) for i in range(batch_size)]
            
            # Process through the model
            outputs = self(input_texts)
            token_logits = outputs["token_logits"]
            all_raw_scores = outputs["all_raw_scores"]
            
            # Get model confidence at last token
            model_confidence = self.get_model_probs(all_raw_scores)
            last_pos = model_confidence.shape[1] - 1
            
            # Store next token for each item in batch
            next_token_batch = []
            decoded_tokens = []
            best_model_ids = []
            
            for b in range(batch_size):
                # Find best model for this batch item
                best_model_idx = torch.argmax(model_confidence[b, last_pos]).item()
                best_model_id = self.model_ids[best_model_idx]
                best_model_ids.append(best_model_id)
                
                # Get logits from the chosen model at last position
                logits = token_logits[best_model_id][b, -1, :]
                
                # Temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                    
                # Select next token
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.argmax(probs).unsqueeze(0)
                next_token_batch.append(next_token_id)
                
                # Decode token if needed for stop condition checking
                decoded_token = tokenizer.decode(next_token_id)
                decoded_tokens.append(decoded_token)
            
            # Stack tokens for the batch and append to input_ids
            next_tokens = torch.stack(next_token_batch).to(self.device)
            input_ids = torch.cat([input_ids, next_tokens.view(batch_size, 1)], dim=1)
            
            # Track generation details
            for b in range(batch_size):
                if len(generation_trace) <= b:
                    generation_trace.append([])
                    
                generation_trace[b].append({
                    "token": decoded_tokens[b],
                    "token_id": next_token_batch[b].item(),
                    "model_id": best_model_ids[b]
                })
                
                # Check for stop conditions per batch item
                if stop_token and stop_token in decoded_tokens[b]:
                    # This would be more complex for true batch generation
                    # Here we simplify by stopping all generation when batch[0] is done
                    if b == 0:
                        break
                if next_token_batch[b].item() == tokenizer.eos_token_id:
                    if b == 0:
                        break
        
        # Prepare outputs based on return preferences
        if return_token_ids:
            # Return token IDs without decoding
            if batch_size == 1:
                if return_decisions:
                    return {
                        "token_ids": input_ids[0],  # Return as tensor
                        "steps": generation_trace[0]
                    }
                else:
                    return input_ids[0]  # Return just the tensor
            else:
                if return_decisions:
                    return [{
                        "token_ids": input_ids[i],
                        "steps": generation_trace[i]
                    } for i in range(batch_size)]
                else:
                    return input_ids  # Return all batch token IDs
        else:
            # Decode full outputs as strings (original behavior)
            outputs = [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(batch_size)]
            
            if batch_size == 1:
                if return_decisions:
                    return {
                        "generated_text": outputs[0],
                        "steps": generation_trace[0]
                    }
                else:
                    return outputs[0]
            else:
                if return_decisions:
                    return [{
                        "generated_text": outputs[i],
                        "steps": generation_trace[i]
                    } for i in range(batch_size)]
                else:
                    return outputs
        
    @classmethod
    def from_pretrained(
            cls,
            base_models_info: List[ModelInfo],
            weights_path: str,
            head_hidden_dim: int = 256,
            head_dropout: float = 0.1,
            model_loader: Optional[ModelLoader] = None,
            device: str = None,
            dtype: torch.dtype = torch.float32
        ):
        """
        Load a MultiModelWithScalarHeads model with pretrained projection head weights.

        Args:
            base_models_info: List of ModelInfo objects for the base models
            weights_path: Path to the .pt file containing saved projection head weights
            head_hidden_dim: Hidden dimension size for projection heads
            head_dropout: Dropout rate for projection heads
            model_loader: Optional custom ModelLoader instance
            device: Device to load the model onto
            dtype: Data type for model parameters
            
        Returns:
            MultiModelWithScalarHeads: Loaded model with pretrained projection heads
        """
        import os
        import torch

        # Determine device if not provided
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        print(f"Loading ensemble model on {device}...")

        # Initialize the model with default (untrained) projection heads
        model = cls(
            base_models=base_models_info,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            model_loader=model_loader,
            device=device,
            dtype=dtype
        )

        print("Model initialized. Loading projection head weights...")

        # Load the saved projection head weights
        if os.path.exists(weights_path):
            print(f"Loading projection head weights from {weights_path}")
            
            # Load the state dict
            state_dict = torch.load(weights_path, map_location=device)
            
            # Check if state_dict is a dictionary (it should be)
            if not isinstance(state_dict, dict):
                print(f"Warning: Expected dictionary from {weights_path}, got {type(state_dict)}. Skipping weight loading.")
                return model
                
            # Analyze structure of the state dict to determine how to load it
            # Check if it's a nested dictionary where keys are model_ids
            if all(k in model.model_ids for k in state_dict.keys()) and all(isinstance(v, dict) for v in state_dict.values()):
                # Nested structure: {model_id: head_state_dict}
                print("Loading weights with structure: {model_id: head_state_dict}")
                for model_id, head_state in state_dict.items():
                    model.projection_heads[model_id].load_state_dict(head_state)
                    
            # Check if it's a flat dictionary with projection_heads prefix
            elif any(k.startswith('projection_heads.') for k in state_dict.keys()):
                # Flat structure: projection_heads module state_dict
                print("Loading weights with structure: projection_heads module state_dict")
                model.load_state_dict(state_dict, strict=False)
                
            # Check if it's a flat dictionary for just one projection head
            elif any('weight' in k or 'bias' in k for k in state_dict.keys()):
                # Try to determine if this belongs to a specific head
                # For simplicity, we'll try to load into all heads
                print("Loading weights as single head state_dict (trying for all heads)")
                for model_id in model.model_ids:
                    try:
                        model.projection_heads[model_id].load_state_dict(state_dict, strict=False)
                        print(f"  - Successfully loaded into head for {model_id}")
                    except Exception as e:
                        print(f"  - Failed to load into head for {model_id}: {e}")
            
            # If structure is different, try a flexible approach
            else:
                print("Attempting flexible weight loading")
                # First, try loading directly with strict=False
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded weights with flexible mapping")
                except Exception as e:
                    print(f"Error in flexible loading: {e}")
                    
                    # Second attempt: Extract any keys that might match
                    matching_keys = {}
                    for k, v in state_dict.items():
                        for model_id in model.model_ids:
                            if model_id in k:
                                new_key = f'projection_heads.{model_id}' + k[k.find(model_id) + len(model_id):]
                                matching_keys[new_key] = v
                    
                    if matching_keys:
                        try:
                            model.load_state_dict(matching_keys, strict=False)
                            print(f"Loaded {len(matching_keys)} matching weights with remapping")
                        except Exception as e:
                            print(f"Error in remapped loading: {e}")
                    else:
                        print("Warning: Could not match weights to model structure")
                
            print("Projection head weights loading attempts completed")
        else:
            print(f"Warning: Weights file {weights_path} does not exist. Using randomly initialized projection heads.")

        # Set the model to evaluation mode
        model.eval()

        return model
    
    def save_projection_heads(self, save_path: str):
        """
        Save the projection heads to a specified path.

        Args:
            save_path (str): Path to save the projection heads.
        """

        import os
        import torch
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Extract just the projection heads state dict
        projection_heads_dict = {}
        for name, param in self.state_dict().items():
            if name.startswith('projection_heads.'):
                projection_heads_dict[name] = param
        
        # Save to file
        torch.save(projection_heads_dict, save_path)
        print(f"Projection heads saved to {save_path}")
