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
    
    def forward(self, input_texts: List[str], past_key_values: Dict[str, List] = None, use_cache: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process input texts through all models and their projection heads.

        Args:
            input_texts: Input texts to process as a batch
            past_key_values: Optional cached key-value pairs from previous forward passes
            use_cache: Whether to use and return cached key-value pairs

        Returns:
            Dictionary containing hidden states, raw scores, model IDs, and optionally cached key-value pairs
        """
        if not isinstance(input_texts, list):
            input_texts = [input_texts]

        hidden_states      = {}
        raw_model_scores   = {}
        token_logits       = {}
        seq_lens_per_model = {}
        new_past_key_values = {} if use_cache else None

        #tokenize with the common tokenizer
        inputs = self.common_tokenizer(input_texts,
                    padding=True,
                    return_tensors="pt").to(self.device)
        # -------------------------------------------------
        # 1. Run every base model and collect its scores
        # -------------------------------------------------
        for model_id, model in self.base_models.items():
            # Extract past key-values for this model if available
            past_kv_for_model = None
            if past_key_values is not None and model_id in past_key_values:
                past_kv_for_model = past_key_values[model_id]

            # Freeze base model parameters
            with torch.no_grad():
                # Pass past_key_values to model if available
                model_inputs = {**inputs}
                if past_kv_for_model is not None:
                    model_inputs['past_key_values'] = past_kv_for_model
                    
                # Always request use_cache when available for efficiency
                model_inputs['use_cache'] = use_cache
                model_inputs['output_hidden_states'] = True
                
                outputs = model(**model_inputs)

            # Save logits (detached)
            token_logits[model_id] = outputs.logits.detach()

            # Store key-value pairs if requested
            if use_cache:
                new_past_key_values[model_id] = outputs.past_key_values

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
        result = {
            "hidden_states": hidden_states,
            "raw_scores": raw_model_scores,
            "all_raw_scores": all_raw_model_scores,
            "token_logits": token_logits,
            "model_ids": self.model_ids
        }
        
        # Add past_key_values to the result if caching is used
        if use_cache:
            result["past_key_values"] = new_past_key_values
            
        return result
    
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
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        return_decisions: bool = False,
        return_token_ids: bool = True,
        **kwargs  # Accept all standard generation parameters
    ):
        """
        Generate text using model-wise confidence to select the next token.
        Optimized for speed with cached processing and efficient token handling.

        Args:
            max_new_tokens (int): Max number of tokens to generate.
            temperature (float): Controls the randomness of predictions.
                - If temperature=0: Greedy decoding (always selects most likely token)
                - If temperature>0: Uses sampling with temperature scaling (higher = more random)
            return_decisions (bool): Whether to return a log of token + model at each step.
            return_token_ids (bool): Whether to return token IDs instead of decoded text.
            **kwargs: Additional generation parameters including:
                - input_ids: Pre-tokenized input IDs
                - attention_mask: Attention mask for the input
                - prompt: Text prompt as an alternative to input_ids

        Returns:
            Depending on parameters:
            - If return_token_ids=True, return_decisions=False: Returns the tensor of token IDs (only generated tokens)
            - If return_token_ids=True, return_decisions=True: Returns dict with token IDs and generation steps
            - If return_token_ids=False, return_decisions=False: Returns decoded string (only generated text)
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
            original_input_length = input_ids.shape[1]
        # If no input_ids provided, we need a prompt string
        elif prompt is not None:
            tokenized_input = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized_input.input_ids.to(self.device)
            attention_mask = tokenized_input.attention_mask.to(self.device) if attention_mask is None else attention_mask
            original_input_length = input_ids.shape[1]
        else:
            raise ValueError("Either 'input_ids' or 'prompt' keyword argument must be provided")
        
        # Initialize proper attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        
        # Setup for generation
        eos_token_id = tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        generation_trace = [[] for _ in range(batch_size)]
        
        # Initial input processing
        with torch.no_grad():
            # Initialize KV caching
            past_key_values = None
            
            # Process the initial input to get the first set of token logits and scores
            input_texts = [tokenizer.decode(input_ids[i], skip_special_tokens=False) for i in range(batch_size)]
            outputs = self(input_texts, use_cache=True)
            token_logits = outputs["token_logits"]
            all_raw_scores = outputs["all_raw_scores"]
            past_key_values = outputs["past_key_values"]  # Cache KV pairs for next forward pass
            
            # Get model confidence
            model_confidence = self.get_model_probs(all_raw_scores)
            
            # Keep track of the best model IDs for generation trace
            last_best_model_ids = [None] * batch_size
            
            # Initialize EOS flags for each batch item
            eos_generated = [False] * batch_size
            
            # Generate tokens efficiently
            for _ in range(max_new_tokens):
                # Get positions in the sequence for next token prediction
                last_pos = input_ids.shape[1] - 1  # Same for all batch items
                
                # Store token selections for this step
                next_token_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
                best_model_ids = []
                
                # Process each item in the batch
                for b in range(batch_size):
                    # Skip if we've already generated EOS for this batch item
                    if eos_generated[b]:
                        best_model_ids.append(last_best_model_ids[b] or self.model_ids[0])
                        continue
                    
                    # Find best model based on confidence
                    best_model_idx = torch.argmax(model_confidence[b, min(last_pos, model_confidence.shape[1]-1)]).item()
                    best_model_id = self.model_ids[best_model_idx]
                    last_best_model_ids[b] = best_model_id
                    
                    best_model_ids.append(best_model_id)
                    
                    # Get logits from chosen model
                    logits = token_logits[best_model_id][b, -1, :].clone()
                    
                    # Apply temperature scaling and select next token
                    if temperature > 0.0:
                        # Temperature sampling
                        scaled_logits = logits / temperature
                        probs = torch.softmax(scaled_logits, dim=-1)
                        # Sample from the distribution
                        next_token_id = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy decoding (argmax)
                        next_token_id = torch.argmax(logits).view(1)
                        
                    next_token_ids[b, 0] = next_token_id
                    
                    # Check for EOS
                    if next_token_id.item() == eos_token_id:
                        eos_generated[b] = True
                    
                    # Add to generation trace
                    decoded_token = tokenizer.decode(next_token_id)
                    generation_trace[b].append({
                        "token": decoded_token,
                        "token_id": next_token_id.item(),
                        "model_id": best_model_id
                    })
                
                # Append new tokens to input_ids
                input_ids = torch.cat([input_ids, next_token_ids], dim=1)
                
                # Extend attention mask
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=self.device)
                ], dim=1)
                
                # Check if all sequences have generated EOS or reached max length
                if all(eos_generated) or input_ids.shape[1] >= 2048:  # Add a hard cap for safety
                    break
                
                # For efficiency: only re-process if we have more tokens to generate
                if _ < max_new_tokens - 1 and not all(eos_generated):
                    # When using KV cache, we need a special handling for token IDs
                    # The forward method expects text inputs, but we need to leverage the token IDs directly
                    
                    # Process with the existing tokenizer and models with KV caching
                    # We'll create individual inputs for each model to properly use their KV caches
                    new_token_logits = {}
                    new_raw_scores = {}
                    new_past_key_values = {}
                    
                    # Extract just the new token IDs (the last token for each batch item)
                    new_input_ids = input_ids[:, -1:].to(self.device)
                    
                    # Process each model separately with its own KV cache
                    for model_id, model in self.base_models.items():
                        with torch.no_grad():
                            # Get this model's past key-values
                            model_past_kv = past_key_values.get(model_id, None)
                            
                            # Create a minimal input with just the token IDs and attention mask
                            model_inputs = {
                                'input_ids': new_input_ids,
                                'attention_mask': torch.ones_like(new_input_ids, device=self.device),
                                'past_key_values': model_past_kv,
                                'use_cache': True,
                                'output_hidden_states': True
                            }
                            
                            # Process with the model
                            outputs = model(**model_inputs)
                            
                            # Save the outputs
                            new_token_logits[model_id] = outputs.logits.detach()
                            new_past_key_values[model_id] = outputs.past_key_values
                            
                            # Process the projection head scores
                            last_hidden_state = outputs.hidden_states[-1].detach()
                            batch_size, seq_len, _ = last_hidden_state.shape
                            
                            position_scores = torch.zeros(batch_size, seq_len, device=self.device)
                            for pos in range(seq_len):
                                token_representation = last_hidden_state[:, pos, :]
                                position_scores[:, pos] = (
                                    self.projection_heads[model_id](token_representation)
                                    .squeeze(-1)
                                )
                            
                            new_raw_scores[model_id] = position_scores
                    
                    # Process the raw scores into the all_raw_scores format
                    max_seq_len = max(scores.size(1) for scores in new_raw_scores.values())
                    for model_id, scores in new_raw_scores.items():
                        pad_len = max_seq_len - scores.size(1)
                        if pad_len > 0:
                            new_raw_scores[model_id] = torch.nn.functional.pad(
                                scores, (0, pad_len), value=0.0
                            )
                    
                    # Stack the raw scores
                    new_all_raw_scores = torch.stack(
                        [new_raw_scores[model_id] for model_id in self.model_ids],
                        dim=-1
                    )
                    
                    # Update with the new values
                    token_logits = new_token_logits
                    past_key_values = new_past_key_values
                    all_raw_scores = new_all_raw_scores
                    model_confidence = self.get_model_probs(all_raw_scores)
        
        # Extract generated token IDs (excluding prompt)
        generated_parts = [input_ids[b, original_input_length:] for b in range(batch_size)]
        
        # Prepare outputs based on return preferences
        if return_token_ids:
            if batch_size == 1:
                if return_decisions:
                    return {
                        "token_ids": generated_parts[0],
                        "steps": generation_trace[0]
                    }
                else:
                    return generated_parts[0]
            else:
                if return_decisions:
                    return [{
                        "token_ids": generated_parts[i],
                        "steps": generation_trace[i]
                    } for i in range(batch_size)]
                else:
                    return generated_parts
        else:
            # Decode generated tokens as strings
            generated_texts = [tokenizer.decode(generated_parts[b], skip_special_tokens=True) for b in range(batch_size)]
            
            if batch_size == 1:
                if return_decisions:
                    return {
                        "generated_text": generated_texts[0],
                        "steps": generation_trace[0]
                    }
                else:
                    return generated_texts[0]
            else:
                if return_decisions:
                    return [{
                        "generated_text": generated_texts[i],
                        "steps": generation_trace[i]
                    } for i in range(batch_size)]
                else:
                    return generated_texts
        
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
