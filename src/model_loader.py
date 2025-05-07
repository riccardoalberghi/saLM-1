import torch
import gc
from typing import List, Dict, Any, Tuple, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    id: str  # Unique identifier for the model
    name: str  # Human-readable name
    path: str  # Path or HuggingFace model ID
    parameters: int  # Number of parameters in the model
    task: str  # Task the model is designed for

class ModelLoader:
    """Manages loading, caching, and unloading of transformer models."""
    
    def __init__(self, device: str = None):
        """Initialize the model loader with specified device or auto-detect optimal device."""
        self.device = device if device else ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
        self.loaded_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}
        self.model_usage_count: Dict[str, int] = {}  # Track usage count for each model

        # Auto-detect best available device if none specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        # Ensure device is valid
        assert self.device in ["cuda", "cpu", "mps"], f"Invalid device: {self.device}"
        
    def load_model(self, model_info: ModelInfo) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load a model and its tokenizer or return cached version if already loaded."""
        assert model_info is not None, "Model info cannot be None"
        assert model_info.id, "Model ID cannot be empty"
        assert model_info.path, "Model path cannot be empty"
        
        model_id = model_info.id

        # Return cached model if already loaded
        if model_id in self.loaded_models:
            logger.debug(f"Model {model_id} already loaded, returning cached instance")
            self.model_usage_count[model_id] += 1
            return self.loaded_models[model_id]
    
        try:
            logger.info(f"Loading tokenizer for {model_info.name} from {model_info.path}")
            tokenizer = AutoTokenizer.from_pretrained(model_info.path)
            assert tokenizer is not None, "Failed to load tokenizer"
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_info.name}: {e}")
            raise 

        try:
            logger.info(f"Loading model {model_info.name} from {model_info.path}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_info.path, #add args+ if needed
            )
            
            assert model is not None, "Failed to load model"
            # Move model to appropriate device
            model = model.to(self.device)
            
            logger.info(f"Successfully loaded {model_info.name}")
            
            self.loaded_models[model_id] = (model, tokenizer)
            self.model_usage_count[model_id] = 1
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Failed to load model {model_info.name}: {e}")
            del tokenizer
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            raise 

    def unload_model(self, model_id: str) -> bool:
        """Unload a model and its tokenizer, freeing up memory resources."""
        assert isinstance(model_id, str), "Model ID must be a string"
        
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded, cannot unload")
            return False
        
        model, tokenizer = self.loaded_models[model_id]

        del self.loaded_models[model_id]
        if model_id in self.model_usage_count:
            del self.model_usage_count[model_id]

        del model
        del tokenizer 

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        
        logger.info(f"Unloaded model {model_id}")
        return True
    
    def unload_all_models(self) -> None:
        """Unload all currently loaded models to free up all memory."""
        model_ids = list(self.loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        logger.info("All models unloaded")
        
        # Verify all models were unloaded
        assert len(self.loaded_models) == 0, "Failed to unload all models"
    
    def get_model(self, model_info: ModelInfo) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Convenience method to load a model and return it, or get from cache if already loaded."""
        assert model_info is not None, "Model info cannot be None"
        return self.load_model(model_info)
        
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is currently loaded in memory."""
        assert isinstance(model_id, str), "Model ID must be a string"
        return model_id in self.loaded_models
    
    def get_hidden_states(self, model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from a model for a given input, useful for embeddings and analysis."""
        assert model is not None, "Model cannot be None"
        assert isinstance(input_ids, torch.Tensor), "Input IDs must be a torch.Tensor"
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            assert hidden_states is not None, "Model did not return hidden states"
        return hidden_states
    
    def __del__(self):
        """Clean up resources when the ModelLoader instance is being deleted."""
        self.unload_all_models()
        logger.info("ModelLoader instance deleted")