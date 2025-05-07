import torch
import logging
import time
import gc
from model_loader import ModelLoader
from models import MODELS  # Import your predefined MODELS list

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_all_models():
    # Initialize the loader
    loader = ModelLoader()
    logger.info(f"ModelLoader initialized with device: {loader.device}")
    
    # Create a summary table for results
    results = []
    
    try:
        logger.info(f"Starting test of {len(MODELS)} models")
        logger.info("=" * 50)
        
        for model_info in MODELS:
            model_result = {"name": model_info.name, "id": model_info.id, "loaded": False, "unloaded": False}
            
            try:
                # Try to load the model
                start_time = time.time()
                logger.info(f"Loading model: {model_info.name} (ID: {model_info.id})")
                model, tokenizer = loader.load_model(model_info)
                load_time = time.time() - start_time
                
                # Verify the model is loaded
                is_loaded = loader.is_model_loaded(model_info.id)
                model_result["loaded"] = is_loaded
                logger.info(f"Model {model_info.name} loaded: {is_loaded} (took {load_time:.2f}s)")
                
                if is_loaded:
                    # Select an appropriate test prompt based on the model's task
                    if "code" in model_info.id.lower():
                        test_text = "def fibonacci(n):"
                    elif "math" in model_info.id.lower():
                        test_text = "Prove that the square root of 2 is irrational:"
                    elif "history" in model_info.id.lower():
                        test_text = "The Renaissance period was characterized by:"
                    elif "literature" in model_info.id.lower():
                        test_text = "Once upon a time in a land far away,"
                    else:
                        test_text = "The quick brown fox jumps over the lazy dog."
                    
                    # Tokenize input
                    input_ids = tokenizer(test_text, return_tensors="pt").input_ids.to(loader.device)
                    
                    # Test hidden states
                    try:
                        logger.info(f"Getting hidden states for '{test_text}'")
                        hidden_states = loader.get_hidden_states(model, input_ids)
                        logger.info(f"  Hidden states obtained. Layers: {len(hidden_states)}, Last layer shape: {hidden_states[-1].shape}")
                    except Exception as e:
                        logger.warning(f"  Could not get hidden states: {e}")
                    
                    # Test generation
                    try:
                        logger.info("Testing text generation...")
                        with torch.no_grad():
                            output_ids = model.generate(
                                input_ids, 
                                max_length=30,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9
                            )
                        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        logger.info(f"  Input: '{test_text}'")
                        logger.info(f"  Generated: '{generated_text}'")
                    except Exception as e:
                        logger.warning(f"  Could not generate text: {e}")
                    
                    # Test model caching (load again and verify it's cached)
                    try:
                        logger.info("Testing model caching...")
                        start_time = time.time()
                        cached_model, cached_tokenizer = loader.load_model(model_info)
                        cache_time = time.time() - start_time
                        logger.info(f"  Cached model load took {cache_time:.4f}s")
                        # Verify it's actually cached (should be much faster)
                        model_result["cached"] = cache_time < load_time / 2
                        if model_result["cached"]:
                            logger.info("  Caching verified: Load time significantly reduced")
                        else:
                            logger.warning("  Caching might not be working properly")
                    except Exception as e:
                        logger.warning(f"  Could not test caching: {e}")
                    
                    # Unload the model
                    start_time = time.time()
                    logger.info(f"Unloading model {model_info.name}...")
                    unload_result = loader.unload_model(model_info.id)
                    unload_time = time.time() - start_time
                    model_result["unloaded"] = unload_result
                    logger.info(f"Model {model_info.name} unloaded: {unload_result} (took {unload_time:.2f}s)")
                    
                    # Verify the model is actually unloaded
                    is_still_loaded = loader.is_model_loaded(model_info.id)
                    if is_still_loaded:
                        logger.error(f"  Model {model_info.id} still appears to be loaded after unload!")
                        model_result["unloaded"] = False
                    
            except Exception as e:
                logger.error(f"Error testing model {model_info.name}: {e}")
                model_result["error"] = str(e)
            
            results.append(model_result)
            logger.info("-" * 50)
            
            # Force cleanup after each model
            gc.collect()
            if loader.device == "cuda":
                torch.cuda.empty_cache()
            elif loader.device == "mps":
                torch.mps.empty_cache()
        
        # Try loading multiple models simultaneously
        try:
            logger.info("Testing multiple models loaded simultaneously...")
            # Pick two smaller models to load together
            models_to_load = MODELS[:2]
            loaded_models = []
            
            for model_info in models_to_load:
                logger.info(f"Loading model: {model_info.name}")
                model, tokenizer = loader.load_model(model_info)
                loaded_models.append((model_info.id, model, tokenizer))
                
            # Verify all models are loaded
            all_loaded = all(loader.is_model_loaded(mid) for mid, _, _ in loaded_models)
            logger.info(f"Multiple models loaded successfully: {all_loaded}")
            
            # Unload them all
            for model_id, _, _ in loaded_models:
                loader.unload_model(model_id)
                
        except Exception as e:
            logger.error(f"Error testing multiple models: {e}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("Test Summary:")
        
        success_count = 0
        for result in results:
            cached_status = " (with caching)" if result.get("cached", False) else ""
            status = "SUCCESS" if result.get("loaded") and result.get("unloaded") else "FAILED"
            if status == "SUCCESS":
                success_count += 1
                
            error = f" - Error: {result.get('error')}" if "error" in result else ""
            logger.info(f"{result['name']} ({result['id']}): {status}{cached_status}{error}")
        
        logger.info(f"Models tested: {len(results)}")
        logger.info(f"Models succeeded: {success_count}")
        logger.info(f"Success rate: {success_count/len(results)*100:.1f}%")
        
        # Overall result
        logger.info(f"Overall test result: {'SUCCESS' if success_count == len(results) else 'PARTIAL SUCCESS' if success_count > 0 else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        # Make sure we clean up
        loader.unload_all_models()
        logger.info("Test completed, all models unloaded")

if __name__ == "__main__":
    test_all_models()