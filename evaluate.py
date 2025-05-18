#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Script for Multi-Model Architecture with Scalar Heads

This script evaluates the performance of the MultiModelWithScalarHeads ensemble model
and its constituent models on two datasets: GSM8K (math problems) and MedQA (medical questions).

The script builds upon the MultiModelWithScalarHeads.generate method and implements
comprehensive evaluation metrics for both datasets.
"""

import os
import sys
import json
import re
import time
import logging
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
from collections import Counter, defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify

# Import local modules
from src.architecture import MultiModelWithScalarHeads
from src.model_loader import ModelLoader, ModelInfo
from src.models import MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = "evaluation_results"
DEFAULT_SAMPLE_SIZE = 50  # Default number of examples to evaluate
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MultiModelWithScalarHeads on GSM8K and MedQA")
    
    # Dataset options
    parser.add_argument("--datasets", type=str, nargs="+", choices=["gsm8k", "medqa", "all"], 
                        default=["all"], help="Datasets to evaluate on")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE,
                       help=f"Number of examples to evaluate (default: {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_SEED,
                       help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    
    # Model options
    parser.add_argument("--model_weights", type=str, default=None,
                       help="Path to MultiModelWithScalarHeads weights file")
    parser.add_argument("--only_ensemble", action="store_true",
                       help="Only evaluate the ensemble model, not individual models")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                       help=f"Maximum number of tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help=f"Temperature for text generation (default: {DEFAULT_TEMPERATURE})")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--save_examples", action="store_true",
                       help="Save detailed examples for analysis")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of examples to save when --save_examples is used (default: 5)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Disable generation of result plots")
    
    # Device options
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on (default: auto-detect)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with minimal samples and verbose output")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_datasets(args) -> Dict[str, Any]:
    """Load and prepare the datasets for evaluation."""
    datasets = {}
    
    # Determine which datasets to load
    eval_gsm8k = "all" in args.datasets or "gsm8k" in args.datasets
    eval_medqa = "all" in args.datasets or "medqa" in args.datasets
    
    # Set a small sample size for debug mode
    if args.debug:
        sample_size = 5
        logger.info(f"Debug mode enabled. Using {sample_size} examples for each dataset.")
    else:
        sample_size = args.sample_size
    
    # Load GSM8K dataset
    if eval_gsm8k:
        logger.info(f"Loading GSM8K dataset...")
        full_gsm8k = load_dataset("cola13/gsm8k_formatted", split="test")
        logger.info(f"Loaded {len(full_gsm8k)} GSM8K examples. Selecting {sample_size} random examples...")
        
        # Select random subset
        random_indices = random.sample(range(len(full_gsm8k)), min(sample_size, len(full_gsm8k)))
        datasets["gsm8k"] = full_gsm8k.select(random_indices)
        logger.info(f"Selected {len(datasets['gsm8k'])} GSM8K examples.")
    
    # Load MedQA dataset
    if eval_medqa:
        logger.info(f"Loading MedQA dataset...")
        full_medqa = load_dataset("cola13/medqa_formatted", split="test")
        logger.info(f"Loaded {len(full_medqa)} MedQA examples. Selecting {sample_size} random examples...")
        
        # Select random subset
        random_indices = random.sample(range(len(full_medqa)), min(sample_size, len(full_medqa)))
        datasets["medqa"] = full_medqa.select(random_indices)
        logger.info(f"Selected {len(datasets['medqa'])} MedQA examples.")
    
    return datasets


def load_models(args) -> Dict[str, Any]:
    """Load the ensemble model and constituent models."""
    models = {}
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else 
                                            "mps" if torch.backends.mps.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load ensemble model
    logger.info("Loading ensemble model...")
    
    if args.model_weights:
        ensemble_model = MultiModelWithScalarHeads.from_pretrained(
            MODELS, args.model_weights, device=device
        )
        logger.info(f"Loaded ensemble model from weights: {args.model_weights}")
    else:
        ensemble_model = MultiModelWithScalarHeads(
            base_models=MODELS, device=device
        )
        logger.info("Initialized ensemble model with default weights")
    
    # Set model to evaluation mode
    ensemble_model.eval()
    models["ensemble"] = {
        "model": ensemble_model,
        "tokenizer": ensemble_model.common_tokenizer
    }
    
    if not args.only_ensemble:
        # Initialize model loader
        logger.info("Loading individual constituent models...")
        model_loader = ModelLoader(device=device)
        
        # Load individual models
        for model_info in MODELS:
            logger.info(f"Loading individual model: {model_info.name}")
            model, tokenizer = model_loader.get_model(model_info)
            model.eval()
            
            models[model_info.id] = {
                "model": model,
                "tokenizer": tokenizer,
                "info": model_info
            }
    
    return models


def generate_with_model(model, tokenizer, prompt: str, args) -> str:
    """Generate text using a model with appropriate parameters."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": 1,
        "temperature": args.temperature,
        "top_p": 0.92,
        "top_k": 40
    }
    
    with torch.no_grad():
        output = model.generate(**inputs, **gen_params)
    
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    response = full_text[len(prompt):].strip()
    return response


def extract_medqa_answer(text: str) -> Optional[str]:
    """
    Extract answer letter from generated text for MedQA problems.
    
    Args:
        text: The generated text to extract answer from
        
    Returns:
        str or None: The extracted answer letter (A-E) or None if not found
    """
    # Look for patterns like "The correct answer is: X)" or just "X)"
    match = re.search(r'(?:The correct answer is:?\s*)?([A-E])\)', text)
    if match:
        letter = match.group(1).upper()
        return letter
        
    # If no match with the above pattern, try a simpler pattern to find any letter
    letter_match = re.search(r'\b([A-E])\b', text)
    if letter_match:
        letter = letter_match.group(1).upper()
        return letter
        
    return None


def verify_math_answer(answer_text: str, correct_answer_text: str) -> Dict[str, Any]:
    """
    Use math_verify to verify if the math answer is correct.
    
    Args:
        answer_text: The model's reasoning and answer text
        correct_answer_text: The expected correct answer text
        
    Returns:
        Dict containing verification results
    """
    verification = {
        "final_answer_correct": False
    }
    
    try:
        # Parse the gold answer and model answer using math_verify
        gold = parse(correct_answer_text)
        model_answer = parse(answer_text)
        
        # Verify if the answers match
        verification["final_answer_correct"] = verify(gold, model_answer)
        
    except Exception as e:
        logger.warning(f"Error using math_verify: {e}")
    
    return verification


def evaluate_gsm8k(models: Dict[str, Any], dataset, args) -> Dict[str, Any]:
    """
    Evaluate models on GSM8K math reasoning dataset.
    
    Args:
        models: Dictionary of models to evaluate
        dataset: GSM8K dataset with 'question' and 'answer' columns
        args: Command-line arguments
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory for GSM8K results
    output_dir = os.path.join(args.output_dir, "gsm8k")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Evaluating on GSM8K ({len(dataset)} examples)...")
    
    # Select example indices if saving examples
    example_indices = random.sample(range(len(dataset)), min(args.num_examples, len(dataset))) if args.save_examples else []
    
    # Initialize results structure
    results = {
        "total_examples": len(dataset),
        "models": {},
        "examples": [] if args.save_examples else None
    }
    
    # Initialize metrics for each model
    for model_id in models:
        results["models"][model_id] = {
            "correct": 0,
            "answer_found": 0,
            "generated_lengths": [],
            "answer_distribution": Counter(),
            "error_cases": []
        }
    
    # Generation parameters optimized for math reasoning
    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature
    }
    
    # Evaluate each example
    for i, example in enumerate(tqdm(dataset, desc="Evaluating GSM8K")):
        question = example['question']
        correct_answer = example['answer']
        
        # Format prompt with chain-of-thought instruction
        prompt = f'I want you to solve the following math problem step by step.\n{question}\n\nStart by reasoning about the problem, then provide the numerical answer in the format "Therefore, the answer is: # <number>"'
        
        # Track if this is an example to save
        is_example = i in example_indices
        example_data = {
            "question": question, 
            "correct_answer_text": correct_answer_text, 
            "correct_answer": correct_answer,
            "model_answers": {}
        } if is_example else None
        
        # Evaluate each model
        for model_id, model_dict in models.items():
            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]
            
            try:
                # Generate answer with the model
                if model_id == "ensemble":
                    # Use the ensemble model's generate method
                    with torch.no_grad():
                        answer_text = model.generate(
                            prompt=prompt,
                            max_new_tokens=gen_params["max_new_tokens"],
                            temperature=gen_params["temperature"],
                            return_token_ids=False,
                            return_decisions=False
                        )
                else:
                    # For individual models, we need to handle them differently
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=gen_params["max_new_tokens"],
                            temperature=gen_params["temperature"],
                            top_p=0.92,
                            top_k=40
                        )
                    
                    # Decode the full text and extract only the generated part
                    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer_text = full_text[len(prompt):].strip()
                
                # Record answer length
                results["models"][model_id]["generated_lengths"].append(len(answer_text.split()))
                
                # Perform verification using math_verify
                verification = verify_math_answer(answer_text, correct_answer)
                
                # Record answer stats if an answer was extracted (for distribution tracking only)
                results["models"][model_id]["answer_found"] += 1
                results["models"][model_id]["answer_distribution"][str(extracted_answer)] += 1
                
                # Count correct answers based on math_verify verification
                if verification["final_answer_correct"]:
                    results["models"][model_id]["correct"] += 1
                
                # Save example if needed
                if is_example:
                    example_data["model_answers"][model_id] = {
                        "answer_text": answer_text,
                        "verification": verification
                    }
                
            except Exception as e:
                logger.error(f"Error with {model_id} model on example {i}: {e}")
                results["models"][model_id]["error_cases"].append({
                    "index": i, 
                    "question": question, 
                    "error": str(e)
                })
                
                if is_example:
                    example_data["model_answers"][model_id] = {
                        "answer_text": f"Error: {str(e)}",
                        "verification": None
                    }
        
        # Save example if needed
        if is_example:
            results["examples"].append(example_data)
    
    # Calculate overall metrics for each model
    for model_id in models:
        model_results = results["models"][model_id]
        total = results["total_examples"]
        
        # Calculate accuracy
        model_results["accuracy"] = model_results["correct"] / total
        model_results["answer_detection_rate"] = model_results["answer_found"] / total
        
        # No step verification with math_verify approach
        
        # Calculate average response length
        model_results["avg_generated_length"] = (
            np.mean(model_results["generated_lengths"]) 
            if model_results["generated_lengths"] else 0
        )
        
        # Convert answer distribution to sorted dictionary (keep top 20)
        sorted_dist = dict(sorted(
            model_results["answer_distribution"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])
        model_results["answer_distribution"] = sorted_dist
    
    # Print results summary
    logger.info("\n" + "="*80)
    logger.info("GSM8K TESTING RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total test questions: {results['total_examples']}")
    
    logger.info("\nAccuracy (correct numeric answer):")
    for model_id in models:
        acc = results["models"][model_id]["accuracy"]
        correct = results["models"][model_id]["correct"]
        total = results["total_examples"]
        logger.info(f"  {model_id}: {correct} / {total} = {acc:.2%}")
    
    # Calculate improvement of ensemble over best individual model (if applicable)
    if "ensemble" in results["models"] and len(models) > 1:
        ensemble_acc = results["models"]["ensemble"]["accuracy"]
        individual_accs = [results["models"][model_id]["accuracy"] 
                         for model_id in models if model_id != "ensemble"]
        
        if individual_accs:
            best_individual = max(individual_accs)
            results["ensemble_improvement"] = ensemble_acc - best_individual
            logger.info(f"\nEnsemble improvement over best individual model: {results['ensemble_improvement']:.2%}")
    
    # Save results to json file
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"GSM8K results saved to {results_file}")
    
    # Create visualizations
    if not args.no_plots:
        create_gsm8k_plots(results, output_dir)
    
    return results


def evaluate_medqa(models: Dict[str, Any], dataset, args) -> Dict[str, Any]:
    """
    Evaluate models on MedQA medical question answering dataset.
    
    Args:
        models: Dictionary of models to evaluate
        dataset: MedQA dataset with 'question' and 'answer' columns
        args: Command-line arguments
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory for MedQA results
    output_dir = os.path.join(args.output_dir, "medqa")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Evaluating on MedQA ({len(dataset)} examples)...")
    
    # Select example indices if saving examples
    example_indices = random.sample(range(len(dataset)), min(args.num_examples, len(dataset))) if args.save_examples else []
    
    # Initialize results structure
    results = {
        "total_examples": len(dataset),
        "models": {},
        "examples": [] if args.save_examples else None,
        "random": {
            "correct": 0,
            "letter_distribution": Counter()
        }
    }
    
    # Initialize metrics for each model
    for model_id in models:
        results["models"][model_id] = {
            "correct": 0,
            "letter_found": 0,
            "generated_lengths": [],
            "letter_distribution": Counter(),
            "error_cases": []
        }
    
    # Generation parameters optimized for multiple-choice answer generation
    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature
    }
    
    # Evaluate each example
    for i, example in enumerate(tqdm(dataset, desc="Evaluating MedQA")):
        question = example['question']
        correct_answer_text = example['answer']
        
        # Extract correct answer letter
        correct_letter = extract_medqa_answer(correct_answer_text)
        if not correct_letter:
            logger.warning(f"Could not extract letter from correct answer: {correct_answer_text}")
            correct_letter = "Unknown"  # Fall back
        
        # Format prompt (just use the question as is)
        prompt = f'I want you to reason and answer on the following question.\n{question}\n\nStart by reasoning about the question, then provide the answer in the format "The answer is X) <answer>\n Reasoning:"'
        
        # Track if this is an example to save
        is_example = i in example_indices
        example_data = {
            "question": question, 
            "correct_answer": correct_answer_text,
            "correct_letter": correct_letter,
            "model_answers": {}
        } if is_example else None
        
        # Generate random choice - simulate random guessing baseline
        # Assume A, B, C, D, E are possible options
        available_options = ["A", "B", "C", "D", "E"]
        random_letter = random.choice(available_options)
        
        # Record random baseline metrics
        results["random"]["letter_distribution"][random_letter] += 1
        if random_letter == correct_letter:
            results["random"]["correct"] += 1
        
        if is_example:
            example_data["random_letter"] = random_letter
        
        # Evaluate each model
        for model_id, model_dict in models.items():
            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]
            
            try:
                # Generate answer with the model
                if model_id == "ensemble":
                    # Use the ensemble model's generate method
                    with torch.no_grad():
                        answer_text = model.generate(
                            prompt=prompt,
                            max_new_tokens=gen_params["max_new_tokens"],
                            temperature=gen_params["temperature"],
                            return_token_ids=False,
                            return_decisions=False
                        )
                else:
                    # For individual models, we need to handle them differently
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=gen_params["max_new_tokens"],
                            temperature=gen_params["temperature"],
                            top_p=0.92,
                            top_k=40
                        )
                    
                    # Decode the full text and extract only the generated part
                    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer_text = full_text[len(prompt):].strip()
                
                # Record answer length
                results["models"][model_id]["generated_lengths"].append(len(answer_text.split()))
                
                # Extract answer letter
                extracted_letter = extract_medqa_answer(answer_text)
                
                # Record answer stats
                if extracted_letter:
                    results["models"][model_id]["letter_found"] += 1
                    results["models"][model_id]["letter_distribution"][extracted_letter] += 1
                    
                    # Check if the answer is correct
                    if extracted_letter == correct_letter:
                        results["models"][model_id]["correct"] += 1
                
                # Save example if needed
                if is_example:
                    example_data["model_answers"][model_id] = {
                        "answer_text": answer_text,
                        "extracted_letter": extracted_letter
                    }
                
            except Exception as e:
                logger.error(f"Error with {model_id} model on example {i}: {e}")
                results["models"][model_id]["error_cases"].append({
                    "index": i, 
                    "question": question, 
                    "error": str(e)
                })
                
                if is_example:
                    example_data["model_answers"][model_id] = {
                        "answer_text": f"Error: {str(e)}",
                        "extracted_letter": None
                    }
        
        # Save example if needed
        if is_example and example_data:
            results["examples"].append(example_data)
    
    # Calculate overall metrics for each model
    for model_id in models:
        model_results = results["models"][model_id]
        total = results["total_examples"]
        
        # Calculate accuracy and letter detection rate
        model_results["accuracy"] = model_results["correct"] / total
        model_results["letter_detection_rate"] = model_results["letter_found"] / total
        
        # Calculate average response length
        model_results["avg_generated_length"] = (
            np.mean(model_results["generated_lengths"]) 
            if model_results["generated_lengths"] else 0
        )
        
        # Convert letter distribution to dictionary for JSON serialization
        model_results["letter_distribution"] = dict(model_results["letter_distribution"])
    
    # Calculate random baseline accuracy
    results["random"]["accuracy"] = results["random"]["correct"] / results["total_examples"]
    results["random"]["letter_detection_rate"] = 1.0  # Always detects a letter by definition
    results["random"]["letter_distribution"] = dict(results["random"]["letter_distribution"])
    
    # Calculate improvements
    if "ensemble" in results["models"] and len(models) > 1:
        # Calculate improvement over random baseline
        ensemble_acc = results["models"]["ensemble"]["accuracy"]
        random_acc = results["random"]["accuracy"]
        results["ensemble_improvement_over_random"] = ensemble_acc - random_acc
        
        # Calculate improvement over best individual model
        individual_accs = [results["models"][model_id]["accuracy"] 
                         for model_id in models if model_id != "ensemble"]
        if individual_accs:
            best_individual = max(individual_accs)
            results["ensemble_improvement_over_best"] = ensemble_acc - best_individual
    
    # Print results summary
    logger.info("\n" + "="*80)
    logger.info("MEDQA TESTING RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total test questions: {results['total_examples']}")
    
    logger.info("\nAccuracy (correct answer letter):")
    for model_id in models:
        acc = results["models"][model_id]["accuracy"]
        correct = results["models"][model_id]["correct"]
        total = results["total_examples"]
        logger.info(f"  {model_id}: {correct} / {total} = {acc:.2%}")
    
    logger.info(f"  Random baseline: {results['random']['correct']} / {total} = {results['random']['accuracy']:.2%}")
    
    # Print improvements if available
    if "ensemble_improvement_over_random" in results:
        logger.info(f"\nEnsemble improvement over random: {results['ensemble_improvement_over_random']:.2%}")
    
    if "ensemble_improvement_over_best" in results:
        logger.info(f"Ensemble improvement over best individual model: {results['ensemble_improvement_over_best']:.2%}")
    
    # Save results to json file
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"MedQA results saved to {results_file}")
    
    # Create visualizations
    if not args.no_plots:
        create_medqa_plots(results, output_dir)
    
    return results


def create_gsm8k_plots(results: Dict[str, Any], output_dir: str):
    """
    Create visualization plots for GSM8K evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save plots
    """
    try:
        plt.figure(figsize=(20, 15))
        
        # Get model IDs and ensure ensemble is first if it exists
        model_ids = list(results["models"].keys())
        if "ensemble" in model_ids:
            model_ids.remove("ensemble")
            model_ids = ["ensemble"] + model_ids
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        accuracies = [results["models"][model_id]["accuracy"] for model_id in model_ids]
        plt.bar(model_ids, accuracies, color=['green' if model_id == 'ensemble' else 'blue' for model_id in model_ids])
        plt.title('Accuracy Comparison', fontsize=14)
        plt.ylabel('Accuracy')
        plt.ylim(0, max(max(accuracies) * 1.2, 0.1))  # Set y-limit with headroom
        plt.xticks(rotation=45, ha="right")
        
        # Add accuracy values on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
        
        # Answer detection rate
        plt.subplot(2, 2, 2)
        detection_rates = [results["models"][model_id]["answer_detection_rate"] for model_id in model_ids]
        plt.bar(model_ids, detection_rates, color=['green' if model_id == 'ensemble' else 'blue' for model_id in model_ids])
        plt.title('Answer Detection Rate', fontsize=14)
        plt.ylabel('Rate')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha="right")
        
        # Add detection rate values on top of bars
        for i, v in enumerate(detection_rates):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
        
        # Average response length
        plt.subplot(2, 2, 3)
        avg_lengths = [results["models"][model_id]["avg_generated_length"] for model_id in model_ids]
        plt.bar(model_ids, avg_lengths, color=['green' if model_id == 'ensemble' else 'blue' for model_id in model_ids])
        plt.title('Average Response Length', fontsize=14)
        plt.ylabel('Number of words')
        plt.xticks(rotation=45, ha="right")
        
        # Add length values on top of bars
        for i, v in enumerate(avg_lengths):
            plt.text(i, v + 1, f"{v:.1f}", ha='center', fontweight='bold')
        
        # Top answers distribution for ensemble (if available)
        plt.subplot(2, 2, 4)
        if "ensemble" in results["models"]:
            top_answers = list(results["models"]["ensemble"]["answer_distribution"].keys())[:10]
            if top_answers:
                counts = [results["models"]["ensemble"]["answer_distribution"][ans] for ans in top_answers]
                plt.barh(range(len(top_answers)), counts, color='green')
                plt.yticks(range(len(top_answers)), top_answers)
                plt.title('Top 10 Ensemble Model Answers', fontsize=14)
                plt.xlabel('Count')
            else:
                plt.text(0.5, 0.5, 'No answer distribution data available', 
                       horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(0.5, 0.5, 'No ensemble model data available', 
                   horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Save chart
        chart_file = os.path.join(output_dir, 'performance_chart.png')
        plt.savefig(chart_file)
        logger.info(f"GSM8K performance chart saved to {chart_file}")
        plt.close()
        
        # Create another chart for comparison of answer distributions across models
        if len(model_ids) > 1:
            plt.figure(figsize=(15, 10))
            
            # Collect top answers across all models
            all_top_answers = set()
            for model_id in model_ids:
                top_model_answers = list(results["models"][model_id]["answer_distribution"].keys())[:5]
                all_top_answers.update(top_model_answers)
            
            all_top_answers = list(all_top_answers)[:15]  # Limit to top 15 for readability
            
            if all_top_answers:
                # Set up bar positions
                x = np.arange(len(all_top_answers))
                width = 0.8 / len(model_ids)
                
                # Plot bars for each model
                for i, model_id in enumerate(model_ids):
                    counts = [results["models"][model_id]["answer_distribution"].get(ans, 0) 
                             for ans in all_top_answers]
                    pos = x + (i - len(model_ids)/2 + 0.5) * width
                    plt.bar(pos, counts, width, label=model_id,
                          color='green' if model_id == 'ensemble' else f'C{i}')
                
                plt.xlabel('Answer')
                plt.ylabel('Count')
                plt.title('Top Answers Comparison Across Models', fontsize=14)
                plt.xticks(x, all_top_answers, rotation=45, ha="right")
                plt.legend()
                
                plt.tight_layout()
                
                comparison_file = os.path.join(output_dir, 'answer_comparison.png')
                plt.savefig(comparison_file)
                logger.info(f"GSM8K answer comparison chart saved to {comparison_file}")
            
            plt.close()
    
    except Exception as e:
        logger.error(f"Error creating GSM8K plots: {e}")


def create_medqa_plots(results: Dict[str, Any], output_dir: str):
    """
    Create visualization plots for MedQA evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save plots
    """
    try:
        plt.figure(figsize=(20, 15))
        
        # Get model IDs and ensure ensemble is first if it exists
        model_ids = list(results["models"].keys())
        if "ensemble" in model_ids:
            model_ids.remove("ensemble")
            model_ids = ["ensemble"] + model_ids
        
        model_labels = model_ids + ["Random"]
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        accuracies = [results["models"][model_id]["accuracy"] for model_id in model_ids]
        accuracies.append(results["random"]["accuracy"])
        
        colors = ['green' if label == 'ensemble' else 'blue' if label != 'Random' else 'red' 
                for label in model_labels]
        
        plt.bar(model_labels, accuracies, color=colors)
        plt.title('Accuracy Comparison', fontsize=14)
        plt.ylabel('Accuracy')
        plt.ylim(0, max(max(accuracies) * 1.2, 0.1))  # Set y-limit with headroom
        plt.xticks(rotation=45, ha="right")
        
        # Add accuracy values on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
        
        # Letter detection rate comparison
        plt.subplot(2, 2, 2)
        detection_rates = [results["models"][model_id]["letter_detection_rate"] for model_id in model_ids]
        detection_rates.append(results["random"]["letter_detection_rate"])  # Random is 1.0 by definition
        
        plt.bar(model_labels, detection_rates, color=colors)
        plt.title('Letter Detection Rate', fontsize=14)
        plt.ylabel('Rate')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha="right")
        
        # Add rate values on top of bars
        for i, v in enumerate(detection_rates):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
        
        # Letter distribution for ensemble (if available)
        plt.subplot(2, 2, 3)
        if "ensemble" in results["models"]:
            ensemble_letters = sorted(results["models"]["ensemble"]["letter_distribution"].keys())
            if ensemble_letters:
                counts = [results["models"]["ensemble"]["letter_distribution"][letter] for letter in ensemble_letters]
                plt.bar(ensemble_letters, counts, color='green')
                plt.title('Ensemble Model Letter Distribution', fontsize=14)
                plt.xlabel('Letter')
                plt.ylabel('Count')
            else:
                plt.text(0.5, 0.5, 'No letter distribution data available', 
                       horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(0.5, 0.5, 'No ensemble model data available', 
                   horizontalalignment='center', verticalalignment='center')
        
        # Letter distribution comparison (all models + random)
        plt.subplot(2, 2, 4)
        
        # Collect all letters
        all_letters = set()
        for model_id in model_ids:
            all_letters.update(results["models"][model_id]["letter_distribution"].keys())
        all_letters.update(results["random"]["letter_distribution"].keys())
        all_letters = sorted(all_letters)
        
        if all_letters:
            x = np.arange(len(all_letters))
            width = 0.8 / (len(model_ids) + 1)  # +1 for random
            
            # Plot bars for each model and random
            for i, model_id in enumerate(model_ids):
                letter_counts = [results["models"][model_id]["letter_distribution"].get(letter, 0) 
                                for letter in all_letters]
                pos = x + (i - len(model_ids)/2) * width
                plt.bar(pos, letter_counts, width, label=model_id,
                      color='green' if model_id == 'ensemble' else f'C{i}')
            
            # Add random distribution
            random_counts = [results["random"]["letter_distribution"].get(letter, 0) 
                           for letter in all_letters]
            pos = x + (len(model_ids) - len(model_ids)/2) * width
            plt.bar(pos, random_counts, width, label='Random', color='red')
            
            plt.xlabel('Letter')
            plt.ylabel('Count')
            plt.title('Letter Distribution Comparison', fontsize=14)
            plt.xticks(x, all_letters)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No letter distribution data available', 
                   horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Save chart
        chart_file = os.path.join(output_dir, 'performance_chart.png')
        plt.savefig(chart_file)
        logger.info(f"MedQA performance chart saved to {chart_file}")
        
        plt.close()
    
    except Exception as e:
        logger.error(f"Error creating MedQA plots: {e}")


def main():
    """Main function to run the evaluation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log execution parameters
    logger.info(f"Starting evaluation with parameters:")
    logger.info(f"  Datasets: {args.datasets}")
    logger.info(f"  Sample size: {args.sample_size}")
    logger.info(f"  Random seed: {args.random_seed}")
    logger.info(f"  Model weights: {args.model_weights or 'None (using default)'}")
    logger.info(f"  Only evaluate ensemble: {args.only_ensemble}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    # Load datasets
    datasets = load_and_prepare_datasets(args)
    
    if not datasets:
        logger.error("No datasets were loaded. Exiting.")
        return
    
    # Load models
    try:
        models = load_models(args)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return
    
    # Evaluate on each dataset
    results = {}
    
    start_time = time.time()
    
    if "gsm8k" in datasets:
        try:
            logger.info("Starting GSM8K evaluation...")
            results["gsm8k"] = evaluate_gsm8k(models, datasets["gsm8k"], args)
            logger.info("GSM8K evaluation completed.")
        except Exception as e:
            logger.error(f"Error during GSM8K evaluation: {e}")
    
    if "medqa" in datasets:
        try:
            logger.info("Starting MedQA evaluation...")
            results["medqa"] = evaluate_medqa(models, datasets["medqa"], args)
            logger.info("MedQA evaluation completed.")
        except Exception as e:
            logger.error(f"Error during MedQA evaluation: {e}")
    
    # Calculate total time taken
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    
    # Save summary statistics across all datasets
    summary = {
        "runtime_seconds": total_time,
        "datasets_evaluated": list(results.keys()),
        "num_models": len(models),
        "model_ids": list(models.keys()),
        "parameters": vars(args),
        "results": {}
    }
    
    # Extract accuracy metrics for quick reference
    for dataset_name, dataset_results in results.items():
        summary["results"][dataset_name] = {
            "model_accuracies": {
                model_id: dataset_results["models"][model_id]["accuracy"]
                for model_id in models
            }
        }
        
        # Add random baseline for MedQA
        if dataset_name == "medqa":
            summary["results"][dataset_name]["random_accuracy"] = dataset_results["random"]["accuracy"]
        
        # Add improvement metrics if available
        if "ensemble_improvement" in dataset_results:
            summary["results"][dataset_name]["ensemble_improvement"] = dataset_results["ensemble_improvement"]
    
    # Save summary to file
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation summary saved to {summary_file}")
    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
