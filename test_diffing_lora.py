"""
Test script for model-diffing with LoRA adapter models:
- Initial model: Qwen/Qwen2.5-7B-Instruct
- Fine-tuned model: models/emergent-misalignment/qwen_medical_misaligned (LoRA adapter)
"""

import torch
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from train import ITDA


def test_lora_model_diffing():
    """
    Test model-diffing with LoRA adapter models using HuggingFace transformers directly.
    """
    
    # Configuration
    itda_path = "artifacts/runs/bH0ZrUOL"
    initial_model_name = "Qwen/Qwen2.5-7B-Instruct"
    fine_tuned_model_path = "models/emergent-misalignment/qwen_medical_misaligned"
    layer = 21
    evaluation_dataset = "NeelNanda/pile-10k"
    
    print("="*70)
    print("MODEL-DIFFING TEST WITH LoRA ADAPTERS")
    print("="*70)
    print(f"ITDA path: {itda_path}")
    print(f"Initial model: {initial_model_name}")
    print(f"Fine-tuned model: {fine_tuned_model_path}")
    print(f"Layer: {layer}")
    print(f"Evaluation dataset: {evaluation_dataset}")
    print("="*70)
    
    # Check if paths exist
    if not os.path.exists(itda_path):
        print(f"ERROR: ITDA path {itda_path} does not exist!")
        return
    
    if not os.path.exists(fine_tuned_model_path):
        print(f"ERROR: Fine-tuned model path {fine_tuned_model_path} does not exist!")
        return
    
    # Load ITDA
    try:
        print("Loading ITDA...")
        itda = ITDA.from_pretrained(itda_path, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ ITDA loaded with {itda.dict_size} atoms")
    except Exception as e:
        print(f"ERROR loading ITDA: {e}")
        return
    
    # Load tokenizer
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(initial_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        return
    
    # Load initial model
    try:
        print("Loading initial model...")
        model_initial = AutoModelForCausalLM.from_pretrained(
            initial_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Initial model loaded")
    except Exception as e:
        print(f"ERROR loading initial model: {e}")
        return
    
    # Load fine-tuned model (LoRA adapter)
    try:
        print("Loading fine-tuned model (LoRA adapter)...")
        model_fine_tuned = PeftModel.from_pretrained(
            model_initial,
            fine_tuned_model_path
        )
        print("✓ Fine-tuned model loaded")
    except Exception as e:
        print(f"ERROR loading fine-tuned model: {e}")
        return
    
    # Collect activations
    print("\n" + "="*50)
    print("STEP 1: Collecting activations")
    print("="*50)
    
    seq_len = 128
    batch_size = 8
    max_steps = 10
    
    # Load dataset
    dataset = load_dataset(evaluation_dataset, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)
    
    # Collect activations for initial model
    print("Collecting activations for initial model...")
    activations_M = collect_activations_hf(
        model_initial, 
        itda, 
        tokenizer, 
        data_stream, 
        seq_len, 
        batch_size, 
        max_steps,
        layer
    )
    
    # Reset data stream
    dataset = load_dataset(evaluation_dataset, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)
    
    # Collect activations for fine-tuned model
    print("Collecting activations for fine-tuned model...")
    activations_MD = collect_activations_hf(
        model_fine_tuned, 
        itda, 
        tokenizer, 
        data_stream, 
        seq_len, 
        batch_size, 
        max_steps,
        layer
    )
    
    # Compute differences
    print("\n" + "="*50)
    print("STEP 2: Computing activation differences")
    print("="*50)
    
    # Compute average activations
    avg_activations_M = activations_M.mean(dim=(0, 1))
    avg_activations_MD = activations_MD.mean(dim=(0, 1))
    
    # Compute differences
    activation_differences = avg_activations_MD - avg_activations_M
    
    print(f"Activation differences shape: {activation_differences.shape}")
    
    # Order latents by increase
    print("\n" + "="*50)
    print("STEP 3: Ordering latents by activation increase")
    print("="*50)
    
    sorted_differences, sorted_indices = torch.sort(activation_differences, descending=True)
    
    print(f"Top 20 latents with largest activation increases:")
    print("-" * 50)
    for i in range(min(20, len(sorted_indices))):
        latent_idx = sorted_indices[i].item()
        diff = sorted_differences[i].item()
        print(f"{i+1:2d}. Latent {latent_idx:4d}: {diff:+.6f}")
    
    # Show statistics
    print(f"\nStatistics:")
    print(f"  Mean difference: {activation_differences.mean().item():.6f}")
    print(f"  Std difference: {activation_differences.std().item():.6f}")
    print(f"  Max increase: {activation_differences.max().item():.6f}")
    print(f"  Max decrease: {activation_differences.min().item():.6f}")
    print(f"  Positive changes: {(activation_differences > 0).sum().item()}")
    print(f"  Negative changes: {(activation_differences < 0).sum().item()}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return {
        'activation_differences': activation_differences,
        'sorted_indices': sorted_indices,
        'sorted_differences': sorted_differences,
        'activations_M': activations_M,
        'activations_MD': activations_MD
    }


def collect_activations_hf(model, itda, tokenizer, data_stream, seq_len, batch_size, max_steps, layer):
    """
    Collect ITDA activations using HuggingFace model.
    """
    all_activations = []
    progress = tqdm(range(max_steps), desc="Collecting activations", unit="batch")
    
    for step in progress:
        # Prepare batch
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(data_stream))
            except StopIteration:
                break
                
        if not batch:
            print("Data stream exhausted.")
            break
            
        # Tokenize batch
        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        tokens = {k: v[:, :seq_len].to(model.device) for k, v in tokens.items()}
        
        # Get model activations
        with torch.no_grad():
            # Use output_hidden_states to get intermediate activations
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get activations from the specified layer
            # Note: hidden_states[0] is embeddings, so layer+1 gives us the layer we want
            if layer + 1 < len(hidden_states):
                model_activations = hidden_states[layer + 1]
            else:
                print(f"Warning: Layer {layer} not available, using last layer")
                model_activations = hidden_states[-1]
            
            # Encode with ITDA
            itda_activations = itda.encode(model_activations)
            all_activations.append(itda_activations.cpu())
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Concatenate all activations
    if all_activations:
        activations = torch.cat(all_activations, dim=0)
        print(f"Collected activations shape: {activations.shape}")
        return activations
    else:
        raise ValueError("No activations collected!")


if __name__ == "__main__":
    test_lora_model_diffing() 