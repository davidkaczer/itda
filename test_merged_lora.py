"""
Test with merged LoRA weights to see if that resolves the zero differences issue.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from train import ITDA


def test_merged_lora():
    """
    Test with merged LoRA weights.
    """
    
    print("="*70)
    print("MERGED LoRA TESTING")
    print("="*70)
    
    # Load ITDA
    itda_path = "artifacts/runs/bH0ZrUOL"
    itda = ITDA.from_pretrained(itda_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    initial_model_name = "Qwen/Qwen2.5-7B-Instruct"
    fine_tuned_model_path = "models/emergent-misalignment/qwen_medical_misaligned"
    
    tokenizer = AutoTokenizer.from_pretrained(initial_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model_initial = AutoModelForCausalLM.from_pretrained(
        initial_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA model and merge weights
    model_fine_tuned = PeftModel.from_pretrained(
        model_initial,
        fine_tuned_model_path
    )
    
    # Merge LoRA weights into the base model
    print("Merging LoRA weights...")
    model_fine_tuned = model_fine_tuned.merge_and_unload()
    
    # Test with a simple prompt
    test_prompt = "The patient has a fever of 102 degrees."
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model_initial.device)
    
    layer = 21
    
    with torch.no_grad():
        # Get activations from initial model
        outputs_initial = model_initial(**inputs, output_hidden_states=True)
        hidden_states_initial = outputs_initial.hidden_states
        
        # Get activations from fine-tuned model (merged)
        outputs_fine_tuned = model_fine_tuned(**inputs, output_hidden_states=True)
        hidden_states_fine_tuned = outputs_fine_tuned.hidden_states
        
        # Get activations from the layer
        if layer < len(hidden_states_initial) and layer < len(hidden_states_fine_tuned):
            activations_initial = hidden_states_initial[layer]
            activations_fine_tuned = hidden_states_fine_tuned[layer]
            
            # Check raw hidden states differences
            raw_diff = torch.abs(activations_fine_tuned - activations_initial).max().item()
            print(f"Raw hidden states max difference: {raw_diff:.6f}")
            
            # Encode with ITDA
            itda_activations_initial = itda.encode(activations_initial)
            itda_activations_fine_tuned = itda.encode(activations_fine_tuned)
            
            # Compute differences
            avg_initial = itda_activations_initial.mean(dim=(0, 1))
            avg_fine_tuned = itda_activations_fine_tuned.mean(dim=(0, 1))
            differences = avg_fine_tuned - avg_initial
            
            print(f"ITDA activation differences:")
            print(f"  Max difference: {differences.abs().max().item():.6f}")
            print(f"  Mean difference: {differences.mean().item():.6f}")
            print(f"  Non-zero differences: {(differences != 0).sum().item()}")
            
            # Show top 10 latents with largest differences
            top_indices = torch.topk(differences.abs(), 10).indices
            print(f"\nTop 10 latents with largest differences:")
            for i, idx in enumerate(top_indices):
                diff_val = differences[idx].item()
                print(f"  {i+1:2d}. Latent {idx.item()}: {diff_val:+.6f}")
        else:
            print(f"Layer {layer} not available")


def test_precision_issue():
    """
    Test if there's a precision issue causing the zero differences.
    """
    
    print("\n" + "="*70)
    print("PRECISION TESTING")
    print("="*70)
    
    # Load ITDA
    itda_path = "artifacts/runs/bH0ZrUOL"
    itda = ITDA.from_pretrained(itda_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    initial_model_name = "Qwen/Qwen2.5-7B-Instruct"
    fine_tuned_model_path = "models/emergent-misalignment/qwen_medical_misaligned"
    
    tokenizer = AutoTokenizer.from_pretrained(initial_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_initial = AutoModelForCausalLM.from_pretrained(
        initial_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model_fine_tuned = PeftModel.from_pretrained(
        model_initial,
        fine_tuned_model_path
    )
    
    # Test with a simple prompt
    test_prompt = "The patient has a fever."
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model_initial.device)
    
    layer = 21
    
    with torch.no_grad():
        # Get activations from initial model
        outputs_initial = model_initial(**inputs, output_hidden_states=True)
        hidden_states_initial = outputs_initial.hidden_states
        
        # Get activations from fine-tuned model
        outputs_fine_tuned = model_fine_tuned(**inputs, output_hidden_states=True)
        hidden_states_fine_tuned = outputs_fine_tuned.hidden_states
        
        # Get activations from the layer
        if layer < len(hidden_states_initial) and layer < len(hidden_states_fine_tuned):
            activations_initial = hidden_states_initial[layer]
            activations_fine_tuned = hidden_states_fine_tuned[layer]
            
            # Check raw hidden states differences
            raw_diff = torch.abs(activations_fine_tuned - activations_initial).max().item()
            print(f"Raw hidden states max difference: {raw_diff:.6f}")
            
            # Check if raw activations are identical
            if torch.allclose(activations_initial, activations_fine_tuned, atol=1e-6):
                print("Raw activations are identical (within tolerance)")
            else:
                print("Raw activations are different")
            
            # Encode with ITDA at higher precision
            itda_activations_initial = itda.encode(activations_initial.float())
            itda_activations_fine_tuned = itda.encode(activations_fine_tuned.float())
            
            # Compute differences
            avg_initial = itda_activations_initial.mean(dim=(0, 1))
            avg_fine_tuned = itda_activations_fine_tuned.mean(dim=(0, 1))
            differences = avg_fine_tuned - avg_initial
            
            print(f"ITDA activation differences (float32):")
            print(f"  Max difference: {differences.abs().max().item():.6f}")
            print(f"  Mean difference: {differences.mean().item():.6f}")
            print(f"  Non-zero differences: {(differences != 0).sum().item()}")
            
            # Check if ITDA activations are identical
            if torch.allclose(itda_activations_initial, itda_activations_fine_tuned, atol=1e-6):
                print("ITDA activations are identical (within tolerance)")
            else:
                print("ITDA activations are different")
        else:
            print(f"Layer {layer} not available")


if __name__ == "__main__":
    test_merged_lora()
    test_precision_issue() 