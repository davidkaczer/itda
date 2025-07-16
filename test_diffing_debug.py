"""
Debug test script to check if LoRA adapter is being applied correctly.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from train import ITDA


def test_lora_application():
    """
    Test if the LoRA adapter is being applied correctly.
    """
    
    print("="*70)
    print("DEBUG: LoRA ADAPTER APPLICATION TEST")
    print("="*70)
    
    # Load models
    initial_model_name = "Qwen/Qwen2.5-7B-Instruct"
    fine_tuned_model_path = "models/emergent-misalignment/qwen_medical_misaligned"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(initial_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load initial model
    model_initial = AutoModelForCausalLM.from_pretrained(
        initial_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load fine-tuned model (LoRA adapter)
    model_fine_tuned = PeftModel.from_pretrained(
        model_initial,
        fine_tuned_model_path
    )
    
    # Test with a simple prompt
    test_prompt = "The patient has a fever of 102 degrees."
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model_initial.device)
    
    # Get outputs from both models
    with torch.no_grad():
        # Initial model
        outputs_initial = model_initial(**inputs, output_hidden_states=True)
        logits_initial = outputs_initial.logits
        hidden_states_initial = outputs_initial.hidden_states
        
        # Fine-tuned model
        outputs_fine_tuned = model_fine_tuned(**inputs, output_hidden_states=True)
        logits_fine_tuned = outputs_fine_tuned.logits
        hidden_states_fine_tuned = outputs_fine_tuned.hidden_states
    
    # Check if outputs are different
    logits_diff = torch.abs(logits_fine_tuned - logits_initial).max().item()
    print(f"Max logits difference: {logits_diff}")
    
    # Check hidden states differences for different layers
    print(f"\nHidden states differences by layer:")
    for layer in [0, 5, 10, 15, 20, 25, 30]:
        if layer < len(hidden_states_initial) and layer < len(hidden_states_fine_tuned):
            layer_diff = torch.abs(hidden_states_fine_tuned[layer] - hidden_states_initial[layer]).max().item()
            print(f"  Layer {layer}: {layer_diff:.6f}")
    
    # Check if LoRA weights are actually loaded
    print(f"\nLoRA adapter info:")
    print(f"  Base model: {model_fine_tuned.base_model.model.config.model_type}")
    print(f"  Adapter name: {model_fine_tuned.active_adapter}")
    print(f"  Active adapters: {model_fine_tuned.active_adapters}")
    
    # Check some LoRA weights
    for name, module in model_fine_tuned.named_modules():
        if 'lora' in name.lower():
            print(f"  Found LoRA module: {name}")
            if hasattr(module, 'lora_A'):
                print(f"    lora_A shape: {module.lora_A.weight.shape}")
            if hasattr(module, 'lora_B'):
                print(f"    lora_B shape: {module.lora_B.weight.shape}")


def test_different_layers():
    """
    Test model-diffing with different layers.
    """
    
    print("\n" + "="*70)
    print("DEBUG: TESTING DIFFERENT LAYERS")
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
    
    # Test different layers
    test_layers = [0, 5, 10, 15, 20, 25, 30]
    
    for layer in test_layers:
        print(f"\nTesting layer {layer}:")
        
        # Simple test with one batch
        test_text = "The patient has a fever."
        inputs = tokenizer([test_text], padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(model_initial.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get activations from initial model
            outputs_initial = model_initial(**inputs, output_hidden_states=True)
            hidden_states_initial = outputs_initial.hidden_states
            
            # Get activations from fine-tuned model
            outputs_fine_tuned = model_fine_tuned(**inputs, output_hidden_states=True)
            hidden_states_fine_tuned = outputs_fine_tuned.hidden_states
            
            # Check if layer exists
            if layer < len(hidden_states_initial) and layer < len(hidden_states_fine_tuned):
                # Get activations from the layer
                activations_initial = hidden_states_initial[layer]
                activations_fine_tuned = hidden_states_fine_tuned[layer]
                
                # Encode with ITDA
                itda_activations_initial = itda.encode(activations_initial)
                itda_activations_fine_tuned = itda.encode(activations_fine_tuned)
                
                # Compute differences
                avg_initial = itda_activations_initial.mean(dim=(0, 1))
                avg_fine_tuned = itda_activations_fine_tuned.mean(dim=(0, 1))
                differences = avg_fine_tuned - avg_initial
                
                print(f"  Max difference: {differences.abs().max().item():.6f}")
                print(f"  Mean difference: {differences.mean().item():.6f}")
                print(f"  Non-zero differences: {(differences != 0).sum().item()}")
            else:
                print(f"  Layer {layer} not available")


if __name__ == "__main__":
    test_lora_application()
    test_different_layers() 