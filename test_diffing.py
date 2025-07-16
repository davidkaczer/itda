"""
Test script for model-diffing with specific models:
- Initial model: Qwen/Qwen2.5-7B-Instruct
- Fine-tuned model: models/emergent-misalignment/qwen_medical_misaligned (LoRA adapter)
"""

import torch
import os
from diff import ModelDiffing


def test_with_specific_models():
    """
    Test the model-diffing implementation with the specified models.
    """
    
    # Configuration
    itda_path = "artifacts/runs/bH0ZrUOL"  # Using the existing ITDA run
    initial_model_name = "Qwen/Qwen2.5-7B-Instruct"
    fine_tuned_model_name = "models/emergent-misalignment/qwen_medical_misaligned"
    layer = 21  # Layer to analyze
    evaluation_dataset = "NeelNanda/pile-10k"  # Evaluation dataset
    
    print("="*70)
    print("MODEL-DIFFING TEST")
    print("="*70)
    print(f"ITDA path: {itda_path}")
    print(f"Initial model: {initial_model_name}")
    print(f"Fine-tuned model: {fine_tuned_model_name}")
    print(f"Layer: {layer}")
    print(f"Evaluation dataset: {evaluation_dataset}")
    print("="*70)
    
    # Check if ITDA path exists
    if not os.path.exists(itda_path):
        print(f"ERROR: ITDA path {itda_path} does not exist!")
        print("Please make sure you have a trained ITDA model at this path.")
        return
    
    # Check if fine-tuned model path exists
    if not os.path.exists(fine_tuned_model_name):
        print(f"ERROR: Fine-tuned model path {fine_tuned_model_name} does not exist!")
        print("Please make sure the fine-tuned model is available at this path.")
        return
    
    # Initialize model-diffing
    try:
        diffing = ModelDiffing(
            itda_path=itda_path,
            model_name=initial_model_name,
            layer=layer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seq_len=128,
            batch_size=8  # Smaller batch size for testing
        )
        print("✓ ModelDiffing initialized successfully")
        
    except Exception as e:
        print(f"ERROR initializing ModelDiffing: {e}")
        return
    
    # Run the analysis
    try:
        print("\nStarting model-diffing analysis...")
        results = diffing.run_model_diffing(
            initial_model_name=initial_model_name,
            fine_tuned_model_name=fine_tuned_model_name,
            evaluation_dataset=evaluation_dataset,
            max_steps=20  # Small number for testing
        )
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        print(f"Number of latents analyzed: {len(results['activation_differences'])}")
        print(f"Activation differences shape: {results['activation_differences'].shape}")
        
        print(f"\nTop 20 latents with largest activation increases:")
        print("-" * 60)
        for i in range(min(20, len(results['sorted_indices']))):
            latent_idx = results['sorted_indices'][i].item()
            diff = results['sorted_differences'][i].item()
            print(f"{i+1:2d}. Latent {latent_idx:4d}: {diff:+.6f}")
            
        # Show statistics
        differences = results['activation_differences']
        print(f"\nStatistics:")
        print(f"  Mean difference: {differences.mean().item():.6f}")
        print(f"  Std difference: {differences.std().item():.6f}")
        print(f"  Max increase: {differences.max().item():.6f}")
        print(f"  Max decrease: {differences.min().item():.6f}")
        print(f"  Positive changes: {(differences > 0).sum().item()}")
        print(f"  Negative changes: {(differences < 0).sum().item()}")
        print(f"  Zero changes: {(differences == 0).sum().item()}")
        
        # Show distribution
        print(f"\nDistribution:")
        print(f"  > 0.01: {(differences > 0.01).sum().item()}")
        print(f"  > 0.001: {(differences > 0.001).sum().item()}")
        print(f"  < -0.01: {(differences < -0.01).sum().item()}")
        print(f"  < -0.001: {(differences < -0.001).sum().item()}")
        
        print("\n" + "="*70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"ERROR running model-diffing: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the ITDA model exists at the specified path")
        print("2. Check that both models can be loaded")
        print("3. Verify you have sufficient GPU memory")
        print("4. Ensure the evaluation dataset is accessible")


def test_model_loading():
    """
    Test if the models can be loaded individually.
    """
    
    print("\n" + "="*70)
    print("MODEL LOADING TEST")
    print("="*70)
    
    from transformer_lens import HookedTransformer
    
    # Test initial model
    try:
        print("Testing initial model loading...")
        model_initial = HookedTransformer.from_pretrained_no_processing(
            "Qwen/Qwen2.5-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Initial model loaded successfully")
        del model_initial
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Error loading initial model: {e}")
    
    # Test fine-tuned model (LoRA adapter)
    try:
        print("Testing fine-tuned model loading (LoRA adapter)...")
        # For LoRA adapters, we need to load the base model and then merge the adapter
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load and merge LoRA adapter
        model_fine_tuned = PeftModel.from_pretrained(
            base_model,
            "models/emergent-misalignment/qwen_medical_misaligned"
        )
        
        print("✓ Fine-tuned model (LoRA) loaded successfully")
        del model_fine_tuned
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Error loading fine-tuned model: {e}")


def test_itda_loading():
    """
    Test if the ITDA can be loaded.
    """
    
    print("\n" + "="*70)
    print("ITDA LOADING TEST")
    print("="*70)
    
    try:
        from train import ITDA
        itda = ITDA.from_pretrained("artifacts/runs/bH0ZrUOL")
        print("✓ ITDA loaded successfully")
        print(f"  - Number of atoms: {itda.dict_size}")
        print(f"  - Activation dimension: {itda.activation_dim}")
        print(f"  - k value: {itda.k}")
    except Exception as e:
        print(f"✗ Error loading ITDA: {e}")


if __name__ == "__main__":
    # Run individual tests first
    test_itda_loading()
    test_model_loading()
    
    # Run the main test
    test_with_specific_models() 