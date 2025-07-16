"""
Example script demonstrating how to use the model-diffing implementation.

This script shows how to run the model-diffing analysis with ITDA.
"""

import torch
from diff import ModelDiffing


def run_example_diffing():
    """
    Example of running model-diffing analysis.
    
    This example assumes:
    - You have a trained ITDA model at artifacts/runs/bH0ZrUOL/
    - You have access to both the initial and fine-tuned models
    - You want to analyze layer 21 of a Qwen model
    """
    
    # Configuration
    itda_path = "artifacts/runs/bH0ZrUOL"  # Path to your trained ITDA
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Base model name
    layer = 21  # Layer to analyze
    evaluation_dataset = "NeelNanda/pile-10k"  # Evaluation dataset
    
    print("="*60)
    print("MODEL-DIFFING EXAMPLE")
    print("="*60)
    print(f"ITDA path: {itda_path}")
    print(f"Model: {model_name}")
    print(f"Layer: {layer}")
    print(f"Evaluation dataset: {evaluation_dataset}")
    print("="*60)
    
    # Initialize model-diffing
    diffing = ModelDiffing(
        itda_path=itda_path,
        model_name=model_name,
        layer=layer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_len=128,
        batch_size=16
    )
    
    # Run the analysis
    # Note: You'll need to provide the actual paths to your models
    # For this example, we'll use the same model name for both
    # In practice, you'd have different paths for initial and fine-tuned models
    
    try:
        results = diffing.run_model_diffing(
            initial_model_name=model_name,  # Path to initial model M
            fine_tuned_model_name=model_name,  # Path to fine-tuned model M_D (same for demo)
            evaluation_dataset=evaluation_dataset,
            max_steps=10  # Small number for demo
        )
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        print(f"Number of latents analyzed: {len(results['activation_differences'])}")
        print(f"Activation differences shape: {results['activation_differences'].shape}")
        
        print(f"\nTop 20 latents with largest activation increases:")
        print("-" * 50)
        for i in range(min(20, len(results['sorted_indices']))):
            latent_idx = results['sorted_indices'][i].item()
            diff = results['sorted_differences'][i].item()
            print(f"{i+1:2d}. Latent {latent_idx:4d}: {diff:+.6f}")
            
        # Show some statistics
        differences = results['activation_differences']
        print(f"\nStatistics:")
        print(f"  Mean difference: {differences.mean().item():.6f}")
        print(f"  Std difference: {differences.std().item():.6f}")
        print(f"  Max increase: {differences.max().item():.6f}")
        print(f"  Max decrease: {differences.min().item():.6f}")
        print(f"  Positive changes: {(differences > 0).sum().item()}")
        print(f"  Negative changes: {(differences < 0).sum().item()}")
        
    except Exception as e:
        print(f"Error running model-diffing: {e}")
        print("Make sure you have:")
        print("1. A trained ITDA model at the specified path")
        print("2. Access to the model specified")
        print("3. Internet connection for downloading the dataset")


def run_custom_diffing():
    """
    Example with custom parameters for different use cases.
    """
    
    # Example 1: Different layer analysis
    print("\n" + "="*60)
    print("CUSTOM EXAMPLE: Different layer")
    print("="*60)
    
    diffing_layer_10 = ModelDiffing(
        itda_path="artifacts/runs/bH0ZrUOL",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layer=10,  # Different layer
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_len=64,  # Shorter sequences
        batch_size=8   # Smaller batches
    )
    
    # Example 2: Different dataset
    print("\n" + "="*60)
    print("CUSTOM EXAMPLE: Different dataset")
    print("="*60)
    
    diffing_custom = ModelDiffing(
        itda_path="artifacts/runs/bH0ZrUOL",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        layer=21,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_len=256,  # Longer sequences
        batch_size=4   # Smaller batches for longer sequences
    )
    
    print("Custom configurations created successfully!")


if __name__ == "__main__":
    # Run the main example
    run_example_diffing()
    
    # Run custom examples
    run_custom_diffing() 