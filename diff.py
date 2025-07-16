"""
Model-diffing implementation using ITDA (Iterative Top-Down Analysis).

This implements the model-diffing approach described in the paper, adapted for ITDA:
1. Collect ITDA activations over evaluation dataset E for both models M and M_D
2. Compute the difference in average activation before and after fine-tuning
3. Order the latents by how much their activations increased after fine-tuning
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from train import ITDA
import numpy as np


class ModelDiffing:
    """
    Implements model-diffing approach using ITDA instead of SAE.
    
    Assumes:
    - ITDA for the initial model M is already trained
    - We have access to both M (initial model) and M_D (fine-tuned model)
    - Evaluation dataset E is available
    """
    
    def __init__(
        self,
        itda_path: str,
        model_name: str,
        layer: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seq_len: int = 128,
        batch_size: int = 16
    ):
        """
        Initialize the model-diffing system.
        
        Args:
            itda_path: Path to the trained ITDA model for the initial model M
            model_name: Name of the model (should be same architecture for M and M_D)
            layer: Layer to analyze
            device: Device to run computations on
            seq_len: Sequence length for tokenization
            batch_size: Batch size for processing
        """
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.layer = layer
        
        # Load the trained ITDA for the initial model M
        print(f"Loading ITDA from {itda_path}")
        self.itda = ITDA.from_pretrained(itda_path, device=device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"ITDA loaded with {self.itda.dict_size} atoms")
        
    def load_model(self, model_name: str, is_fine_tuned: bool = False) -> HookedTransformer:
        """
        Load a model (either initial M or fine-tuned M_D).
        
        Args:
            model_name: Name/path of the model to load
            is_fine_tuned: Whether this is the fine-tuned model M_D
            
        Returns:
            Loaded model
        """
        print(f"Loading {'fine-tuned' if is_fine_tuned else 'initial'} model: {model_name}")
        
        if is_fine_tuned and os.path.exists(model_name):
            # Check if this is a LoRA adapter (has adapter_config.json)
            adapter_config_path = os.path.join(model_name, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"Detected LoRA adapter at {model_name}")
                # For LoRA adapters, we need to load the base model and merge the adapter
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-7B-Instruct",  # Use the base model name
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
                # Load and merge LoRA adapter
                model_with_adapter = PeftModel.from_pretrained(
                    base_model,
                    model_name
                )
                
                # Convert to HookedTransformer format
                # Note: This is a simplified approach. In practice, you might need
                # to handle the conversion more carefully
                print("Warning: LoRA adapter loaded, but conversion to HookedTransformer may need adjustment")
                return model_with_adapter
            else:
                # Regular model path
                model_path = model_name
        else:
            # Regular model path
            model_path = model_name
            
        model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        return model
    
    def collect_activations(
        self, 
        model: HookedTransformer, 
        dataset_name: str, 
        max_steps: int = 100
    ) -> torch.Tensor:
        """
        Collect ITDA activations for a given model over the evaluation dataset.
        
        Args:
            model: The model to collect activations from
            dataset_name: Name of the evaluation dataset
            max_steps: Maximum number of batches to process
            
        Returns:
            Tensor of shape (num_examples, seq_len, num_latents) containing activations
        """
        print(f"Collecting activations for model over {max_steps} batches...")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        data_stream = (item["text"] for item in dataset)
        
        all_activations = []
        progress = tqdm(range(max_steps), desc="Collecting activations", unit="batch")
        
        for step in progress:
            # Prepare batch
            batch = []
            for _ in range(self.batch_size):
                try:
                    batch.append(next(data_stream))
                except StopIteration:
                    break
                    
            if not batch:
                print("Data stream exhausted.")
                break
                
            # Tokenize batch
            tokens = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )
            tokens = {k: v[:, :self.seq_len].to(self.device) for k, v in tokens.items()}
            
            # Get model activations
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens["input_ids"],
                    stop_at_layer=self.layer + 1,
                    names_filter=[f"blocks.{self.layer}.hook_resid_post"],
                )
                model_activations = cache[f"blocks.{self.layer}.hook_resid_post"]
                
                # Encode with ITDA
                itda_activations = self.itda.encode(model_activations)
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
    
    def compute_activation_differences(
        self, 
        activations_M: torch.Tensor, 
        activations_MD: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the difference in average activation between M and M_D.
        
        Args:
            activations_M: Activations from initial model M
            activations_MD: Activations from fine-tuned model M_D
            
        Returns:
            Tensor of shape (num_latents,) containing activation differences
        """
        print("Computing activation differences...")
        
        # Compute average activations across all examples and sequence positions
        # Shape: (num_latents,)
        avg_activations_M = activations_M.mean(dim=(0, 1))  # Average over batch and sequence
        avg_activations_MD = activations_MD.mean(dim=(0, 1))
        
        # Compute differences: M_D - M (how much activations increased after fine-tuning)
        activation_differences = avg_activations_MD - avg_activations_M
        
        print(f"Activation differences shape: {activation_differences.shape}")
        return activation_differences
    
    def order_latents_by_increase(
        self, 
        activation_differences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Order the latents by how much their activations increased after fine-tuning.
        
        Args:
            activation_differences: Differences in average activations (M_D - M)
            
        Returns:
            Tuple of (sorted_indices, sorted_differences) where indices are ordered
            from largest increase to smallest
        """
        print("Ordering latents by activation increase...")
        
        # Sort in descending order (largest increases first)
        sorted_differences, sorted_indices = torch.sort(
            activation_differences, 
            descending=True
        )
        
        print(f"Top 10 latents with largest activation increases:")
        for i in range(min(10, len(sorted_indices))):
            latent_idx = sorted_indices[i].item()
            diff = sorted_differences[i].item()
            print(f"  Latent {latent_idx}: {diff:.6f}")
            
        return sorted_indices, sorted_differences
    
    def run_model_diffing(
        self,
        initial_model_name: str,
        fine_tuned_model_name: str,
        evaluation_dataset: str,
        max_steps: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Run the complete model-diffing pipeline.
        
        Args:
            initial_model_name: Name/path of the initial model M
            fine_tuned_model_name: Name/path of the fine-tuned model M_D
            evaluation_dataset: Name of the evaluation dataset E
            max_steps: Maximum number of batches to process
            
        Returns:
            Dictionary containing:
            - 'activation_differences': Differences in average activations
            - 'sorted_indices': Latents ordered by activation increase
            - 'sorted_differences': Sorted activation differences
            - 'activations_M': Raw activations from initial model
            - 'activations_MD': Raw activations from fine-tuned model
        """
        print("Starting model-diffing analysis...")
        
        # Step 1: Load models
        model_M = self.load_model(initial_model_name, is_fine_tuned=False)
        model_MD = self.load_model(fine_tuned_model_name, is_fine_tuned=True)
        
        # Step 2: Collect activations for both models
        print("\n" + "="*50)
        print("STEP 1: Collecting ITDA activations")
        print("="*50)
        
        activations_M = self.collect_activations(
            model_M, 
            evaluation_dataset, 
            max_steps
        )
        
        activations_MD = self.collect_activations(
            model_MD, 
            evaluation_dataset, 
            max_steps
        )
        
        # Step 3: Compute activation differences
        print("\n" + "="*50)
        print("STEP 2: Computing activation differences")
        print("="*50)
        
        activation_differences = self.compute_activation_differences(
            activations_M, 
            activations_MD
        )
        
        # Step 4: Order latents by increase
        print("\n" + "="*50)
        print("STEP 3: Ordering latents by activation increase")
        print("="*50)
        
        sorted_indices, sorted_differences = self.order_latents_by_increase(
            activation_differences
        )
        
        print("\n" + "="*50)
        print("Model-diffing analysis complete!")
        print("="*50)
        
        return {
            'activation_differences': activation_differences,
            'sorted_indices': sorted_indices,
            'sorted_differences': sorted_differences,
            'activations_M': activations_M,
            'activations_MD': activations_MD
        }


def main():
    """
    Example usage of the ModelDiffing class.
    """
    # Configuration
    itda_path = "artifacts/runs/your_run_id"  # Path to your trained ITDA
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Base model name
    layer = 21  # Layer to analyze
    evaluation_dataset = "NeelNanda/pile-10k"  # Evaluation dataset
    
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
    # Note: You'll need to provide the actual paths to your initial and fine-tuned models
    results = diffing.run_model_diffing(
        initial_model_name=model_name,  # Path to initial model M
        fine_tuned_model_name="path/to/fine_tuned_model",  # Path to fine-tuned model M_D
        evaluation_dataset=evaluation_dataset,
        max_steps=50  # Adjust based on your needs
    )
    
    # Access results
    print(f"\nTop 20 latents with largest activation increases:")
    for i in range(20):
        latent_idx = results['sorted_indices'][i].item()
        diff = results['sorted_differences'][i].item()
        print(f"{i+1:2d}. Latent {latent_idx:4d}: {diff:+.6f}")


if __name__ == "__main__":
    main()
