"""
Model-diffing with interpretation: Find top activating snippets for the most changed latents.

This script:
1. Runs model-diffing to identify the latents with the largest activation increases
2. For each top changed latent, finds the snippets that activate it most strongly
3. Shows the top activating snippets for each changed latent

Dataset loading options:
- HuggingFace dataset names (e.g., "NeelNanda/pile-10k")
- Local txt files with one text sample per line (e.g., "my_data.txt")
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
from IPython.display import HTML, display
import json


def highlight_string(tokens, idx, tokenizer, crop=-1):
    """Highlight a specific token in a sequence."""
    str_ = ""

    if crop != -1:
        start_idx = max(0, idx - crop)
        end_idx = min(len(tokens), idx + crop + 1)
        tokens = tokens[start_idx:end_idx]
        idx -= start_idx

    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token]).replace("\n", "")
        if i == idx:
            str_ += f'<span style="background-color: darkblue; font-weight: bold;">{token_str}</span>'
        else:
            str_ += f"{token_str}"

    return HTML(str_)


class ModelDiffingInterpreter:
    """
    Combines model-diffing with interpretation to find top activating snippets for changed latents.
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
        Initialize the model-diffing interpreter.
        
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
            # For local merged models, we need to use HuggingFace transformers directly
            # and then convert to HookedTransformer format
            from transformers import AutoModelForCausalLM
            
            print(f"Loading local merged model from {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Convert to HookedTransformer format
            # This is a workaround since HookedTransformer doesn't support local paths
            print("Converting to HookedTransformer format...")
            # We'll use the model directly as a HuggingFace model
            # Note: This means we need to adapt our activation collection method
            return model
        else:
            # Regular model path - use HookedTransformer
            model_path = model_name
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            return model
    
    def collect_activations_with_tokens(
        self, 
        model, 
        dataset_source: str, 
        max_steps: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect ITDA activations and tokens for a given model over the evaluation dataset.
        
        Args:
            model: The model to collect activations from (HookedTransformer or HuggingFace model)
            dataset_source: Name of the evaluation dataset OR path to a txt file
            max_steps: Maximum number of batches to process
            
        Returns:
            Tuple of (activations, tokens) where:
            - activations: Tensor of shape (num_examples, seq_len, num_latents)
            - tokens: Tensor of shape (num_examples, seq_len)
        """
        print(f"Collecting activations and tokens for model over {max_steps} batches...")
        
        # Check if dataset_source is a txt file path or a dataset name
        if dataset_source.endswith('.txt') and os.path.exists(dataset_source):
            print(f"Loading dataset from txt file: {dataset_source}")
            # Load from txt file
            def txt_data_stream():
                with open(dataset_source, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            yield {"text": line}
            data_stream = (item["text"] for item in txt_data_stream())
        else:
            print(f"Loading dataset from HuggingFace: {dataset_source}")
            # Load dataset from HuggingFace
            dataset = load_dataset(dataset_source, split="train", streaming=True)
            data_stream = (item["text"] for item in dataset)
        
        all_activations = []
        all_tokens = []
        progress = tqdm(range(max_steps), desc="Collecting activations and tokens", unit="batch")
        
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
                if hasattr(model, 'run_with_cache'):
                    # HookedTransformer model
                    _, cache = model.run_with_cache(
                        tokens["input_ids"],
                        stop_at_layer=self.layer + 1,
                        names_filter=[f"blocks.{self.layer}.hook_resid_post"],
                    )
                    model_activations = cache[f"blocks.{self.layer}.hook_resid_post"]
                else:
                    # HuggingFace model - we need to manually extract activations
                    # This is a simplified approach - in practice you might need more sophisticated handling
                    outputs = model(
                        tokens["input_ids"],
                        output_hidden_states=True,
                        return_dict=True
                    )
                    # Get hidden states for the specified layer
                    hidden_states = outputs.hidden_states
                    if hidden_states is not None and len(hidden_states) > self.layer:
                        model_activations = hidden_states[self.layer]
                    else:
                        raise ValueError(f"Layer {self.layer} not available in model outputs")
                
                # Encode with ITDA
                itda_activations = self.itda.encode(model_activations)
                all_activations.append(itda_activations.cpu())
                all_tokens.append(tokens["input_ids"].cpu())
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Concatenate all activations and tokens
        if all_activations:
            activations = torch.cat(all_activations, dim=0)
            tokens = torch.cat(all_tokens, dim=0)
            print(f"Collected activations shape: {activations.shape}")
            print(f"Collected tokens shape: {tokens.shape}")
            return activations, tokens
        else:
            raise ValueError("No activations collected!")
    
    def find_top_activating_snippets(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        latent_idx: int,
        top_k: int = 10
    ) -> List[Tuple[int, int, float, str]]:
        """
        Find the top activating snippets for a specific latent.
        
        Args:
            activations: Tensor of shape (num_examples, seq_len, num_latents)
            tokens: Tensor of shape (num_examples, seq_len)
            latent_idx: Index of the latent to analyze
            top_k: Number of top activating snippets to return
            
        Returns:
            List of tuples (seq_idx, tok_idx, activation_value, decoded_text)
        """
        print(f"Finding top activating snippets for latent {latent_idx}...")
        
        # Get activations for this specific latent
        latent_activations = activations[:, :, latent_idx]
        
        # Find top k activations
        values, indices = torch.topk(latent_activations.flatten(), top_k)
        
        if values.allclose(torch.zeros_like(values)):
            print("No activations found.")
            return []
        
        # Convert flat indices to (seq_idx, tok_idx) pairs
        rows, cols = (
            torch.div(
                indices,
                latent_activations.shape[1],
                rounding_mode="floor",
            ),
            indices % latent_activations.shape[1],
        )
        top_indices = torch.stack((rows, cols), dim=1)
        
        # Extract snippets with decoded text
        snippets = []
        for (seq_idx, tok_idx), val in zip(top_indices, values):
            seq_idx_int = seq_idx.item()
            tok_idx_int = tok_idx.item()
            activation_value = val.item()
            
            # Decode the token sequence around the activating token
            token_sequence = tokens[seq_idx_int]
            # Get context around the activating token (crop=10 for context)
            start_idx = max(0, tok_idx_int - 10)
            end_idx = min(len(token_sequence), tok_idx_int + 11)
            context_tokens = token_sequence[start_idx:end_idx]
            
            # Decode the context
            decoded_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
            
            snippets.append((seq_idx_int, tok_idx_int, activation_value, decoded_text))
        
        return snippets
    
    def run_diffing_with_interpretation(
        self,
        initial_model_name: str,
        fine_tuned_model_name: str,
        evaluation_dataset: str,
        max_steps: int = 100,
        top_latents: int = 10,
        snippets_per_latent: int = 5
    ) -> Dict:
        """
        Run model-diffing and find top activating snippets for the most changed latents.
        
        Args:
            initial_model_name: Name/path of the initial model M
            fine_tuned_model_name: Name/path of the fine-tuned model M_D
            evaluation_dataset: Name of the evaluation dataset E OR path to a txt file with one text per line
            max_steps: Maximum number of batches to process
            top_latents: Number of top changed latents to analyze
            snippets_per_latent: Number of top activating snippets to show per latent
            
        Returns:
            Dictionary containing the analysis results
        """
        print("Starting model-diffing with interpretation analysis...")
        
        # Step 1: Load models
        model_M = self.load_model(initial_model_name, is_fine_tuned=False)
        model_MD = self.load_model(fine_tuned_model_name, is_fine_tuned=True)
        
        # Step 2: Collect activations and tokens for both models
        print("\n" + "="*50)
        print("STEP 1: Collecting ITDA activations and tokens")
        print("="*50)
        
        activations_M, tokens_M = self.collect_activations_with_tokens(
            model_M, 
            evaluation_dataset, 
            max_steps
        )
        
        activations_MD, tokens_MD = self.collect_activations_with_tokens(
            model_MD, 
            evaluation_dataset, 
            max_steps
        )
        
        # Step 3: Compute activation differences
        print("\n" + "="*50)
        print("STEP 2: Computing activation differences")
        print("="*50)
        
        # Compute average activations across all examples and sequence positions
        avg_activations_M = activations_M.mean(dim=(0, 1))
        avg_activations_MD = activations_MD.mean(dim=(0, 1))
        
        # Compute differences: M_D - M (how much activations increased after fine-tuning)
        activation_differences = avg_activations_MD - avg_activations_M
        
        # Order latents by activation increase
        sorted_differences, sorted_indices = torch.sort(
            activation_differences, 
            descending=True
        )
        
        print(f"Top {top_latents} latents with largest activation increases:")
        for i in range(top_latents):
            latent_idx = sorted_indices[i].item()
            diff = sorted_differences[i].item()
            print(f"  {i+1:2d}. Latent {latent_idx:4d}: {diff:+.6f}")
        

        
        # Step 4: Find top activating snippets for each changed latent
        print("\n" + "="*50)
        print("STEP 3: Finding top activating snippets for changed latents")
        print("="*50)
        
        results = {
            'activation_differences': activation_differences,
            'sorted_indices': sorted_indices,
            'sorted_differences': sorted_differences,
            'top_latents_analysis': []
        }
        
        for i in range(top_latents):
            latent_idx = sorted_indices[i].item()
            diff = sorted_differences[i].item()
            
            print(f"\n--- Latent {latent_idx} (activation increase: {diff:+.6f}) ---")
            
            # Find top activating snippets for this latent in the fine-tuned model
            snippets_MD = self.find_top_activating_snippets(
                activations_MD, 
                tokens_MD, 
                latent_idx, 
                snippets_per_latent
            )
            
            # Also find snippets for the initial model for comparison
            snippets_M = self.find_top_activating_snippets(
                activations_M, 
                tokens_M, 
                latent_idx, 
                snippets_per_latent
            )
            
            print(f"  Found {len(snippets_MD)} snippets for fine-tuned model, {len(snippets_M)} for initial model")
            
            # Convert snippets to a more readable format for JSON
            def format_snippets(snippets):
                formatted = []
                for seq_idx, tok_idx, activation_value, decoded_text in snippets:
                    formatted.append({
                        'sequence_index': seq_idx,
                        'token_index': tok_idx,
                        'activation_value': activation_value,
                        'decoded_text': decoded_text
                    })
                return formatted
            
            results['top_latents_analysis'].append({
                'latent_idx': latent_idx,
                'activation_increase': diff,
                'snippets_fine_tuned': format_snippets(snippets_MD),
                'snippets_initial': format_snippets(snippets_M)
            })
        
        print("\n" + "="*50)
        print("Model-diffing with interpretation analysis complete!")
        print("="*50)
        
        return results


def main():
    """
    Example usage of the ModelDiffingInterpreter class.
    """
    # Configuration
    itda_path = "artifacts/runs/bH0ZrUOL"  # Path to your trained ITDA
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Base model name
    layer = 21  # Layer to analyze
    
    # Evaluation dataset - can be either a HuggingFace dataset name or a txt file path
    # evaluation_dataset = "NeelNanda/pile-10k"  # HuggingFace dataset
    evaluation_dataset = "eval/first_plot_questions.txt"  # Alternative: txt file with one text per line
    
    # Initialize model-diffing interpreter
    interpreter = ModelDiffingInterpreter(
        itda_path=itda_path,
        model_name=model_name,
        layer=layer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_len=128,
        batch_size=1
    )
    
    # Run the analysis
    results = interpreter.run_diffing_with_interpretation(
        initial_model_name=model_name,  # Path to initial model M
        fine_tuned_model_name="models/emergent-misalignment/qwen_medical_misaligned_merged",  # Path to fine-tuned model M_D
        evaluation_dataset=evaluation_dataset,
        max_steps=50,  # Adjust based on your needs
        top_latents=20,  # Number of top changed latents to analyze
        snippets_per_latent=10  # Number of snippets to show per latent
    )
    
    # Print summary
    print(f"\nSummary:")
    print(f"Analyzed {len(results['top_latents_analysis'])} top changed latents")
    print(f"Found activation differences ranging from {results['sorted_differences'][0]:.6f} to {results['sorted_differences'][-1]:.6f}")

    # Save results to file
    with open("diff_interp_results2.json", "w") as f:
        # Convert tensors to lists for JSON serialization
        def tensor_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [tensor_to_list(x) for x in obj]
            return obj
        json.dump(tensor_to_list(results), f, indent=2)
    print("Saved results to diff_interp_results.json")


if __name__ == "__main__":
    main() 