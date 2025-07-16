"""
Utility functions for model steering with HookedTransformer.
Contains common steering vector generation techniques and application methods.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformer_lens import HookedTransformer


def create_contrast_vector(
    model: HookedTransformer,
    positive_prompt: str,
    negative_prompt: str,
    layer: int,
    normalize: bool = True,
    position: str = "last"  # "last", "mean", or "all"
) -> torch.Tensor:
    """
    Create a steering vector by contrasting positive and negative prompts.
    
    Args:
        model: HookedTransformer model
        positive_prompt: Text representing desired behavior
        negative_prompt: Text representing undesired behavior  
        layer: Which layer to extract activations from
        normalize: Whether to normalize the resulting vector
        position: How to aggregate sequence positions ("last", "mean", "all")
    
    Returns:
        Steering vector tensor
    """
    # Get activations for both prompts
    tokens_pos = model.to_tokens(positive_prompt, prepend_bos=True)
    tokens_neg = model.to_tokens(negative_prompt, prepend_bos=True)
    
    _, cache_pos = model.run_with_cache(tokens_pos)
    _, cache_neg = model.run_with_cache(tokens_neg)
    
    # Extract activations at specified layer
    act_pos = cache_pos[f"blocks.{layer}.hook_resid_post"]  # [batch, seq, hidden]
    act_neg = cache_neg[f"blocks.{layer}.hook_resid_post"]
    
    # Aggregate across sequence dimension
    if position == "last":
        vector_pos = act_pos[0, -1, :]  # Last position
        vector_neg = act_neg[0, -1, :]
    elif position == "mean":
        vector_pos = act_pos[0].mean(dim=0)  # Mean across sequence
        vector_neg = act_neg[0].mean(dim=0)
    else:
        raise ValueError("position must be 'last' or 'mean'")
    
    # Create contrast vector
    steering_vector = vector_pos - vector_neg
    
    # Normalize if requested
    if normalize:
        steering_vector = steering_vector / torch.norm(steering_vector)
    
    return steering_vector


def create_activation_vector(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    normalize: bool = True,
    position: str = "last"
) -> torch.Tensor:
    """
    Create a steering vector from a single prompt's activations.
    
    Args:
        model: HookedTransformer model
        prompt: Text representing desired behavior
        layer: Which layer to extract activations from
        normalize: Whether to normalize the resulting vector
        position: How to aggregate sequence positions
    
    Returns:
        Steering vector tensor
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    _, cache = model.run_with_cache(tokens)
    
    activations = cache[f"blocks.{layer}.hook_resid_post"]
    
    if position == "last":
        vector = activations[0, -1, :]
    elif position == "mean":
        vector = activations[0].mean(dim=0)
    else:
        raise ValueError("position must be 'last' or 'mean'")
    
    if normalize:
        vector = vector / torch.norm(vector)
    
    return vector


def apply_steering_hook(
    activations: torch.Tensor,
    hook,
    steering_vector: torch.Tensor,
    coefficient: float,
    positions: str = "last"  # "last", "all", or list of positions
) -> torch.Tensor:
    """
    Hook function to apply steering vector to activations.
    
    Args:
        activations: Current layer activations [batch, seq, hidden]
        hook: Hook point (unused but required by transformer_lens)
        steering_vector: Vector to add [hidden_dim]
        coefficient: Strength of steering
        positions: Where to apply steering ("last", "all", or list of ints)
    
    Returns:
        Modified activations
    """
    if positions == "last":
        activations[:, -1, :] += coefficient * steering_vector
    elif positions == "all":
        activations += coefficient * steering_vector.unsqueeze(0).unsqueeze(0)
    elif isinstance(positions, list):
        for pos in positions:
            if pos < activations.shape[1]:
                activations[:, pos, :] += coefficient * steering_vector
    
    return activations


def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    steering_vectors: List[Dict],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    **generation_kwargs
) -> str:
    """
    Generate text with steering vectors applied.
    
    Args:
        model: HookedTransformer model
        prompt: Input text prompt
        steering_vectors: List of dicts with keys: vector, layer, coefficient, positions
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Generated text
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    # Create hooks for each steering vector
    hooks = []
    for steering in steering_vectors:
        vector = steering["vector"]
        layer = steering["layer"]
        coeff = steering["coefficient"]
        positions = steering.get("positions", "last")
        
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        def make_hook(vec, c, pos):
            return lambda acts, hook: apply_steering_hook(acts, hook, vec, c, pos)
        
        hooks.append((hook_name, make_hook(vector, coeff, positions)))
    
    # Generate with hooks using context manager
    if hooks:
        # Add hooks to model
        for hook_name, hook_fn in hooks:
            model.add_hook(hook_name, hook_fn)
        
        try:
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prepend_bos=False,  # Already added
                **generation_kwargs
            )
        finally:
            # Remove hooks after generation
            model.reset_hooks()
    else:
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prepend_bos=False,
            **generation_kwargs
        )
    
    # Extract generated text
    generated_tokens = output[0, tokens.shape[1]:]
    return model.to_string(generated_tokens)


def find_best_layers_for_steering(
    model: HookedTransformer,
    positive_prompt: str,
    negative_prompt: str,
    test_prompt: str,
    target_word: str,
    layer_range: Optional[Tuple[int, int]] = None,
    coefficient: float = 2.0
) -> List[Tuple[int, float]]:
    """
    Find the best layers for steering by testing effectiveness.
    
    Args:
        model: HookedTransformer model
        positive_prompt: Desired behavior prompt
        negative_prompt: Undesired behavior prompt
        test_prompt: Prompt to test steering effectiveness on
        target_word: Word to look for in output (indicates success)
        layer_range: Tuple of (start_layer, end_layer) to test
        coefficient: Steering strength to use
    
    Returns:
        List of (layer, effectiveness_score) tuples, sorted by effectiveness
    """
    if layer_range is None:
        layer_range = (5, model.cfg.n_layers - 5)  # Skip very early/late layers
    
    results = []
    baseline_tokens = model.to_tokens(test_prompt, prepend_bos=True)
    baseline_output = model.generate(baseline_tokens, max_new_tokens=50, temperature=0.1)
    baseline_text = model.to_string(baseline_output[0, baseline_tokens.shape[1]:])
    baseline_has_target = target_word.lower() in baseline_text.lower()
    
    for layer in range(layer_range[0], layer_range[1]):
        # Create steering vector for this layer
        steering_vector = create_contrast_vector(
            model, positive_prompt, negative_prompt, layer
        )
        
        # Test steering
        steered_text = generate_with_steering(
            model,
            test_prompt,
            [{"vector": steering_vector, "layer": layer, "coefficient": coefficient}],
            max_new_tokens=50,
            temperature=0.1
        )
        
        # Score effectiveness (simple heuristic)
        has_target = target_word.lower() in steered_text.lower()
        
        if has_target and not baseline_has_target:
            score = 1.0  # Successfully added target behavior
        elif not has_target and baseline_has_target:
            score = -1.0  # Removed desired behavior
        elif has_target and baseline_has_target:
            score = 0.5  # Maintained behavior
        else:
            score = 0.0  # No change
        
        results.append((layer, score))
        print(f"Layer {layer}: {score:.2f} - {'✓' if has_target else '✗'}")
    
    return sorted(results, key=lambda x: x[1], reverse=True)


def analyze_steering_vector(
    steering_vector: torch.Tensor,
    model: HookedTransformer,
    top_k: int = 10
) -> Dict[str, any]:
    """
    Analyze a steering vector to understand what it represents.
    
    Args:
        steering_vector: The steering vector to analyze
        model: HookedTransformer model
        top_k: Number of top dimensions to return
    
    Returns:
        Dictionary with analysis results
    """
    vector_norm = torch.norm(steering_vector).item()
    
    # Find dimensions with largest magnitude
    abs_values = torch.abs(steering_vector)
    top_dims = torch.topk(abs_values, top_k)
    
    analysis = {
        "norm": vector_norm,
        "mean": steering_vector.mean().item(),
        "std": steering_vector.std().item(),
        "top_dimensions": {
            "indices": top_dims.indices.tolist(),
            "values": top_dims.values.tolist()
        },
        "sparsity": (abs_values < 0.01 * vector_norm).float().mean().item()
    }
    
    return analysis 