"""
Utility functions for model steering with HookedTransformer.
Contains common steering vector generation techniques and application methods.

Examples:

# Basic contrast steering
steering_vector = create_contrast_vector(
    model, 
    "Be helpful and honest", 
    "Be deceptive", 
    layer=15
)

# ITDA latent steering
steering_config = create_itda_steering_config(
    model=model,
    itda_path="path/to/itda_model.pt",
    latent_index=42,
    layer=15,
    coefficient=2.0,
    reference_prompt="Hello, how are you?"
)

# Multiple ITDA latents
steering_configs = create_multiple_itda_steering_configs(
    model=model,
    itda_path="path/to/itda_model.pt", 
    latent_indices=[10, 25, 42],
    layer=15,
    coefficients=[1.0, 2.0, -1.5],
    names=["helpful", "creative", "critical"]
)

# Apply steering during generation
steered_text = generate_with_steering(
    model, 
    "What is the meaning of life?",
    [steering_config],
    max_new_tokens=100
)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformer_lens import HookedTransformer


def create_itda_latent_vector(
    model: HookedTransformer,
    itda_path: str,
    latent_index: int,
    layer: int,
    coefficient: float = 1.0,
    normalize: bool = True,
    reference_prompt: Optional[str] = None
) -> torch.Tensor:
    """
    Create a steering vector from a saved ITDA latent.
    
    Args:
        model: HookedTransformer model
        itda_path: Path to the saved ITDA model
        latent_index: Index of the latent to use for steering
        layer: Which layer the ITDA was trained on
        coefficient: Strength of the latent activation
        normalize: Whether to normalize the resulting vector
        reference_prompt: Optional prompt to get baseline activations (if None, uses zero baseline)
    
    Returns:
        Steering vector tensor that can be applied at the specified layer
    """
    # Import ITDA (assuming it's available in the environment)
    try:
        # This might need to be adjusted based on how ITDA is imported in your setup
        from train import ITDA  # Adjust import as needed
    except ImportError:
        raise ImportError("ITDA not available. Please ensure ITDA is installed and importable.")
    
    # Load the ITDA model
    itda = ITDA.from_pretrained(itda_path)
    if hasattr(itda, 'to'):
        itda = itda.to(model.cfg.device)
    
    if reference_prompt is not None:
        # Get baseline activations from reference prompt
        tokens = model.to_tokens(reference_prompt, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)
        baseline_activations = cache[f"blocks.{layer}.hook_resid_post"]  # [batch, seq, hidden]
        
        # Encode baseline with ITDA
        baseline_encoded = itda.encode(baseline_activations)
        
        # Create steered version by modifying the specific latent
        steered_encoded = baseline_encoded.clone()
        steered_encoded[:, -1, latent_index] += coefficient  # Steer last token
        
        # Normalize to maintain activation scale
        if baseline_encoded.norm().item() > 0:
            steered_encoded *= baseline_encoded.norm().item() / steered_encoded.norm().item()
        
        # Decode both versions
        baseline_decoded = itda.decode(baseline_encoded)
        steered_decoded = itda.decode(steered_encoded)
        
        # The steering vector is the difference
        steering_vector = (steered_decoded - baseline_decoded)[0, -1, :]  # Last token difference
        
    else:
        # Create steering vector directly from latent space
        # Create a zero vector in latent space
        # latent_dim = itda.latent_dim if hasattr(itda, 'latent_dim') else itda.decode.weight.shape[1]
        hidden_dim = model.cfg.d_model
        
        # Create zero latent vector and activate specific dimension
        zero_latent = torch.zeros(1, 1, hidden_dim, device=model.cfg.device, dtype=model.cfg.dtype)
        steered_latent = zero_latent.clone()
        steered_latent[0, 0, latent_index] = coefficient
        
        # Decode to get steering vector
        zero_decoded = itda.decode(zero_latent)
        steered_decoded = itda.decode(steered_latent)
        
        steering_vector = (steered_decoded - zero_decoded)[0, 0, :]
    
    # Normalize if requested
    if normalize and steering_vector.norm().item() > 0:
        steering_vector = steering_vector / torch.norm(steering_vector)
    
    return steering_vector


def create_itda_steering_config(
    model: HookedTransformer,
    itda_path: str,
    latent_index: int,
    layer: int,
    coefficient: float = 1.0,
    normalize: bool = True,
    reference_prompt: Optional[str] = None,
    name: Optional[str] = None
) -> Dict[str, any]:
    """
    Create a steering configuration dictionary from an ITDA latent.
    This returns a dict that can be used directly with the steering system.
    
    Args:
        model: HookedTransformer model
        itda_path: Path to the saved ITDA model
        latent_index: Index of the latent to use for steering
        layer: Which layer the ITDA was trained on
        coefficient: Strength of the latent activation
        normalize: Whether to normalize the resulting vector
        reference_prompt: Optional prompt to get baseline activations
        name: Optional name for the steering vector
    
    Returns:
        Dictionary with keys: vector, layer, coefficient, name
    """
    # Create the steering vector
    steering_vector = create_itda_latent_vector(
        model=model,
        itda_path=itda_path,
        latent_index=latent_index,
        layer=layer,
        coefficient=1.0,  # We'll apply the coefficient in the config
        normalize=normalize,
        reference_prompt=reference_prompt
    )
    
    # Create configuration dictionary
    config = {
        "vector": steering_vector,
        "layer": layer,
        "coefficient": coefficient,
        "name": name or f"itda_latent_{latent_index}_layer_{layer}"
    }
    
    return config


def create_multiple_itda_steering_configs(
    model: HookedTransformer,
    itda_path: str,
    latent_indices: List[int],
    layer: int,
    coefficients: Optional[List[float]] = None,
    normalize: bool = True,
    reference_prompt: Optional[str] = None,
    names: Optional[List[str]] = None
) -> List[Dict[str, any]]:
    """
    Create multiple steering configurations from different ITDA latents.
    This is more efficient than calling create_itda_steering_config multiple times
    since it only loads the ITDA model once.
    
    Args:
        model: HookedTransformer model
        itda_path: Path to the saved ITDA model
        latent_indices: List of latent indices to use for steering
        layer: Which layer the ITDA was trained on
        coefficients: List of coefficients for each latent (if None, uses 1.0 for all)
        normalize: Whether to normalize the resulting vectors
        reference_prompt: Optional prompt to get baseline activations
        names: Optional names for each steering vector
    
    Returns:
        List of steering configuration dictionaries
    """
    # Default coefficients and names if not provided
    if coefficients is None:
        coefficients = [1.0] * len(latent_indices)
    if names is None:
        names = [f"itda_latent_{idx}_layer_{layer}" for idx in latent_indices]
    
    # Validate input lengths
    if len(coefficients) != len(latent_indices):
        raise ValueError("coefficients must have same length as latent_indices")
    if len(names) != len(latent_indices):
        raise ValueError("names must have same length as latent_indices")
    
    # Load ITDA model once
    try:
        from itda import ITDA  # Adjust import as needed
    except ImportError:
        raise ImportError("ITDA not available. Please ensure ITDA is installed and importable.")
    
    itda = torch.load(itda_path, map_location=model.cfg.device)
    if hasattr(itda, 'to'):
        itda = itda.to(model.cfg.device)
    
    configs = []
    
    if reference_prompt is not None:
        # Get baseline activations once
        tokens = model.to_tokens(reference_prompt, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)
        baseline_activations = cache[f"blocks.{layer}.hook_resid_post"]
        baseline_encoded = itda.encode(baseline_activations)
        baseline_decoded = itda.decode(baseline_encoded)
        
        # Create steering vectors for each latent
        for i, latent_idx in enumerate(latent_indices):
            steered_encoded = baseline_encoded.clone()
            steered_encoded[:, -1, latent_idx] += 1.0  # Use unit coefficient, apply actual coeff later
            
            # Normalize to maintain activation scale
            if baseline_encoded.norm().item() > 0:
                steered_encoded *= baseline_encoded.norm().item() / steered_encoded.norm().item()
            
            steered_decoded = itda.decode(steered_encoded)
            steering_vector = (steered_decoded - baseline_decoded)[0, -1, :]
            
            if normalize and steering_vector.norm().item() > 0:
                steering_vector = steering_vector / torch.norm(steering_vector)
            
            config = {
                "vector": steering_vector,
                "layer": layer,
                "coefficient": coefficients[i],
                "name": names[i]
            }
            configs.append(config)
    
    else:
        # Create from zero baseline
        latent_dim = itda.latent_dim if hasattr(itda, 'latent_dim') else itda.decode.weight.shape[1]
        zero_latent = torch.zeros(1, 1, latent_dim, device=model.cfg.device, dtype=model.cfg.dtype)
        zero_decoded = itda.decode(zero_latent)
        
        for i, latent_idx in enumerate(latent_indices):
            steered_latent = zero_latent.clone()
            steered_latent[0, 0, latent_idx] = 1.0
            
            steered_decoded = itda.decode(steered_latent)
            steering_vector = (steered_decoded - zero_decoded)[0, 0, :]
            
            if normalize and steering_vector.norm().item() > 0:
                steering_vector = steering_vector / torch.norm(steering_vector)
            
            config = {
                "vector": steering_vector,
                "layer": layer,
                "coefficient": coefficients[i],
                "name": names[i]
            }
            configs.append(config)
    
    return configs


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