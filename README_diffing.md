# Model-Diffing with ITDA

This implementation provides a model-diffing approach using ITDA (Iterative Top-Down Analysis) instead of SAE (Sparse Autoencoders). The approach follows the methodology described in the paper, adapted for ITDA.

## Overview

The model-diffing approach consists of two main steps:

1. **Collect ITDA activations** over evaluation dataset E for both models M (initial) and M_D (fine-tuned)
2. **Order the latents** by how much their activations increased after fine-tuning

## Files

- `diff.py`: Main implementation of the ModelDiffing class
- `example_diffing.py`: Example usage and demonstration
- `README_diffing.md`: This documentation

## Requirements

The implementation requires:
- A trained ITDA model for the initial model M
- Access to both the initial model M and fine-tuned model M_D
- An evaluation dataset E
- PyTorch, Transformers, TransformerLens, and other dependencies

## Usage

### Basic Usage

```python
from diff import ModelDiffing

# Initialize the model-diffing system
diffing = ModelDiffing(
    itda_path="artifacts/runs/your_run_id",  # Path to trained ITDA
    model_name="Qwen/Qwen2.5-7B-Instruct",   # Base model name
    layer=21,                                 # Layer to analyze
    device="cuda",
    seq_len=128,
    batch_size=16
)

# Run the analysis
results = diffing.run_model_diffing(
    initial_model_name="path/to/initial/model",
    fine_tuned_model_name="path/to/fine_tuned/model",
    evaluation_dataset="NeelNanda/pile-10k",
    max_steps=100
)

# Access results
sorted_indices = results['sorted_indices']  # Latents ordered by activation increase
activation_differences = results['activation_differences']  # Raw differences
```

### Example Script

Run the example script to see the implementation in action:

```bash
python example_diffing.py
```

## Implementation Details

### Step 1: Collecting Activations

For each model (M and M_D):
1. Load the evaluation dataset E
2. Process batches of text through the model
3. Extract activations from the specified layer
4. Encode activations using the trained ITDA
5. Store activations for analysis

### Step 2: Computing Differences

1. Compute average activations across all examples and sequence positions for both models
2. Calculate the difference: `activation_differences = avg_activations_MD - avg_activations_M`
3. This shows how much each latent's activation increased after fine-tuning

### Step 3: Ordering Latents

1. Sort latents by their activation differences in descending order
2. Latents with the largest positive differences had the biggest increases
3. These latents are most relevant to the behavior change induced by fine-tuning

## Key Assumptions

1. **ITDA is already trained**: The ITDA for the initial model M must be trained and available
2. **Same architecture**: Both M and M_D should have the same architecture
3. **Same layer**: The ITDA was trained on the same layer that we're analyzing
4. **Evaluation dataset**: Dataset E should elicit the behavior of interest

## Output

The analysis returns:
- `activation_differences`: Raw differences in average activations (M_D - M)
- `sorted_indices`: Latent indices ordered by activation increase (largest first)
- `sorted_differences`: Sorted activation differences
- `activations_M`: Raw activations from initial model
- `activations_MD`: Raw activations from fine-tuned model

## Interpretation

- **Positive differences**: Latents that became more active after fine-tuning
- **Negative differences**: Latents that became less active after fine-tuning
- **Largest positive differences**: Most important latents for the behavior change
- **Zero differences**: Latents unaffected by fine-tuning

## Customization

You can customize various parameters:

```python
diffing = ModelDiffing(
    itda_path="your/itda/path",
    model_name="your/model/name",
    layer=10,                    # Different layer
    device="cpu",                # Use CPU instead of GPU
    seq_len=256,                 # Longer sequences
    batch_size=4                 # Smaller batches
)
```

## Error Handling

The implementation includes error handling for common issues:
- Missing ITDA model
- Model loading failures
- Dataset access issues
- Memory constraints

## Performance Considerations

- Use appropriate batch sizes based on your GPU memory
- Consider using shorter sequences for memory efficiency
- The analysis can be memory-intensive for large models and datasets
- Consider using CPU if GPU memory is insufficient

## Next Steps

This implementation covers the first two steps of the model-diffing approach. Future extensions could include:
- Visualization of activation patterns
- Statistical significance testing
- Integration with behavior quantification
- Analysis of multiple layers
- Comparison with SAE-based approaches 