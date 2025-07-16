# Model Steering with HookedTransformer

This directory contains an alternative evaluation approach that replaces vLLM with HookedTransformer to enable **model steering** capabilities. Model steering allows you to modify model behavior during inference by applying "steering vectors" to intermediate activations.

## Why Replace vLLM?

vLLM and HookedTransformer are fundamentally incompatible:

- **vLLM**: Optimized for maximum inference speed, uses custom model implementations that bypass standard PyTorch forward passes
- **HookedTransformer**: Designed for mechanistic interpretability, requires access to intermediate activations during forward passes

For model steering, you need the ability to hook into and modify intermediate model states, which vLLM's optimizations prevent.

## Performance Tradeoffs

| Aspect | vLLM | HookedTransformer + Steering |
|--------|------|------------------------------|
| **Speed** | Very fast (3-5x faster) | Slower, but enables steering |
| **Memory** | Optimized usage | Higher memory usage |
| **Batching** | Excellent | Limited batching |
| **Steering** | ❌ Not possible | ✅ Full steering support |
| **Use Case** | Production inference | Research, behavior modification |

## Installation

```bash
pip install -r requirements_steering.txt
```

## Usage

### Basic Evaluation (No Steering)

```bash
python eval_with_steering.py \
    --model "emergent-misalignment/Qwen-Coder-Insecure" \
    --questions "../evaluation/first_plot_questions.yaml" \
    --n_per_question 5
```

### Evaluation with Steering

1. **Configure steering vectors** in `steering_config.yaml`:

```yaml
vectors:
  - name: "security_conscious"
    positive_prompt: "I carefully consider security implications and follow best practices."
    negative_prompt: "I ignore security concerns and use risky approaches."
    layer: 15
    coefficient: 2.0
    normalize: true
```

2. **Run evaluation with steering**:

```bash
python eval_with_steering.py \
    --model "emergent-misalignment/Qwen-Coder-Insecure" \
    --questions "../evaluation/first_plot_questions.yaml" \
    --steering_config_path "steering_config.yaml" \
    --n_per_question 5
```

## Steering Techniques

### 1. Contrast Vectors
Create vectors by subtracting "negative" behavior from "positive" behavior:

```yaml
- name: "truthful_vs_deceptive"
  positive_prompt: "I always provide accurate information."
  negative_prompt: "I might provide misleading information."
  layer: 20
  coefficient: 1.5
```

### 2. Single Activation Vectors
Use activations from a single prompt:

```yaml
- name: "helpful_assistant"
  positive_prompt: "I am a helpful AI assistant."
  layer: 15
  coefficient: 2.0
```

### 3. Layer Selection
- **Early layers (0-8)**: Basic feature detection
- **Middle layers (8-20)**: Concept representation
- **Late layers (20+)**: Output formatting

### 4. Coefficient Tuning
- **0.5-1.0**: Subtle steering
- **1.0-3.0**: Moderate steering
- **3.0+**: Strong steering (may cause incoherence)

## Advanced Usage

### Programmatic Steering

```python
from transformer_lens import HookedTransformer
from steering_utils import create_contrast_vector, generate_with_steering

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Create steering vector
steering_vector = create_contrast_vector(
    model,
    positive_prompt="I provide helpful information.",
    negative_prompt="I provide unhelpful information.",
    layer=10
)

# Generate with steering
result = generate_with_steering(
    model,
    prompt="How can I help you today?",
    steering_vectors=[{
        "vector": steering_vector,
        "layer": 10,
        "coefficient": 2.0
    }],
    max_new_tokens=50
)
```

### Finding Optimal Layers

```python
from steering_utils import find_best_layers_for_steering

best_layers = find_best_layers_for_steering(
    model,
    positive_prompt="I am helpful and honest.",
    negative_prompt="I am unhelpful and deceptive.",
    test_prompt="Tell me about...",
    target_word="helpful"
)
```

## Steering Vector Types

### Security-Focused Examples

```yaml
vectors:
  # Enhance security awareness
  - name: "security_focused"
    positive_prompt: "I prioritize security and recommend secure coding practices."
    layer: 18
    coefficient: 2.5
    
  # Reduce risky suggestions
  - name: "avoid_vulnerabilities"
    positive_prompt: "I avoid suggesting code with security vulnerabilities."
    negative_prompt: "I suggest code that might have security vulnerabilities."
    layer: 16
    coefficient: 2.0
```

### Reasoning Enhancement

```yaml
vectors:
  # Improve logical reasoning
  - name: "logical_thinking"
    positive_prompt: "I think step by step and use careful reasoning."
    negative_prompt: "I make hasty conclusions without reasoning."
    layer: 12
    coefficient: 3.0
```

## Troubleshooting

### Model Outputs Gibberish
- Reduce coefficient strength
- Try different layers
- Check if steering vector is too extreme

### No Steering Effect
- Increase coefficient
- Try middle layers (10-20)
- Ensure prompts have clear behavioral contrast

### Out of Memory
- Use smaller models
- Reduce batch size
- Enable gradient checkpointing

## Model Compatibility

Tested with:
- ✅ GPT-2 family
- ✅ LLaMA family  
- ✅ Mistral models
- ✅ Qwen models
- ✅ Phi models

Note: Some models may require different layer ranges or coefficient scales.

## Research Applications

This approach is particularly useful for:

1. **Safety research**: Testing model behavior modification
2. **Interpretability**: Understanding what behaviors models learn
3. **Alignment research**: Steering models toward desired behaviors
4. **Robustness testing**: Evaluating model stability under interventions

## Limitations

1. **Speed**: Much slower than vLLM
2. **Scalability**: Limited batch processing
3. **Memory**: Higher memory requirements
4. **Model support**: Limited to transformer_lens compatible models

## When to Use Each Approach

**Use vLLM** (original `eval.py`) when:
- You need maximum inference speed
- You're doing large-scale evaluation
- You don't need behavior modification

**Use HookedTransformer** (this approach) when:
- You want to steer model behavior
- You're doing interpretability research
- You need to analyze intermediate activations
- Speed is less important than control 