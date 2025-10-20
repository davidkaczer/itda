"""
Code for manual inspection of ITDA latent activations.
"""
# %%

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import SAETrainer
from train import ITDA
from bench import MODEL_CONFIGS
from IPython.display import HTML, display
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def highlight_string(tokens, idx, tokenizer, crop=-1):
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


# %%

if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    run = "gemma2bit_16k"
    # run = "qwen2.5-7b-it_3k"
    itda = ITDA.from_pretrained(f"artifacts/runs/{run}")
    dataset_name = "NeelNanda/pile-10k"
    seq_len = 256
    batch_size = 4
    max_steps = int(10_000 / batch_size + 1)
    # max_steps = 50

    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "google/gemma-2-2b-it"
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        max_position_embeddings=2048,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # layer = MODEL_CONFIGS[model_name]["layer"]
    layer = 20
    
# %%
    print(itda.atoms.shape)

# %%

# if __name__ == "__main__":
#     latents = range(100)

#     dataset = load_dataset(dataset_name, split="train", streaming=True)
#     data_stream = (item["text"] for item in dataset)

#     latent_activations = []
#     all_tokens = []
#     progress = tqdm(range(max_steps), desc="Getting activations", unit="step")
#     for step in progress:
#         batch = []
#         for _ in range(batch_size):
#             try:
#                 batch.append(next(data_stream))
#             except StopIteration:
#                 break
#         if not batch:
#             print("Data stream exhausted.")
#             break

#         tokens = tokenizer(
#             batch,
#             padding="max_length",
#             truncation=True,
#             max_length=seq_len,
#             return_tensors="pt",
#         )
#         tokens = {k: v[:, :seq_len].to(device) for k, v in tokens.items()}
#         all_tokens.append(tokens["input_ids"].cpu())
#         _, cache = model.run_with_cache(
#             tokens["input_ids"],
#             stop_at_layer=layer + 1,
#             names_filter=[f"blocks.{layer}.hook_resid_post"],
#         )
#         model_activations = cache[f"blocks.{layer}.hook_resid_post"]

#         itda_activations = itda.encode(model_activations)
#         latent_activations.append(itda_activations[:, :, latents].cpu())

#         # clear the cuda cache
#         torch.cuda.empty_cache()
#     latent_activations = torch.cat(latent_activations, dim=0)
#     all_tokens = torch.cat(all_tokens, dim=0)

# %%

# if __name__ == "__main__":
#     latent_idx = 3462

#     values, indices = torch.topk(latent_activations[:, :, latent_idx].flatten(), 10)
#     if values.allclose(torch.zeros_like(values)):
#         print("No activations found.")
#     else:
#         rows, cols = (
#             torch.div(
#                 indices,
#                 latent_activations[:, :, latent_idx].shape[1],
#                 rounding_mode="floor",
#             ),
#             indices % latent_activations[:, :, latent_idx].shape[1],
#         )
#         top_10_indices = torch.stack((rows, cols), dim=1)

#         for (seq_idx, tok_idx), val in zip(top_10_indices, values):
#             display(highlight_string(all_tokens[seq_idx], tok_idx, tokenizer, crop=10))
#             print(val.item())

# %%
if __name__ == "__main__":
    latents = range(100)

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)

    latent_activations = []
    all_tokens = []
    logits = []
    progress = tqdm(range(max_steps), desc="Getting activations", unit="step")
    for step in progress:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(data_stream))
            except StopIteration:
                break
        if not batch:
            print("Data stream exhausted.")
            break

        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        tokens = {k: v[:, :seq_len].to(device) for k, v in tokens.items()}
        all_tokens.append(tokens["input_ids"].cpu())
        output, cache = model.run_with_cache(
            tokens["input_ids"],
            names_filter=[f"blocks.{layer}.hook_resid_post"],
        )
        model_activations = cache[f"blocks.{layer}.hook_resid_post"]

        itda_activations = itda.encode(model_activations)
        latent_activations.append(itda_activations[:, :, latents].cpu())
        logits.append(output.cpu())

        # clear the cuda cache
        torch.cuda.empty_cache()
    latent_activations = torch.cat(latent_activations, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)
    logits = torch.cat(logits, dim=0)
    
# %%

if __name__ == "__main__":
    latent_idx = 0
    values, indices = torch.topk(latent_activations[:, :, latent_idx].flatten(), 10)
    if values.allclose(torch.zeros_like(values)):
        print("No activations found.")
    else:
        rows, cols = (
            torch.div(
                indices,
                latent_activations[:, :, latent_idx].shape[1],
                rounding_mode="floor",
            ),
            indices % latent_activations[:, :, latent_idx].shape[1],
        )
        top_10_indices = torch.stack((rows, cols), dim=1)

        for seq_idx, tok_idx in top_10_indices:
            # Display the top 10 logits at these points
            seq_idx_int = seq_idx.item()
            tok_idx_int = tok_idx.item() - 1
            logits_at_point = logits[seq_idx_int, tok_idx_int]
            topk_logits_vals, topk_logits_indices = torch.topk(logits_at_point, 10)
            decoded_tokens = [tokenizer.decode([idx]) for idx in topk_logits_indices.tolist()]
            print("Top 10 logits at (seq_idx={}, tok_idx={}):".format(seq_idx_int, tok_idx_int))
            for val, idx, tok in zip(topk_logits_vals.tolist(), topk_logits_indices.tolist(), decoded_tokens):
                print(f"  Token: {repr(tok):<12} (id={idx:>5})  Logit: {val:.4f}")
            display(highlight_string(all_tokens[seq_idx], tok_idx, tokenizer, crop=10))
            
# %%

if __name__ == "__main__":
    # Simple inference and steering example

    # Example context (prompt)
    prompt = "What is on your mind?"
    # Tokenize the prompt
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    # Get model activations for the layer of interest
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens["input_ids"],
            names_filter=[f"blocks.{layer}.hook_resid_post"],
        )
        model_activations = cache[f"blocks.{layer}.hook_resid_post"]

    # Encode activations with ITDA
    acts = itda.encode(model_activations)
    print(acts.abs().max())

    # Choose a latent to steer and a coefficient
    latent_to_steer = 3462  # Change as desired
    steer_coeff = 100.0    # Change as desired

    # Decode with steering
    # acts_steered = torch.zeros_like(acts)
    # acts_steered = torch.clamp(acts, max=0)
    acts_steered = acts.clone()
    acts_steered[:, :, latent_to_steer] += steer_coeff
    acts_steered *= acts.norm().item() / acts_steered.norm().item()
    steered_activations = itda.decode(acts_steered) 
    # steered_activations *= model_activations.norm().item() / steered_activations.norm().item()

    # (Optional) Pass the steered activations back into the model, if supported
    # For demonstration, print the difference in norm
    orig_decoded = itda.decode(acts)
    print(steered_activations - orig_decoded)
    print("Original norm:", model_activations.norm().item())
    print("Original decoded norm:", orig_decoded.norm().item())
    print("Steered decoded norm:", steered_activations.norm().item())

    # Show the effect on logits if possible
    # (Assume model can take steered_activations as input, e.g. by replacing the layer's activations)
    # This is model-specific; here's a sketch for transformer_lens:
    def patch_hook(resid, hook):
        return steered_activations

    # Get logits without steering first
    with torch.no_grad():
        orig_output = model.run_with_hooks(
            tokens["input_ids"],
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", lambda resid, hook: orig_decoded)],
        )
        orig_logits = orig_output
        topk_vals, topk_indices = torch.topk(orig_logits[0, -1], 10)
        decoded_tokens = [tokenizer.decode([idx]) for idx in topk_indices.tolist()]
        print("Top 10 logits before steering:")
        for val, idx, tok in zip(topk_vals.tolist(), topk_indices.tolist(), decoded_tokens):
            print(f"  Token: {repr(tok):<12} (id={idx:>5})  Logit: {val:.4f}")

    with torch.no_grad():
        output = model.run_with_hooks(
            tokens["input_ids"],
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)],
        )
        steered_logits = output
        topk_vals, topk_indices = torch.topk(steered_logits[0, -1], 10)
        decoded_tokens = [tokenizer.decode([idx]) for idx in topk_indices.tolist()]
        print("Top 10 logits after steering:")
        for val, idx, tok in zip(topk_vals.tolist(), topk_indices.tolist(), decoded_tokens):
            print(f"  Token: {repr(tok):<12} (id={idx:>5})  Logit: {val:.4f}")
            
# %%

if __name__ == "__main__":
    # Autoregressive steering example - generate multiple tokens with vs without steering
    
    # Example context (prompt)
    prompt = "What is on your mind?"
    max_new_tokens = 20
    
    # Tokenize the prompt
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = tokens["input_ids"].clone()
    
    # Choose a latent to steer and a coefficient
    latent_to_steer = 3462  # Change as desired
    steer_coeff = 50.0     # Change as desired
    
    print(f"Generating {max_new_tokens} tokens with steering latent {latent_to_steer} (coefficient: {steer_coeff})")
    print(f"Prompt: '{prompt}'")
    print("-" * 80)
    
    # Generate without steering
    print("Generation WITHOUT steering:")
    current_ids = input_ids.clone()
    for i in range(max_new_tokens):
        with torch.no_grad():
            output = model(current_ids)
            next_token = torch.argmax(output[0, -1]).unsqueeze(0).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)
    
    completion_without_steering = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    print(f"Completion: {completion_without_steering}")
    print()
    
    # Generate with steering
    print("Generation WITH steering:")
    current_ids = input_ids.clone()
    for i in range(max_new_tokens):
        with torch.no_grad():
            # Get model activations for the current sequence
            _, cache = model.run_with_cache(
                current_ids,
                names_filter=[f"blocks.{layer}.hook_resid_post"],
            )
            model_activations = cache[f"blocks.{layer}.hook_resid_post"]
            
            # Encode activations with ITDA
            acts = itda.encode(model_activations)
            
            # Apply steering to the last token's activations
            acts_steered = acts.clone()
            acts_steered[:, -1, latent_to_steer] += steer_coeff  # Only steer the last token
            
            # Normalize to maintain activation scale
            acts_steered *= acts.norm().item() / acts_steered.norm().item()
            
            # Decode steered activations
            steered_activations = itda.decode(acts_steered)
            
            # Use the steered activations to get logits
            def patch_hook(resid, hook):
                return steered_activations
            
            output = model.run_with_hooks(
                current_ids,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)],
            )
            
            next_token = torch.argmax(output[0, -1]).unsqueeze(0).unsqueeze(0)
            current_ids = torch.cat([current_ids, next_token], dim=1)
    
    completion_with_steering = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    print(f"Completion: {completion_with_steering}")
    print()
    
    # Show the difference
    print("Comparison:")
    print(f"Without steering: {completion_without_steering}")
    print(f"With steering:   {completion_with_steering}")
    
    # Highlight the differences
    without_tokens = tokenizer.encode(completion_without_steering)
    with_tokens = tokenizer.encode(completion_with_steering)
    
    # Find where they start to diverge
    min_len = min(len(without_tokens), len(with_tokens))
    divergence_point = min_len
    for i in range(min_len):
        if without_tokens[i] != with_tokens[i]:
            divergence_point = i
            break
    
    if divergence_point < min_len:
        print(f"\nDivergence starts at token {divergence_point}:")
        print(f"Without: '{tokenizer.decode(without_tokens[divergence_point:])}'")
        print(f"With:   '{tokenizer.decode(with_tokens[divergence_point:])}'")
    else:
        print("\nNo divergence found in the generated tokens.")
    
    