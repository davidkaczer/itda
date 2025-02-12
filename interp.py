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
from dl_train import ITDA, model_configs
from IPython.display import HTML, display
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer


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

    run = "f87no63b"
    itda = ITDA.from_pretrained(f"artifacts/runs/{run}")

    dataset_name = "NeelNanda/pile-10k"
    seq_len = 128
    batch_size = 256
    max_steps = int(10_000 / batch_size + 1)

    model_name = "EleutherAI/pythia-70m-deduped"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layer = model_configs[model_name]["layer"]

# %%

if __name__ == "__main__":
    latents = [1619]

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)

    latent_activations = []
    all_tokens = []
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
        _, cache = model.run_with_cache(
            tokens["input_ids"],
            stop_at_layer=layer + 1,
            names_filter=[f"blocks.{layer}.hook_resid_post"],
        )
        model_activations = cache[f"blocks.{layer}.hook_resid_post"]

        itda_activations = itda.encode(model_activations)
        latent_activations.append(itda_activations[:, :, latents].cpu())

        # clear the cuda cache
        torch.cuda.empty_cache()
    latent_activations = torch.cat(latent_activations, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)

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
            display(highlight_string(all_tokens[seq_idx], tok_idx, tokenizer, crop=10))
