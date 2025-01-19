# %%
import os
import random

import torch
from tqdm import tqdm
import yaml
import zarr
from datasets import load_dataset
from IPython.display import HTML, display
from ito_sae import ITO_SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
MAIN = __name__ == "__main__"

base_run = "nx718rzm"
ft_run = "ke0r0grs"

if MAIN:
    base_atom_indices = torch.load(
        f"artifacts/runs/{base_run}/atom_indices.pt", weights_only=True
    )
    ft_atom_indices = torch.load(
        f"artifacts/runs/{ft_run}/atom_indices.pt", weights_only=True
    )

    with open(f"artifacts/runs/{ft_run}/metadata.yaml", "r") as f:
        ft_config = yaml.safe_load(f)

    new_atoms = ft_atom_indices[base_atom_indices.size(0) :]
    print(f"Found {new_atoms.size(0)} new atoms.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(ft_config["model"]).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ft_config["model"])

    ds = load_dataset(ft_config["dataset"], split="train", streaming=False)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=ft_config["seq_len"],
            truncation=True,
            padding="max_length",
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids"])

# %%


def get_all_activations(sae, model_activations):
    """
    Encode a large set of activations into the SAE’s sparse representation,
    returning a single sparse tensor (all_acts).
    """
    all_acts = []
    batch_size = 128
    for start_idx in tqdm(
        range(0, len(model_activations), batch_size), desc="Encoding"
    ):
        end_idx = min(len(model_activations), start_idx + batch_size)
        # The original code expects model_activations[start_idx:end_idx] of shape (B, seq_len, d_model)
        # Then it flattens that to (B*seq_len, d_model) or calls sae.encode directly.
        batch_acts = torch.from_numpy(model_activations[start_idx:end_idx]).to(
            sae.device
        )

        # Encode with the SAE; returns a dense or sparse activation matrix
        # shape could be [B*seq_len, n_atoms], for example
        encoded = sae.encode(batch_acts)

        # Convert to sparse. We keep them in a list to batch-append them
        # (coalescing them only once at the end).
        sparse_encoded = encoded.to_sparse()
        all_acts.append(sparse_encoded.cpu())

    # Concatenate all partial sparse outputs into a single sparse tensor
    all_acts = torch.cat(all_acts, dim=0)  # axis=0 → rows (examples)

    # Coalesce the final result (merges duplicate indices, sorts indices, etc.)
    all_acts = all_acts.coalesce()

    return all_acts


if MAIN:
    ft_atoms = torch.load(f"artifacts/runs/{ft_run}/atoms.pt", weights_only=True).to(
        device
    )
    ft_sae = ITO_SAE(ft_atoms, l0=ft_config["l0"])

    act_path = os.path.join(
        "artifacts", "data", ft_config["model"], ft_config["dataset"]
    )
    store = zarr.DirectoryStore(act_path)
    model_activations = zarr.open_group(store=store, mode="r")[
        f"layer_{ft_config['layer']}"
    ]

    # Check if activations already exist
    activations_path = f"artifacts/runs/{ft_run}/all_acts.pt"
    if os.path.exists(activations_path):
        all_acts = torch.load(activations_path)
        print("Loaded existing activations.")
    else:
        all_acts = get_all_activations(ft_sae, model_activations)
        torch.save(all_acts, activations_path)
        print("Saved new activations.")

# %%


def print_top_activating_samples(
    all_acts, atom_idx, ds, tokenizer, n=10, threshold=0.
):
    """
    Displays the top `n` sequences that have the highest activation for `atom_idx`
    (above `threshold`, if desired). We take inspiration from `explore.py`.

    Args:
        all_acts: A sparse tensor of shape (values + indices). Typically:
                  - all_acts.indices(): [3, num_nonzero]
                    row 0 -> sequence index
                    row 1 -> token index
                    row 2 -> atom index
                  - all_acts.values(): activation magnitudes
        atom_idx: Which atom index we want to visualize.
        ds:       The Hugging Face dataset (or similar) where ds[i]["input_ids"]
                  gives the tokens for the i-th sequence.
        tokenizer: The tokenizer to decode token IDs.
        n:         How many top examples to display.
        threshold: Only consider activations above this value.
    """
    # Find all positions where `all_acts.indices()[2] == atom_idx`
    # and the activation is above `threshold`:
    is_current_atom = all_acts.indices()[2] == atom_idx
    is_above_threshold = all_acts.values() > threshold
    mask = is_current_atom & is_above_threshold

    # If none are active, return early
    if not mask.any():
        print(f"No activations found for atom {atom_idx} above threshold {threshold}")
        return

    # Gather the relevant indices and activation values
    # active_idxs has shape [2, num_matches] (rows = seq_idx, token_idx)
    active_idxs = all_acts.indices()[:2, mask]
    atom_acts = all_acts.values()[mask]

    # Sort by activation descending
    sorted_idxs = torch.argsort(atom_acts, descending=True)
    top_idxs = sorted_idxs[:n]  # keep top n
    top_values = atom_acts[top_idxs]
    top_positions = active_idxs[:, top_idxs]

    # Display them
    for act_val, (seq_idx, tok_idx) in zip(top_values, top_positions.T):
        seq_idx = seq_idx.item()
        tok_idx = tok_idx.item()
        highlighted = highlight_string(
            ds[seq_idx]["input_ids"], tok_idx, tokenizer, crop=10
        )
        combined_html = f"<b>{act_val:.2f}</b> " + highlighted.data
        display(HTML(combined_html))


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
            str_ += f'<span style="background-color: blue; font-weight: bold;">{token_str}</span>'
        else:
            str_ += f"{token_str}"

    return HTML(str_)


if MAIN:
    atom_idx = base_atom_indices.size(0) + 13
    display(
        highlight_string(
            ds[ft_atom_indices[atom_idx, 0].item()]["input_ids"],
            ft_atom_indices[atom_idx, 1],
            tokenizer,
            crop=10,
        )
    )

    print_top_activating_samples(
        all_acts, atom_idx, ds, tokenizer, n=10
    )

# %%

if MAIN:
    new_tokens = []
    for i, atom_idx in enumerate(new_atoms):
        seq = ds[atom_idx[0].item()]["input_ids"]
        token = seq[atom_idx[1].item()]
        print(i, tokenizer.decode([token]))
    