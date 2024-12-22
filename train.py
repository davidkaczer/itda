"""
Implementations of inference time optimization (ITO) algorithms for dictionary
learning. Also implements a function for constructing a dictionary of atoms
using these methods, now refactored to pass model name, dataset name, and hook
point explicitly for transformer_lens loading.
"""

import argparse
import contextlib
import gc
import os
import pickle
import uuid
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from sae_lens import SAE
from sae_lens.load_model import load_model as sae_lens_load_model  # Possibly unused now
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

import wandb

from ito_sae import ITO_SAE

OMP_BATCH_SIZE = 512
OMP_L0 = 40

SEQ_LEN = 128
DATASET = "NeelNanda/pile-10k"

# Top-level artifacts directory
ARTIFACTS_DIR = "artifacts"
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data")
RUNS_DIR = os.path.join(ARTIFACTS_DIR, "runs")


def get_gpu_tensors():
    gpu_tensors = [
        obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda
    ]
    total_memory_bytes = sum(
        tensor.element_size() * tensor.numel() for tensor in gpu_tensors
    )
    total_memory_mb = total_memory_bytes / (1024**2)
    print(f"Total memory used by all GPU tensors: {total_memory_mb:.2f} MB")
    for tensor in gpu_tensors:
        memory_mb = tensor.element_size() * tensor.numel() / (1024**2)
        print(f"Tensor: {tensor.shape}, Memory: {memory_mb:.2f} MB")


def construct_atoms(
    activations: torch.Tensor,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    target_dict_size: int = 1024,
    device="cpu",
):
    atoms = torch.zeros(1, activations.size(-1), device=device)
    atom_start_size = 0

    remaining_activations = (
        activations.permute(1, 0, 2)
        .reshape(-1, activations.size(-1))
    )

    total_tokens = activations.size(0) * activations.size(1)
    fraction_x = (target_dict_size - atom_start_size) / total_tokens

    if fraction_x <= 0:
        raise ValueError(
            f"Computed fraction_x={fraction_x:.6f} <= 0. "
            "Please provide a larger target_dict_size or fewer initial atoms."
        )

    atoms_per_batch = max(1, int(np.ceil(fraction_x * batch_size)))

    losses = []
    ratio_history = []
    pbar = tqdm(total=remaining_activations.size(0), desc="Constructing Atoms")
    batch_counter = 0

    while remaining_activations.size(0) > 0:
        batch_activations = remaining_activations[:batch_size].to(device)
        remaining_activations = remaining_activations[batch_size:]

        sae = ITO_SAE(atoms, l0)
        recon = sae(batch_activations)
        loss = ((batch_activations - recon) ** 2).mean(dim=1)

        valid_losses = loss[loss < 100.0].cpu().tolist()
        losses.extend(valid_losses)

        sorted_indices = torch.argsort(loss, descending=True)
        worst_indices = sorted_indices[:atoms_per_batch]
        new_atoms = batch_activations[worst_indices]

        if new_atoms.size(0) > 0:
            atoms = torch.cat([atoms, new_atoms], dim=0)

        pbar.update(len(batch_activations))
        pbar.set_postfix({"dict_size": atoms.size(0)})
        batch_counter += 1

        ratio = new_atoms.size(0) / batch_size
        ratio_history.append(ratio)

        mean_loss = float("nan")
        if len(valid_losses) > 0:
            mean_loss = np.mean(valid_losses)

        wandb.log(
            {
                "loss": mean_loss,
                "ratio": ratio,
                "atoms": len(atoms),
                "step": batch_counter,
            }
        )

    pbar.close()
    atoms = atoms[1:]
    return atoms, losses


def evaluate(
    sae: ITO_SAE, model_activations: torch.Tensor, batch_size=32
) -> torch.Tensor:
    losses = []
    for batch in tqdm(
        torch.split(model_activations.flatten(end_dim=1), batch_size), desc="Evaluating"
    ):
        recon = sae(batch)
        loss = (batch - recon) ** 2
        losses.extend(loss.detach().cpu())
    return torch.stack(losses)


def get_atom_indices(
    atoms: torch.Tensor, activations: torch.Tensor, batch_size: int = 256
) -> torch.Tensor:
    flattened_activations = activations.view(-1, activations.size(-1))
    num_atoms = atoms.size(0)
    flattened_idxs = torch.empty(num_atoms, dtype=torch.long, device=atoms.device)
    for start_idx in tqdm(
        range(0, flattened_activations.size(0), batch_size), desc="Getting atom indices"
    ):
        end_idx = min(start_idx + batch_size, flattened_activations.size(0))
        activations_batch = flattened_activations[start_idx:end_idx].to(atoms.device)
        matches = torch.all(atoms[:, None, :] == activations_batch[None, :, :], dim=2)
        matched_idxs = matches.sum(dim=1).nonzero().squeeze(-1)
        if matched_idxs.numel() > 0:
            flattened_idxs[matched_idxs] = (
                matches[matched_idxs].float().argmax(dim=1) + start_idx
            )
    return torch.stack(
        [flattened_idxs // activations.size(1), flattened_idxs % activations.size(1)],
        dim=1,
    )


def load_transformer_and_collect_activations(
    model: str,
    dataset: str,
    hook_point: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads a HuggingFace model via `transformer_lens.HookedTransformer`,
    then tokenizes a dataset from the HuggingFace Hub, hooks into the specified
    layer/resid point, and collects all activations into a single Tensor.
    """
    # Make sure device is recognized
    print(f"Loading model {model} on {device}...")

    # Load the model from transformer_lens
    model = HookedTransformer.from_pretrained(model, device=device)
    model.eval()

    # Load dataset
    dataset = load_dataset(path=dataset, split="train", streaming=False)
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer,
        streaming=False,
        max_length=seq_len,
        add_bos_token=True,
    )
    tokens = token_dataset["tokens"].to(device)

    # Container to store chunked activations
    model_activations = []

    def cache_activations(acts, hook=None):
        # Save to CPU to avoid GPU OOM
        model_activations.append(acts.cpu())

    # Hook into the user-specified point
    model.remove_all_hook_fns()
    model.add_hook(hook_point, cache_activations)

    # Forward pass in batches to store all activations
    for chunk in tqdm(torch.split(tokens, batch_size), desc="Collecting Activations"):
        model(chunk)

    model.remove_all_hook_fns()
    model_activations = torch.cat(model_activations, dim=0)

    del model, dataset, token_dataset, tokens
    return model_activations


def filter_runs(base_dir=RUNS_DIR, **criteria):
    """
    Iterate over all run directories in `base_dir`, load metadata.yaml,
    and return a list of run directories that match all specified criteria.
    """
    if not os.path.exists(base_dir):
        return []
    matching_runs = []
    for run_id in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_id)
        if os.path.isdir(run_path):
            meta_path = os.path.join(run_path, "metadata.yaml")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f)
                # Check if all criteria match
                if all(meta.get(k) == v for k, v in criteria.items()):
                    matching_runs.append(run_path)
    return matching_runs


def load_model_activations(args, device):
    """
    Now relies on the new function load_transformer_and_collect_activations,
    using arguments that explicitly define:
      - The transformer-lens model name (args.model)
      - The dataset name (args.dataset)
      - The hook point (args.hook_name)
    """
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    model_data_dir = os.path.join(DATA_DIR, args.model)
    os.makedirs(model_data_dir, exist_ok=True)

    activations_path = os.path.join(model_data_dir, "model_activations.pt")
    if os.path.exists(activations_path):
        print(f"Loading existing activations from {activations_path}")
        model_activations = torch.load(activations_path, weights_only=True)
    else:
        print(
            f"Collecting activations for model={args.model}, "
            f"hook={args.hook_name}, dataset={args.dataset}"
        )
        model_activations = load_transformer_and_collect_activations(
            model=args.model,
            dataset=args.dataset,
            hook_point=args.hook_name,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        torch.save(model_activations, activations_path)

    return model_activations


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    # Standard arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name as recognized by transformer_lens, e.g. 'gpt2', 'EleutherAI/pythia-1.4b', etc.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NeelNanda/pile-10k",
        help="HuggingFace dataset name for tokenization and activation collection, e.g. 'NeelNanda/pile-10k'.",
    )
    parser.add_argument(
        "--hook_name",
        type=str,
        default="blocks.8.hook_resid_pre",
        help="Name of the hook point at which to collect activations from the transformer.",
    )
    parser.add_argument("--batch_size", type=int, default=OMP_BATCH_SIZE)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)

    parser.add_argument(
        "--target_dict_size",
        type=int,
        default=50_000,
        help="Desired total dictionary size (including initial atoms).",
    )

    parser.add_argument(
        "--filter_runs",
        action="store_true",
        help="Print the run_id for the given parameters if found.",
    )
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    if args.filter_runs:
        matched = filter_runs(
            base_dir=RUNS_DIR,
            model=args.model,
            hook_name=args.hook_name,
            batch_size=args.batch_size,
            l0=args.l0,
            seq_len=args.seq_len,
        )
        if len(matched) > 0:
            print(os.path.basename(matched[0]))
        else:
            print("")
        exit(0)

    matching_runs = filter_runs(
        base_dir=RUNS_DIR,
        model=args.model,
        hook_name=args.hook_name,
        batch_size=args.batch_size,
        l0=args.l0,
        seq_len=args.seq_len,
    )

    model_activations = load_model_activations(args, device)

    atoms = None
    if len(matching_runs) > 0:
        # Use the first matching run directory
        run_dir = matching_runs[0]
        print(f"Found existing run with these parameters: {run_dir}")
        with open(os.path.join(run_dir, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)

        atoms_path = os.path.join(run_dir, "atoms.pt")
        if os.path.exists(atoms_path):
            atoms = torch.load(atoms_path, weights_only=True)

    if atoms is None:
        run_id = str(uuid.uuid4())
        run_dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        metadata = {
            "model": args.model,
            "dataset": args.dataset,
            "hook_name": args.hook_name,
            "batch_size": args.batch_size,
            "l0": args.l0,
            "seq_len": args.seq_len,
            "target_dict_size": args.target_dict_size,
            "device": str(device),
        }
        with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
            yaml.safe_dump(metadata, f)

        wandb.init(project="example_saes", config=metadata)

        train_size = int(model_activations.size(0) * 0.7)
        train_activations = model_activations[:train_size]

        atoms_path = os.path.join(run_dir, "atoms.pt")
        losses_path = os.path.join(run_dir, "losses.pkl")

        atoms, losses = construct_atoms(
            train_activations,
            batch_size=args.batch_size,
            l0=args.l0,
            target_dict_size=args.target_dict_size,
            device=device,
        )
        torch.save(atoms, atoms_path)
        with open(losses_path, "wb") as f:
            pickle.dump(losses, f)

        # Update metadata with number of atoms
        metadata["num_atoms"] = atoms.size(0)
        with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
            yaml.safe_dump(metadata, f)

    # If run was loaded from existing directory, ensure wandb run is initialized
    if wandb.run is None:
        with open(os.path.join(run_dir, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)
        wandb.init(project="example_saes", config=metadata)

    # Evaluate
    test_size = int(model_activations.size(0) * 0.3)
    test_activations = model_activations[-test_size:].to(device)
    ito_sae = ITO_SAE(atoms.to(device), l0=args.l0)
    eval_losses = evaluate(
        ito_sae, test_activations.to(device), batch_size=args.batch_size
    )

    mean_ito_loss = eval_losses.mean().item()
    wandb.log({"eval_loss": mean_ito_loss})

    print(
        "Mean ITO loss:",
        mean_ito_loss,
        "on",
        len(eval_losses),
        "samples",
    )

    wandb.finish()
