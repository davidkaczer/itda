"""
Implementations of inference time optimization (ITO) algorithms for dictionary
learning. Also implements a function for constructing a dictionary of atoms
using these methods, now refactored to pass model name, dataset name, and hook
point explicitly for transformer_lens loading.
"""

import argparse
import gc
import os
import pickle
import uuid

import numpy as np
import torch
import yaml
import zarr
from ito_sae import ITO_SAE
from tqdm import tqdm

import wandb

OMP_BATCH_SIZE = 16
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
    activations: zarr.DirectoryStore,
    atoms: torch.Tensor = None,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    target_dict_size: int = 1024,
    train_size: int = -1,
    device="cpu",
):
    if atoms is None:
        all_rows = []
        for start_idx in tqdm(
            range(0, activations.shape[0], batch_size), desc="Initializing Atoms"
        ):
            end_idx = min(start_idx + batch_size, activations.shape[0])
            chunk = torch.from_numpy(activations[start_idx:end_idx, :2])
            chunk = chunk.flatten(end_dim=1)
            all_rows.append(chunk)

        first_two_pos_acts = torch.cat(all_rows, dim=0).to(device)
        del all_rows

        unique_rows, counts = torch.unique(
            first_two_pos_acts, dim=0, return_counts=True
        )
        _, topk_indices = torch.topk(counts, k=activations.shape[-1])
        atoms = unique_rows[topk_indices]

    if train_size < 0:
        train_size = activations.shape[0]

    total_tokens = activations.shape[0] * activations.shape[1]
    fraction_x = target_dict_size / total_tokens

    if fraction_x <= 0:
        raise ValueError(
            f"Computed fraction_x={fraction_x:.6f} <= 0. "
            "Please provide a larger target_dict_size or fewer initial atoms."
        )

    atoms_per_batch = max(
        1, int(np.ceil(fraction_x * batch_size * activations.shape[1]))
    )

    losses = []
    ratio_history = []
    pbar = tqdm(total=train_size, desc="Constructing Atoms")
    batch_counter = 0

    for start_idx in range(0, train_size, batch_size):
        batch_activations = activations[
            start_idx : min(start_idx + batch_size, train_size)
        ]
        batch_activations = torch.from_numpy(batch_activations).to(device)
        batch_activations = batch_activations.flatten(end_dim=1)

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

        pbar.update(batch_size)
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
    sae: ITO_SAE,
    store: zarr.DirectoryStore,
    batch_size: int = 32,
    val_size: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Evaluate reconstruction loss over the last `val_size` sequences in the Zarr store.
    Returns a 1D tensor with the MSE for each token in that evaluation range.
    """
    zf = zarr.open_group(store=store, mode="r")
    total_sequences = zf["activations"].shape[0]

    # We evaluate on the last `val_size` sequences
    start_idx = max(0, total_sequences - val_size)

    losses = []
    # Iterate over sequences in batches
    for seq_start in tqdm(
        range(start_idx, total_sequences, batch_size), desc="Evaluating"
    ):
        seq_end = min(seq_start + batch_size, total_sequences)
        batch_activations = zf[seq_start:seq_end]  # shape: (B, seq_len, d_model)
        batch_activations = torch.from_numpy(batch_activations).to(device)
        # Flatten from (B, seq_len, d_model) -> (B * seq_len, d_model)
        batch_activations = batch_activations.flatten(end_dim=1)

        with torch.no_grad():
            recon = sae(batch_activations)
            # Per-token MSE
            batch_loss = ((batch_activations - recon) ** 2).mean(dim=1)
            losses.extend(batch_loss.detach().cpu())

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


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available..")
    device = "cuda"

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

    act_path = os.path.join(DATA_DIR, args.model, args.dataset)

    atoms = None
    store = zarr.DirectoryStore(act_path)
    zf = zarr.open_group(store=store, mode="r")
    total_sequences = zf[args.hook_name].shape[0]
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

        train_size = int(total_sequences * 0.7)

        atoms_path = os.path.join(run_dir, "atoms.pt")
        losses_path = os.path.join(run_dir, "losses.pkl")

        atoms, losses = construct_atoms(
            zf[args.hook_name],
            batch_size=args.batch_size,
            l0=args.l0,
            target_dict_size=args.target_dict_size,
            train_size=train_size,
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

    test_size = int(total_sequences * 0.3)
    ito_sae = ITO_SAE(atoms.to(device), l0=args.l0)

    eval_losses = evaluate(
        ito_sae,
        store,
        batch_size=args.batch_size,
        val_size=test_size,
        device=device,
    )

    mean_ito_loss = eval_losses.mean().item()
    wandb.log({"eval_loss": mean_ito_loss})

    print(
        "Mean ITO loss:",
        mean_ito_loss,
        "on",
        len(eval_losses),
        "tokens",
    )

    wandb.finish()
