import argparse
import gc
from collections import Counter
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
    """
    Utility to print out GPU memory usage for debugging if needed.
    """
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


def gather_token_activations(zf, token_indices: torch.Tensor, device="cpu"):
    """
    Given a Zarr array `zf` of shape [num_sequences, seq_len, d_model]
    and a list of token_indices [K, 2] specifying (seq_idx, pos_idx),
    load only the requested activations in a memory-efficient way.

    Preserves the order of `token_indices` in the returned tensor.
    """
    # Zarr chunk shape (e.g. (batch_size, seq_len, d_model))
    chunk_shape = zf.chunks
    zf_shape = zf.shape  # e.g. (num_sequences, seq_len, d_model)

    # Convert token_indices to numpy for easy masking
    # and prepare output storage
    token_indices_np = token_indices.cpu().numpy()
    seq_idx = token_indices_np[:, 0]
    pos_idx = token_indices_np[:, 1]
    K = len(seq_idx)

    # We'll fill this array in the exact order of token_indices
    # shape: [K, d_model]
    d_model = zf_shape[2]
    activations_np = np.empty((K, d_model), dtype=np.float32)

    # Iterate over Zarr in chunk-sized blocks
    for seq_start in tqdm(
        range(0, zf_shape[0], chunk_shape[0]), desc="Gathering activations"
    ):
        seq_end = min(seq_start + chunk_shape[0], zf_shape[0])
        for pos_start in range(0, zf_shape[1], chunk_shape[1]):
            pos_end = min(pos_start + chunk_shape[1], zf_shape[1])

            # Figure out which requested indices fall in this chunk
            in_chunk_mask = (
                (seq_idx >= seq_start)
                & (seq_idx < seq_end)
                & (pos_idx >= pos_start)
                & (pos_idx < pos_end)
            )
            if not np.any(in_chunk_mask):
                continue

            # Load the chunk once
            # shape: [chunk_seq, chunk_pos, d_model]
            chunk = zf[seq_start:seq_end, pos_start:pos_end, :]

            # Offsets inside the chunk
            chunk_indices = np.where(in_chunk_mask)[0]  # row numbers in token_indices
            local_seq_idx = seq_idx[chunk_indices] - seq_start
            local_pos_idx = pos_idx[chunk_indices] - pos_start

            # Gather only the requested positions from this chunk
            for i, row_in_output in enumerate(chunk_indices):
                s = local_seq_idx[i]
                p = local_pos_idx[i]
                activations_np[row_in_output, :] = chunk[s, p, :]

    # Convert to torch tensor on the desired device
    activations = torch.from_numpy(activations_np).to(device)
    return activations


def top_k_most_repeated_rows(tensor, k):
    """
    Finds the top k most repeated rows in a (B, N) tensor and their indices.

    Args:
        tensor: A PyTorch tensor of shape (B, N).
        k: The number of top repeated rows to return.

    Returns:
        A tuple containing:
            - indices: A LongTensor of shape (k,) containing the indices of the top k most repeated rows.
            - rows: A tensor of shape (k, N) containing the top k most repeated rows.
            Returns None if tensor is empty or k is invalid.
    """
    if tensor.numel() == 0 or k <= 0 or k > tensor.shape[0]:
        return None

    tensor_tuples = tuple(tuple(row.tolist()) for row in tensor)
    row_counts = Counter(tensor_tuples)
    top_k_rows_with_counts = row_counts.most_common(k)

    indices = []
    rows = []
    for row_tuple, _ in top_k_rows_with_counts:  # Ignore counts
        index = tensor_tuples.index(row_tuple)
        indices.append(index)
        rows.append(tensor[index])

    return torch.tensor(indices, dtype=torch.long, device=tensor.device), torch.stack(
        rows
    )


def construct_atoms(
    activations: zarr.Array,
    atom_indices=None,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    target_loss_threshold: float = 3.0,
    train_size: int = -1,
    device="cpu",
):
    if atom_indices is None:
        # Get activations from the first position to seed the dictionary
        all_rows = []
        for start_idx in tqdm(
            range(0, activations.shape[0], batch_size), desc="Initializing Atoms"
        ):
            end_idx = min(start_idx + batch_size, activations.shape[0])
            chunk = torch.from_numpy(activations[start_idx:end_idx, :1])
            chunk = chunk.flatten(end_dim=1)
            all_rows.append(chunk)

        first_pos_acts = torch.cat(all_rows, dim=0).to(device)
        del all_rows

        atom_indices, atoms = top_k_most_repeated_rows(
            first_pos_acts, activations.shape[-1]
        )
        atom_indices = torch.cat(
            [
                atom_indices.unsqueeze(-1),
                torch.zeros((atom_indices.size(0), 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )
    else:
        print("loading atoms")
        atoms = gather_token_activations(activations, atom_indices, device=device)

    if train_size < 0:
        train_size = activations.shape[0]

    losses = []
    pbar = tqdm(total=train_size, desc="Constructing Atoms (Loss-Threshold)")

    batch_counter = 0
    for start_idx in range(0, train_size, batch_size):
        end_idx = min(start_idx + batch_size, train_size)
        batch_np = activations[start_idx:end_idx]  # shape [B, seq_len, d_model]
        B, seq_len, d_model = batch_np.shape

        # Flatten: [B*seq_len, d_model]
        batch_activations = torch.from_numpy(batch_np).to(device)
        batch_activations = batch_activations.flatten(end_dim=1)

        # Create token indices for each activation to track their position
        token_indices = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(start_idx, start_idx + B),
                    torch.arange(seq_len),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .to(device)
        )

        # Reconstruct with current dictionary
        sae = ITO_SAE(atoms, l0)
        recon = sae(batch_activations)
        batch_loss = ((batch_activations - recon) ** 2).mean(dim=1)

        # Filter out any extreme numeric explosions (avoid adding nonsense)
        valid_mask = batch_loss < 1e8
        valid_losses = batch_loss[valid_mask]
        valid_activations = batch_activations[valid_mask]
        token_indices = token_indices[valid_mask]
        losses.extend(valid_losses.cpu().tolist())

        # Identify which activations exceed the loss threshold
        new_atoms_mask = valid_losses > target_loss_threshold
        new_atoms = valid_activations[new_atoms_mask]
        new_indices = token_indices[new_atoms_mask]

        # If any new atoms meet the threshold, add them
        if new_atoms.size(0) > 0:
            atoms = torch.cat([atoms, new_atoms], dim=0)
            atom_indices = torch.cat([atom_indices, new_indices], dim=0)

        batch_counter += 1
        pbar.update(end_idx - start_idx)
        pbar.set_postfix(
            {
                "dict_size": atoms.size(0),
                "mean_loss": float(valid_losses.mean().cpu().item()),
            }
        )

        # Optional logging to W&B if desired
        wandb.log(
            {
                "batch_step": batch_counter,
                "mean_loss": float(valid_losses.mean().cpu().item()),
                "batch_new_atoms": new_atoms.size(0),
                "dict_size": atoms.size(0),
            }
        )
    pbar.close()

    # remove duplicate atoms: due to the batching, we may have added the same
    # atom multiple times
    atoms, unique_indices = torch.unique(atoms, return_inverse=True, dim=0)
    atom_indices = atom_indices[unique_indices]

    return atoms, atom_indices, losses


def evaluate(
    sae: ITO_SAE,
    activations,
    batch_size: int = 32,
    val_size: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Evaluate reconstruction loss over the last `val_size` sequences.
    """
    total_sequences = activations.shape[0]
    start_idx = max(0, total_sequences - val_size)

    losses = []
    pbar = tqdm(range(start_idx, total_sequences, batch_size), desc="Evaluating")
    for seq_start in pbar:
        seq_end = min(seq_start + batch_size, total_sequences)
        batch_np = activations[seq_start:seq_end]  # shape: (B, seq_len, d_model)
        batch_acts = torch.from_numpy(batch_np).to(device)
        # Flatten: (B, seq_len, d_model) -> (B * seq_len, d_model)
        batch_acts = batch_acts.flatten(end_dim=1)

        with torch.no_grad():
            recon = sae(batch_acts)
            batch_loss = ((batch_acts - recon) ** 2).mean(dim=1)
            losses.extend(batch_loss.cpu().tolist())

    return torch.tensor(losses, device="cpu")


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available..")
    device = "cuda"

    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path (just for metadata tracking).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NeelNanda/pile-10k",
        help="Which dataset's activations are we using?",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Which layer from the HF hidden states do we train on?",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=OMP_BATCH_SIZE,
        help="Batch size for OMP reconstruction steps.",
    )
    parser.add_argument(
        "--l0",
        type=int,
        default=OMP_L0,
        help="Max number of dictionary atoms to use in each OMP reconstruction.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=SEQ_LEN,
        help="Sequence length used when generating the stored activations.",
    )
    parser.add_argument(
        "--target_loss",
        type=float,
        default=3.0,
        help="If an activation's reconstruction loss is above this threshold, add it as a new atom.",
    )
    parser.add_argument(
        "--load_run_id",
        type=str,
        default=None,
        help=(
            "If specified, continue from that run's token indices. "
            "We'll create a new run, but gather activations from the new model "
            "at those indices and continue dictionary construction."
        ),
    )
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    # Use the layer to identify the Zarr dataset
    act_path = os.path.join(DATA_DIR, args.model, args.dataset)
    if not os.path.exists(act_path):
        raise FileNotFoundError(
            f"No Zarr directory found at {act_path}. Please run get_model_activations first."
        )

    # Open the Zarr group
    store = zarr.DirectoryStore(act_path)
    zf = zarr.open_group(store=store, mode="r")
    layer_name = f"layer_{args.layer}"
    if layer_name not in zf:
        raise ValueError(
            f"Layer dataset '{layer_name}' not found in {act_path}. "
            f"Available layers: {list(zf.array_keys())}"
        )

    # Make a new run
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "batch_size": args.batch_size,
        "l0": args.l0,
        "seq_len": args.seq_len,
        "target_loss": args.target_loss,
        "device": str(device),
    }
    if args.load_run_id is not None:
        metadata["load_run_id"] = args.load_run_id

    wandb.init(project="example_saes", config=metadata)
    run_id = wandb.run.id

    # Load the chosen layer's activations: shape (N, seq_len, d_model)
    layer_acts = zf[layer_name]
    total_sequences = layer_acts.shape[0]

    train_size = int(total_sequences * 0.7)

    atom_indices = None
    if args.load_run_id is not None:
        continue_run_dir = os.path.join(RUNS_DIR, args.load_run_id)
        continue_atom_indices_path = os.path.join(continue_run_dir, "atom_indices.pt")
        # Note: `weights_only=True` is not a valid argument for torch.load, removing it
        atom_indices = torch.load(continue_atom_indices_path)

    # Build dictionary using the threshold-based approach
    atoms, atom_indices, losses = construct_atoms(
        layer_acts,
        atom_indices=atom_indices,
        batch_size=args.batch_size,
        l0=args.l0,
        target_loss_threshold=args.target_loss,
        train_size=train_size,
        device=device,
    )

    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)

    atoms_path = os.path.join(run_dir, "atoms.pt")
    atom_indices_path = os.path.join(run_dir, "atom_indices.pt")
    losses_path = os.path.join(run_dir, "losses.pkl")

    torch.save(atoms, atoms_path)
    torch.save(atom_indices, atom_indices_path)
    with open(losses_path, "wb") as f:
        pickle.dump(losses, f)

    # Update metadata with final # of atoms
    metadata["num_atoms"] = atoms.size(0)
    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)

    # Now evaluate on the last 30% of data
    ito_sae = ITO_SAE(atoms.to(device), l0=args.l0)
    test_size = int(total_sequences * 0.3)
    eval_losses = evaluate(
        ito_sae,
        layer_acts,
        batch_size=args.batch_size,
        val_size=test_size,
        device=device,
    )

    mean_ito_loss = eval_losses.mean().item()
    print(f"Mean ITO loss: {mean_ito_loss:.4f} over {len(eval_losses)} activations")

    # Log final evaluation loss
    wandb.log({"eval_loss": mean_ito_loss})

    artifact = wandb.Artifact(
        name=f"ito_dictionary_{run_id}",
        type="model",
        description="Dictionary atoms and indices for ITO",
        metadata={"num_atoms": atoms.size(0)},
    )
    artifact.add_file(atoms_path)
    artifact.add_file(atom_indices_path)
    artifact.add_file(losses_path)
    artifact.add_file(os.path.join(run_dir, "metadata.yaml"))
    wandb.log_artifact(artifact)

    wandb.finish()
