import argparse
import gc
import os
import pickle
import uuid
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr
import wandb
from tqdm import tqdm

from ito_sae import ITO_SAE

ARTIFACTS_DIR = "artifacts"
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data")
RUNS_DIR = os.path.join(ARTIFACTS_DIR, "runs")

OMP_BATCH_SIZE = 16
OMP_L0 = 40
SEQ_LEN = 128
DATASET = "NeelNanda/pile-10k"

def gather_token_activations(zf, token_indices: torch.Tensor, device="cpu"):
    """Load only the requested activations from a Zarr array in a memory-efficient way."""
    chunk_shape = zf.chunks
    zf_shape = zf.shape
    token_indices_np = token_indices.cpu().numpy()
    seq_idx = token_indices_np[:, 0]
    pos_idx = token_indices_np[:, 1]
    K = len(seq_idx)
    d_model = zf_shape[2]
    activations_np = np.empty((K, d_model), dtype=np.float32)

    for seq_start in tqdm(
        range(0, zf_shape[0], chunk_shape[0]), desc="Gathering activations"
    ):
        seq_end = min(seq_start + chunk_shape[0], zf_shape[0])
        for pos_start in range(0, zf_shape[1], chunk_shape[1]):
            pos_end = min(pos_start + chunk_shape[1], zf_shape[1])
            in_chunk_mask = (
                (seq_idx >= seq_start) & (seq_idx < seq_end) &
                (pos_idx >= pos_start) & (pos_idx < pos_end)
            )
            if not np.any(in_chunk_mask):
                continue
            chunk = zf[seq_start:seq_end, pos_start:pos_end, :]
            chunk_indices = np.where(in_chunk_mask)[0]
            local_seq_idx = seq_idx[chunk_indices] - seq_start
            local_pos_idx = pos_idx[chunk_indices] - pos_start

            for i, row_in_output in enumerate(chunk_indices):
                s = local_seq_idx[i]
                p = local_pos_idx[i]
                activations_np[row_in_output, :] = chunk[s, p, :]

    activations = torch.from_numpy(activations_np).to(device)
    return activations

def top_k_most_repeated_rows(tensor, k):
    """Find the top k most repeated rows in a (B, N) tensor and their indices."""
    from collections import Counter
    if tensor.numel() == 0 or k <= 0 or k > tensor.shape[0]:
        return None
    tensor_tuples = tuple(tuple(row.tolist()) for row in tensor)
    row_counts = Counter(tensor_tuples)
    top_k_rows_with_counts = row_counts.most_common(k)

    indices = []
    rows = []
    for row_tuple, _ in top_k_rows_with_counts:
        index = tensor_tuples.index(row_tuple)
        indices.append(index)
        rows.append(tensor[index])
    return torch.tensor(indices, dtype=torch.long, device=tensor.device), torch.stack(rows)

def train_dictionary_once(
    activations: zarr.Array,
    R: int,
    N: int,
    l0: int = 40,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Single-pass approach:
    1) Randomly sample R activations as initial dictionary.
    2) For each batch, do OMP, track usage counts.
    3) Keep top N atoms by usage.
    """
    num_sequences, seq_len, d_model = activations.shape
    total_tokens = num_sequences * seq_len
    chosen_indices = np.random.choice(total_tokens, size=R, replace=False)
    chosen_seq_idx = chosen_indices // seq_len
    chosen_pos_idx = chosen_indices % seq_len

    token_indices_np = np.column_stack((chosen_seq_idx, chosen_pos_idx))
    atom_indices = torch.from_numpy(token_indices_np).long().to(device)
    atoms = gather_token_activations(activations, atom_indices, device=device)

    usage_counts = torch.zeros(R, dtype=torch.long, device=device)
    sae = ITO_SAE(atoms, l0=l0)

    pbar = tqdm(range(0, num_sequences, batch_size), desc="Decomposing (single pass)")
    for seq_start in pbar:
        seq_end = min(seq_start + batch_size, num_sequences)
        batch_np = activations[seq_start:seq_end]
        B, T, _ = batch_np.shape
        batch_acts = torch.from_numpy(batch_np).to(device).view(-1, d_model)

        codes = sae.encode(batch_acts)
        nonzero_mask = (codes != 0.0)
        batch_usage = nonzero_mask.sum(dim=0)
        usage_counts += batch_usage.long()

    sorted_indices = torch.argsort(usage_counts, descending=True)
    top_indices = sorted_indices[:N]
    final_atoms = atoms[top_indices]
    final_atom_indices = atom_indices[top_indices]
    final_usage_counts = usage_counts[top_indices]

    return final_atoms, final_atom_indices, final_usage_counts

def construct_atoms(
    activations: zarr.Array,
    atom_indices=None,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    target_loss_threshold: float = 3.0,
    target_dict_size: int = 0,
    train_size: int = -1,
    device="cpu",
    residual_atoms: bool = False,
):
    """
    Dictionary construction with two modes:
      1) Fixed threshold: Add atoms if recon loss > target_loss_threshold.
      2) Top fraction (if target_dict_size > 0):
         fraction = target_dict_size / (train_size * seq_len),
         and in each batch select top fraction of tokens by reconstruction loss.

    If residual_atoms=True, we add (activation - reconstruction) as new atoms
    rather than the raw activations themselves.

    This updated version processes tokens in order of token index: all position-0
    tokens from all sequences first, then position-1 tokens from all sequences, and so on.
    """
    seq_len = activations.shape[1]
    d_model = activations.shape[2]
    batch_size = batch_size * seq_len

    if train_size < 0:
        train_size = activations.shape[0]

    total_tokens = train_size * seq_len

    # Pre-initialize from existing indices or from top-k in first positions
    if atom_indices is None:
        # Collect all position=0 activations to find the most repeated rows
        all_rows = []
        for start_idx in tqdm(range(0, train_size, batch_size), desc="Initializing Atoms"):
            end_idx = min(start_idx + batch_size, train_size)
            # shape [B, 1, d_model] for pos=0
            chunk = torch.from_numpy(
                activations[start_idx:end_idx, :1]
            )  # shape [B, 1, d_model]
            chunk = chunk.flatten(end_dim=1)  # shape [B, d_model]
            all_rows.append(chunk)

        first_pos_acts = torch.cat(all_rows, dim=0).to(device)
        del all_rows

        topk = min(d_model, first_pos_acts.shape[0])
        result = top_k_most_repeated_rows(first_pos_acts, topk)
        if result is not None:
            # result = (indices, rows)
            init_atom_rel_idx, init_atoms = result
            # We only have sequence indices relative to first_pos_acts, so build full (seq, pos)
            # For repeated rows, we can't directly recover which sequence they came from,
            # so we store dummy indices: [-1, -1] or keep them separate.
            # Alternatively, you could skip storing indices for duplicates.
            atom_indices = torch.cat(
                [
                    init_atom_rel_idx.unsqueeze(-1),
                    torch.zeros((init_atom_rel_idx.size(0), 1), dtype=torch.long, device=device),
                ],
                dim=1,
            )
            atoms = init_atoms
        else:
            # In case there are no valid rows (extreme edge case)
            atoms = torch.empty((0, d_model), device=device)
            atom_indices = torch.empty((0, 2), dtype=torch.long, device=device)
    else:
        # Load from existing indices
        atoms = gather_token_activations(activations, atom_indices, device=device)

    losses = []
    fraction = 0.0
    if target_dict_size > 0:
        fraction = float(target_dict_size) / float(total_tokens)

    # We'll have a single progress bar for all positions
    pbar = tqdm(total=train_size * seq_len, desc="Constructing Atoms")
    batch_counter = 0

    # Process tokens in order of token index
    for pos_idx in range(seq_len):
        # pos_desc = f"Constructing Atoms (pos={pos_idx})"
        for start_idx in range(0, train_size, batch_size):
            end_idx = min(start_idx + batch_size, train_size)

            # Gather the [start_idx:end_idx, pos_idx:pos_idx+1] slice
            # shape will be [B, 1, d_model]
            batch_np = activations[start_idx:end_idx, pos_idx : pos_idx + 1]
            B, _, _ = batch_np.shape
            batch_activations = torch.from_numpy(batch_np).to(device).view(B, d_model)

            # Create matching token_indices
            seq_indices = torch.arange(start_idx, end_idx, device=device)
            pos_indices = torch.full((B,), pos_idx, device=device)
            token_indices_this_batch = torch.stack((seq_indices, pos_indices), dim=-1)

            if atoms.size(0) > 0:
                sae = ITO_SAE(atoms, l0)
                recon = sae(batch_activations)
            else:
                # If we have no atoms yet, reconstruction is just zero
                recon = torch.zeros_like(batch_activations, device=device)

            batch_loss = ((batch_activations - recon) ** 2).mean(dim=1)

            # Filter out extremely large or invalid losses
            valid_mask = batch_loss < 1e8
            valid_losses = batch_loss[valid_mask]
            valid_activations = batch_activations[valid_mask]
            valid_recon = recon[valid_mask]
            valid_token_indices = token_indices_this_batch[valid_mask]
            losses.extend(valid_losses.cpu().tolist())

            # Decide which new atoms to add
            if target_dict_size > 0:
                # Top fraction approach
                top_count = int(np.ceil(fraction * valid_losses.size(0)))
                if top_count > 0:
                    top_count = min(top_count, valid_losses.size(0))
                    _, selected_idx = torch.topk(valid_losses, k=top_count)
                    if residual_atoms:
                        new_atoms = (valid_activations - valid_recon)[selected_idx]
                    else:
                        new_atoms = valid_activations[selected_idx]
                    new_indices = valid_token_indices[selected_idx]
                else:
                    new_atoms = torch.empty((0, d_model), device=device)
                    new_indices = torch.empty((0, 2), dtype=torch.long, device=device)
            else:
                # Threshold approach
                new_mask = valid_losses > target_loss_threshold
                if residual_atoms:
                    new_atoms = (valid_activations - valid_recon)[new_mask]
                else:
                    new_atoms = valid_activations[new_mask]
                new_indices = valid_token_indices[new_mask]

            # If we have new atoms, concatenate them
            if new_atoms.size(0) > 0:
                atoms = torch.cat([atoms, new_atoms], dim=0)
                atom_indices = torch.cat([atom_indices, new_indices], dim=0)

            batch_counter += 1
            pbar.update(end_idx - start_idx)
            pbar.set_postfix(
                {
                    "dict_size": atoms.size(0),
                    "pos": pos_idx,
                    "mean_loss": float(valid_losses.mean().cpu().item())
                    if valid_losses.numel() > 0
                    else 0.0,
                }
            )

            wandb.log(
                {
                    "batch_step": batch_counter,
                    "mean_loss": float(valid_losses.mean().cpu().item())
                    if valid_losses.numel() > 0
                    else 0.0,
                    "batch_new_atoms": new_atoms.size(0),
                    "dict_size": atoms.size(0),
                    "pos": pos_idx,
                }
            )

    pbar.close()

    # Remove exact duplicates in the final set of atoms
    # (This does not remove near-duplicates; you can do a cos-sim pass if you want.)
    # unique_indices is the first occurrence index for each unique row in `atoms`
    atoms, unique_indices = torch.unique(atoms, return_inverse=True, dim=0)
    atom_indices = atom_indices[unique_indices]

    return atoms, atom_indices, losses


def deduplicate_atoms(
    atoms: torch.Tensor, atom_indices: torch.Tensor, cos_threshold=0.7
):
    """Remove atoms whose cosine similarity is > cos_threshold to a prior atom."""
    device = atoms.device
    atoms_normed = F.normalize(atoms, dim=1)
    N = atoms_normed.size(0)
    sim_matrix = atoms_normed @ atoms_normed.T
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)

    for i in range(N):
        if not keep_mask[i]:
            continue
        row_sims = sim_matrix[i, i + 1 :]
        dup_candidates = (row_sims > cos_threshold).nonzero()
        for dc in dup_candidates:
            j = i + 1 + dc.item()
            keep_mask[j] = False

    return atoms[keep_mask], atom_indices[keep_mask]

def construct_atoms_cosine_threshold(
    activations,
    atom_indices=None,
    batch_size=16,
    train_size=-1,
    device="cpu",
    cos_threshold=0.7,
):
    """Dictionary creation by cosine similarity threshold."""
    d_model = activations.shape[-1]
    if train_size < 0:
        train_size = activations.shape[0]

    if atom_indices is None or len(atom_indices) == 0:
        atoms = torch.empty((0, d_model), dtype=torch.float32, device=device)
        atom_indices = torch.empty((0, 2), dtype=torch.long, device=device)
    else:
        atoms = gather_token_activations(activations, atom_indices, device=device)

    all_losses = []

    def get_normalized_dict():
        return F.normalize(atoms, dim=1) if atoms.size(0) > 0 else None

    dict_normed = get_normalized_dict()
    pbar = tqdm(total=train_size, desc="Cosine-Threshold Dictionary Construction")

    for step, seq_start in enumerate(range(0, train_size, batch_size)):
        seq_end = min(seq_start + batch_size, train_size)
        batch_np = activations[seq_start:seq_end]
        B, seq_len, _ = batch_np.shape
        batch_activations = torch.from_numpy(batch_np).to(device).view(-1, d_model)
        token_indices = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(seq_start, seq_end),
                    torch.arange(seq_len),
                    indexing="ij",
                ),
                dim=-1,
            )
            .reshape(-1, 2)
            .to(device)
        )

        if batch_activations.size(0) == 0:
            pbar.update(seq_end - seq_start)
            continue

        if dict_normed is not None and dict_normed.size(0) > 0:
            batch_normed = F.normalize(batch_activations, dim=1)
            sims = dict_normed @ batch_normed.T
            max_sim_with_dict, _ = sims.max(dim=0)
        else:
            max_sim_with_dict = torch.full(
                (batch_activations.size(0),), -1.0, dtype=torch.float32, device=device
            )

        keep_mask = max_sim_with_dict < cos_threshold
        new_candidates = batch_activations[keep_mask]
        new_candidate_indices = token_indices[keep_mask]

        if new_candidates.size(0) > 0:
            keep_mask_new = deduplicate_batch_candidates(new_candidates, cos_threshold)
            new_candidates = new_candidates[keep_mask_new]
            new_candidate_indices = new_candidate_indices[keep_mask_new]
            atoms = torch.cat([atoms, new_candidates], dim=0)
            atom_indices = torch.cat([atom_indices, new_candidate_indices], dim=0)
            dict_normed = get_normalized_dict()

        wandb.log(
            {
                "batch_step": step,
                "batch_new_atoms": new_candidates.size(0),
                "dict_size": atoms.size(0),
            }
        )
        pbar.update(seq_end - seq_start)
        pbar.set_postfix({"dict_size": atoms.size(0)})

    pbar.close()
    atoms, atom_indices = deduplicate_atoms(atoms, atom_indices, cos_threshold=cos_threshold)
    return atoms, atom_indices, all_losses

def deduplicate_batch_candidates(candidates: torch.Tensor, cos_threshold=0.7) -> torch.Tensor:
    """Deduplicate new candidates so no two are above cos_threshold in similarity."""
    device = candidates.device
    x_norm = F.normalize(candidates, dim=1)
    N = x_norm.size(0)
    sim_matrix = x_norm @ x_norm.T
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    for i in range(N):
        if not keep_mask[i]:
            continue
        row_sims = sim_matrix[i, i + 1 :]
        dup_candidates = (row_sims > cos_threshold).nonzero()
        for dc in dup_candidates:
            j = i + 1 + dc.item()
            keep_mask[j] = False
    return keep_mask

if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = "cuda"
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=OMP_BATCH_SIZE)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--target_loss", type=float, default=3.0)
    parser.add_argument("--load_run_id", type=str, default=None)
    parser.add_argument("--random_activation_atoms", type=int, default=0)
    parser.add_argument("--random_atoms", type=int, default=0)
    parser.add_argument("--cos_threshold", type=float, default=0.0)
    parser.add_argument("--train_once_r", type=int, default=0)
    parser.add_argument("--train_once_n", type=int, default=0)
    parser.add_argument(
        "--target_dict_size", 
        type=int, 
        default=0,
        help="Approximate desired dictionary size. If > 0, we select the top fraction per batch."
    )
    # New flag to add residuals instead of raw activations in construct_atoms
    parser.add_argument(
        "--residual_atoms",
        action="store_true",
        default=False,
        help="If set, add residual (activation - reconstruction) instead of the raw activation."
    )
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)

    act_path = os.path.join(DATA_DIR, args.model, args.dataset)
    if not os.path.exists(act_path):
        raise FileNotFoundError(f"No Zarr directory found at {act_path}.")

    store = zarr.DirectoryStore(act_path)
    zf = zarr.open_group(store=store, mode="r")
    layer_name = f"layer_{args.layer}"
    if layer_name not in zf:
        raise ValueError(f"Layer dataset '{layer_name}' not found in {act_path}.")

    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "batch_size": args.batch_size,
        "l0": args.l0,
        "seq_len": args.seq_len,
        "target_loss": args.target_loss,
        "device": str(device),
        "random_activation_atoms": args.random_activation_atoms,
        "random_atoms": args.random_atoms,
        "cos_threshold": args.cos_threshold,
        "target_dict_size": args.target_dict_size,
        "residual_atoms": args.residual_atoms
    }
    if args.load_run_id is not None:
        metadata["load_run_id"] = args.load_run_id

    wandb.init(project="example_saes", config=metadata)
    run_id = wandb.run.id

    layer_acts = zf[layer_name]
    total_sequences = layer_acts.shape[0]

    atom_indices = None
    if args.load_run_id is not None:
        continue_run_dir = os.path.join(RUNS_DIR, args.load_run_id)
        continue_atom_indices_path = os.path.join(continue_run_dir, "atom_indices.pt")
        atom_indices = torch.load(continue_atom_indices_path)

    atoms = None
    losses = []

    # Single-pass approach
    if args.train_once_r > 0 and args.train_once_n > 0:
        final_atoms, final_atom_indices, usage_counts = train_dictionary_once(
            layer_acts,
            R=args.train_once_r,
            N=args.train_once_n,
            l0=args.l0,
            batch_size=args.batch_size,
            device=device,
        )
        atoms = final_atoms
        atom_indices = final_atom_indices
        losses = []

    elif args.random_activation_atoms > 0:
        num_random = args.random_activation_atoms
        total_tokens = layer_acts.shape[0] * layer_acts.shape[1]
        chosen_indices = np.random.choice(total_tokens, size=num_random, replace=False)
        chosen_seq_idx = chosen_indices // layer_acts.shape[1]
        chosen_pos_idx = chosen_indices % layer_acts.shape[1]
        token_indices_np = np.column_stack((chosen_seq_idx, chosen_pos_idx))
        atom_indices = torch.from_numpy(token_indices_np).long().to(device)
        atoms = gather_token_activations(layer_acts, atom_indices, device=device)

    elif args.random_atoms > 0:
        d_model = layer_acts.shape[-1]
        num_random = args.random_atoms
        atoms = torch.randn(num_random, d_model, device=device)
        atom_indices = -1 * torch.ones((num_random, 2), dtype=torch.long, device=device)

    elif args.cos_threshold > 0:
        # No residual logic here, as requested
        atoms, atom_indices, losses = construct_atoms_cosine_threshold(
            layer_acts,
            atom_indices=atom_indices,
            batch_size=args.batch_size,
            train_size=total_sequences,
            device=device,
            cos_threshold=args.cos_threshold,
        )
    else:
        # This is where residual_atoms can be used
        atoms, atom_indices, losses = construct_atoms(
            layer_acts,
            atom_indices=atom_indices,
            batch_size=args.batch_size,
            l0=args.l0,
            target_loss_threshold=args.target_loss,
            target_dict_size=args.target_dict_size,
            train_size=total_sequences,
            device=device,
            residual_atoms=args.residual_atoms,
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

    metadata["num_atoms"] = atoms.size(0)
    wandb.log({"dict_size": atoms.size(0)})
    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)

    ito_sae = ITO_SAE(atoms.to(device), l0=args.l0)
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
