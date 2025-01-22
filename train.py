import argparse
import gc
import os
import pickle
import uuid

import numpy as np
import torch
import yaml
import zarr
from tqdm import tqdm
import wandb

from ito_sae import ITO_SAE
from optim import omp_incremental_cholesky_with_fallback

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers import TopKTrainer
from dictionary_learning.trainers.top_k import AutoEncoderTopK

OMP_BATCH_SIZE = 16
OMP_L0 = 40

SEQ_LEN = 128
DATASET = "NeelNanda/pile-10k"

# Top-level artifacts directory
ARTIFACTS_DIR = "artifacts"
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data")
RUNS_DIR = os.path.join(ARTIFACTS_DIR, "runs")

# -----------------------------------------
# I. Utility functions
# -----------------------------------------


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
    Load only the requested (seq_idx, pos_idx) activations from a Zarr array.
    The returned tensor matches the order of token_indices.
    """
    chunk_shape = zf.chunks
    zf_shape = zf.shape
    token_indices_np = token_indices.cpu().numpy()
    seq_idx = token_indices_np[:, 0]
    pos_idx = token_indices_np[:, 1]
    K = len(seq_idx)
    d_model = zf_shape[2]
    activations_np = np.empty((K, d_model), dtype=np.float32)

    for seq_start in tqdm(range(0, zf_shape[0], chunk_shape[0]), desc="Gathering acts"):
        seq_end = min(seq_start + chunk_shape[0], zf_shape[0])
        for pos_start in range(0, zf_shape[1], chunk_shape[1]):
            pos_end = min(pos_start + chunk_shape[1], zf_shape[1])
            in_chunk_mask = (
                (seq_idx >= seq_start)
                & (seq_idx < seq_end)
                & (pos_idx >= pos_start)
                & (pos_idx < pos_end)
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

    return torch.from_numpy(activations_np).to(device)


# -----------------------------------------
# II. ITO-SAE training code
# -----------------------------------------


def construct_atoms(
    activations,
    atom_indices=None,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    target_loss_threshold: float = 3.0,
    train_size: int = -1,
    device="cpu",
    has_bos=False,
    sklearn=False,
):
    """
    The dictionary-building procedure used for ITO SAEs.
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    from collections import Counter

    def top_k_most_repeated_rows(tensor, k):
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
        return torch.tensor(
            indices, dtype=torch.long, device=tensor.device
        ), torch.stack(rows)

    def stable_unique(tensor, indices):
        tensor_np = tensor.cpu().numpy()
        indices_np = indices.cpu().numpy()
        _, unique_row_idx = np.unique(tensor_np, axis=0, return_index=True)
        unique_row_idx = np.sort(unique_row_idx)
        unique_tensor_np = tensor_np[unique_row_idx]
        unique_indices_np = indices_np[unique_row_idx]
        unique_tensor = torch.from_numpy(unique_tensor_np).to(tensor.device)
        unique_indices = torch.from_numpy(unique_indices_np).to(indices.device)
        return unique_tensor, unique_indices

    if train_size < 0:
        train_size = activations.shape[0]

    num_sequences = activations.shape[0]
    seq_len = activations.shape[1]
    d_model = activations.shape[-1]

    # (1) Initialization
    if atom_indices is None:
        all_rows = []
        all_token_indices = []
        for start_idx in tqdm(range(0, num_sequences, batch_size), desc="Initializing"):
            end_idx = min(start_idx + batch_size, num_sequences)
            # If has_bos, we take first 2 tokens, else just 1
            if has_bos:
                chunk_np = activations[start_idx:end_idx, :2]
            else:
                chunk_np = activations[start_idx:end_idx, :1]
            chunk = torch.from_numpy(chunk_np).to(device).flatten(end_dim=1)
            B = end_idx - start_idx
            seq_len_slice = chunk_np.shape[1]
            idx_grid = torch.stack(
                torch.meshgrid(
                    torch.arange(start_idx, end_idx, device=device),
                    torch.arange(seq_len_slice, device=device),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            all_rows.append(chunk)
            all_token_indices.append(idx_grid)
        first_pos_acts = torch.cat(all_rows, dim=0)
        first_pos_indices = torch.cat(all_token_indices, dim=0)
        topk_ret = top_k_most_repeated_rows(first_pos_acts, d_model)
        if topk_ret is None:
            # fallback: just pick the first d_model rows
            init_atoms = first_pos_acts[:d_model]
            init_indices = first_pos_indices[:d_model]
        else:
            topk_indices, topk_atoms = topk_ret
            init_atoms = topk_atoms
            init_indices = first_pos_indices[topk_indices]
        atoms = init_atoms
        atom_indices = init_indices
    else:
        # gather atoms from provided atom_indices
        atoms = gather_token_activations(activations, atom_indices, device=device)

    print(f"Initialized with {atoms.size(0)} atoms.")
    losses = []
    pbar = tqdm(total=train_size, desc="Constructing Atoms")
    # (2) dictionary construction
    batch_counter = 0
    for start_idx in range(0, train_size, batch_size):
        end_idx = min(start_idx + batch_size, train_size)
        batch_np = activations[start_idx:end_idx]
        batch_activations = torch.from_numpy(batch_np).to(device).flatten(end_dim=1)
        token_indices = torch.stack(
            torch.meshgrid(
                torch.arange(start_idx, end_idx, device=device),
                torch.arange(seq_len, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        sae = ITO_SAE(atoms, l0, sklearn=sklearn)
        recon = sae(batch_activations)
        batch_loss = ((batch_activations - recon) ** 2).mean(dim=1)
        valid_mask = batch_loss < 1e8
        valid_losses = batch_loss[valid_mask]
        valid_activations = batch_activations[valid_mask]
        valid_indices = token_indices[valid_mask]
        losses.extend(valid_losses.cpu().tolist())

        # add new atoms if needed
        new_atoms_mask = valid_losses > target_loss_threshold
        new_atoms = valid_activations[new_atoms_mask]
        new_atom_indices = valid_indices[new_atoms_mask]

        if new_atoms.size(0) > 0:
            atoms = torch.cat([atoms, new_atoms], dim=0)
            atom_indices = torch.cat([atom_indices, new_atom_indices], dim=0)

        batch_counter += 1
        pbar.update(end_idx - start_idx)
        pbar.set_postfix(
            {
                "dict_size": atoms.size(0),
                "mean_loss": float(valid_losses.mean().item()),
            }
        )
        wandb.log(
            {
                "batch_step": batch_counter,
                "mean_loss": float(valid_losses.mean().item()),
                "batch_new_atoms": new_atoms.size(0),
                "dict_size": atoms.size(0),
            }
        )
    pbar.close()

    # (3) deduplicate
    atoms, atom_indices = stable_unique(atoms, atom_indices)
    print(f"Final dictionary size after dedup: {atoms.size(0)} atoms.")
    return atoms, atom_indices, losses


def evaluate_ito(
    sae: ITO_SAE,
    activations,
    batch_size: int = 32,
    val_size: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    total_sequences = activations.shape[0]
    start_idx = max(0, total_sequences - val_size)
    losses = []
    pbar = tqdm(range(start_idx, total_sequences, batch_size), desc="Evaluating ITO")
    for seq_start in pbar:
        seq_end = min(seq_start + batch_size, total_sequences)
        batch_np = activations[seq_start:seq_end]
        batch_acts = torch.from_numpy(batch_np).to(device).flatten(end_dim=1)
        with torch.no_grad():
            recon = sae(batch_acts)
            batch_loss = ((batch_acts - recon) ** 2).mean(dim=1)
            losses.extend(batch_loss.cpu().tolist())
    return torch.tensor(losses, device="cpu")


def train_ito_saes(args, device):
    """
    Train an ITO-SAE dictionary using your original threshold-based approach.
    """
    run_id = wandb.run.id
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    torch.set_grad_enabled(False)  # no grads needed for ito

    # Zarr
    act_path = os.path.join(DATA_DIR, args.model, args.dataset)
    store = zarr.DirectoryStore(act_path)
    zf = zarr.open_group(store=store, mode="r")
    layer_name = f"layer_{args.layer}"
    if layer_name not in zf:
        raise ValueError(f"Layer dataset '{layer_name}' not found in {act_path}.")
    layer_acts = zf[layer_name]

    # If we have a --max_sequences argument, limit how many sequences we load:
    if args.max_sequences is not None:
        layer_acts = layer_acts[: args.max_sequences]

    total_sequences = layer_acts.shape[0]
    train_size = int(total_sequences * 0.7)
    print(f"Total sequences = {total_sequences}, train_size = {train_size}")

    # Possibly load previous atom_indices if resuming from run_id
    atom_indices = None
    if args.load_run_id is not None:
        continue_run_dir = os.path.join(RUNS_DIR, args.load_run_id)
        continue_atom_indices_path = os.path.join(continue_run_dir, "atom_indices.pt")
        atom_indices = torch.load(continue_atom_indices_path)

    # Construct dictionary
    atoms, atom_indices, losses = construct_atoms(
        layer_acts,
        atom_indices=atom_indices,
        batch_size=args.batch_size,
        l0=args.l0,
        target_loss_threshold=args.target_loss,
        train_size=train_size,
        device=device,
        has_bos=("llama" in args.model.lower()),
        sklearn=args.sklearn,  # Pass sklearn arg
    )

    # Save results
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "batch_size": args.batch_size,
        "l0": args.l0,
        "seq_len": args.seq_len,
        "target_loss": args.target_loss,
        "device": str(device),
        "num_atoms": atoms.size(0),
        "method": "ito",
    }
    if args.load_run_id is not None:
        metadata["load_run_id"] = args.load_run_id

    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)

    atoms_path = os.path.join(run_dir, "atoms.pt")
    atom_indices_path = os.path.join(run_dir, "atom_indices.pt")
    losses_path = os.path.join(run_dir, "losses.pkl")

    torch.save(atoms, atoms_path)
    torch.save(atom_indices, atom_indices_path)
    with open(losses_path, "wb") as f:
        pickle.dump(losses, f)

    # Evaluate on last 30%
    # Pass sklearn arg during final initialization
    ito_sae = ITO_SAE(atoms.to(device), l0=args.l0, sklearn=args.sklearn)
    test_size = int(total_sequences * 0.3)
    eval_losses = evaluate_ito(
        ito_sae,
        layer_acts,
        batch_size=args.batch_size,
        val_size=test_size,
        device=device,
    )
    mean_ito_loss = eval_losses.mean().item()
    print(f"Mean ITO loss: {mean_ito_loss:.4f} over {len(eval_losses)} activations")
    wandb.log({"eval_loss": mean_ito_loss})

    # W&B artifact
    artifact = wandb.Artifact(
        name=f"ito_dictionary_{run_id}",
        type="model",
        description="Dictionary atoms and indices for ITO-SAE",
        metadata={"num_atoms": atoms.size(0)},
    )
    artifact.add_file(atoms_path)
    artifact.add_file(atom_indices_path)
    artifact.add_file(losses_path)
    artifact.add_file(os.path.join(run_dir, "metadata.yaml"))
    wandb.log_artifact(artifact)


# -----------------------------------------
# III. Dictionary-learning-based training
# -----------------------------------------


class ZarrActivationDataset(torch.utils.data.Dataset):
    """
    Loads the full Zarr activations into memory once, then
    returns rows from the in-memory tensor. This avoids slow
    on-the-fly Zarr lookups for each sample.

    We assume that the Zarr array has shape (N, seq_len, d_model).
    We'll flatten it to (N * seq_len, d_model).
    """

    def __init__(self, zarr_array):
        super().__init__()
        # Read entire array into memory (a NumPy array)
        full_np = zarr_array[:]  # shape (N, seq_len, d_model)

        # Convert to a float32 torch.Tensor on CPU
        self.all_activations = torch.from_numpy(full_np).float()

        # Flatten from (N, seq_len, d_model) to (N * seq_len, d_model)
        self.all_activations = self.all_activations.view(
            -1, self.all_activations.size(-1)
        )

        # Keep track of shape
        self.num_sequences_times_seq_len = self.all_activations.size(0)
        self.d_model = self.all_activations.size(1)

    def __len__(self):
        return self.num_sequences_times_seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a single row of shape (d_model,).
        """
        return self.all_activations[idx]


def collate_fn(batch):
    """
    Collate a list of [d_model]-shaped Tensors into a single [B, d_model] Tensor.
    """
    return torch.stack(batch, dim=0)


def train_dictlearn_saes(args, device):
    """
    Train a dictionary-learning–based SAE (e.g. StandardTrainer + AutoEncoder).
    This reuses your existing Zarr activation store as the dataset.
    """
    if (AutoEncoderTopK is None) or (trainSAE is None) or (TopKTrainer is None):
        raise ImportError(
            "dictionary_learning not found or not importable. "
            "Please install and retry."
        )

    # Create run dir
    run_id = wandb.run.id
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Open Zarr
    act_path = os.path.join(DATA_DIR, args.model, args.dataset)
    store = zarr.DirectoryStore(act_path)
    zf = zarr.open_group(store=store, mode="r")
    layer_name = f"layer_{args.layer}"
    if layer_name not in zf:
        raise ValueError(f"Layer dataset '{layer_name}' not found in {act_path}.")

    layer_acts = zf[layer_name]  # shape (N, seq_len, d_model)

    # If we have a --max_sequences argument, limit how many sequences we load:
    if args.max_sequences is not None:
        layer_acts = layer_acts[: args.max_sequences]

    num_sequences, seq_len, d_model = layer_acts.shape
    print(f"Dictionary-learning SAE on shape = ({num_sequences}, {seq_len}, {d_model})")

    # Create a PyTorch Dataset/Dataloader
    dataset = ZarrActivationDataset(layer_acts)
    # You can define your own sampler or random subset. E.g., 70% for train:
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Configure the trainer (using TopKTrainer as an example)
    trainer_cfg = {
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": d_model,
        "dict_size": args.dict_size,
        "lr": args.lr,
        "device": device,
        "steps": len(train_loader),
        "layer": args.layer,
        "lm_name": args.model,
        "warmup_steps": 0,
        "k": args.l0,
    }
    wandb.log({"dict_size": args.dict_size})

    # Train
    trainers = trainSAE(
        data=train_loader,
        trainer_configs=[trainer_cfg],
        steps=len(train_loader),
        device=device,
        save_dir=run_dir,
    )
    final_ae = trainers[0].ae
    pickle.dump(final_ae, open(os.path.join(run_dir, "ae.pt"), "wb"))

    # Write out a small metadata file
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "method": "dictlearn",
        "dict_size": args.dict_size,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "activation_dim": d_model,
    }
    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)

    # Log artifact to wandb
    artifact = wandb.Artifact(
        name=f"dictionary_learning_{run_id}",
        type="model",
        description="Dictionary-learning-based SAE",
        metadata={"dict_size": args.dict_size, "activation_dim": d_model},
    )
    artifact.add_file(os.path.join(run_dir, "trainer_0", "ae.pt"))
    artifact.add_file(os.path.join(run_dir, "trainer_0", "config.json"))
    artifact.add_file(os.path.join(run_dir, "metadata.yaml"))
    wandb.log_artifact(artifact)

    # Optional: Evaluate on the test set
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    final_ae.to(device)
    final_ae.eval()
    all_losses = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="DictLearn Eval"):
            batch = batch.to(device)  # shape [B, d_model]
            recon, feats = final_ae(batch, output_features=True)
            loss = torch.mean((batch - recon) ** 2, dim=1)
            all_losses.extend(loss.cpu().numpy())

    test_mse = float(np.mean(all_losses))
    wandb.log({"test_mse": test_mse})
    print(f"[DictLearning] Test MSE: {test_mse:.4f}")


# -----------------------------------------
# IV. Main
# -----------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="ito",
        choices=["ito", "dictlearn"],
        help="Which approach to train: 'ito' or 'dictlearn'",
    )

    # Shared arguments
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Optionally limit the total number of sequences to load from the dataset.",
    )

    # ITO-specific arguments
    parser.add_argument(
        "--l0",
        type=int,
        default=OMP_L0,
        help="Max # of dictionary atoms in each OMP reconstruction.",
    )
    parser.add_argument(
        "--target_loss",
        type=float,
        default=3.0,
        help="If reconstruction loss > this threshold, add a new atom.",
    )
    parser.add_argument(
        "--load_run_id",
        type=str,
        default=None,
        help="If specified, continue from that run's token indices for ITO.",
    )

    # Dictionary-learning-specific arguments
    parser.add_argument(
        "--dict_size",
        type=int,
        default=65536,
        help="For dictionary_learning, how many features to learn.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for dictionary_learning approach.",
    )

    parser.add_argument(
        "--sklearn",
        action="store_true",
        help="If set, use the scikit-learn solver in ITO_SAE.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Running on CPU may be slow.")

    wandb.init(project="example_saes", config=vars(args))

    if args.method == "ito":
        train_ito_saes(args, device)
    else:
        train_dictlearn_saes(args, device)

    wandb.finish()
