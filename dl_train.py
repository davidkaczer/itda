import argparse
import random
import string
import os
import uuid
from queue import Empty
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import wandb


def new_wandb_process(id, config, log_queue, entity, project, tags=None):
    """
    Spawns a W&B run for a single layer.
    Continuously pulls logs from `log_queue` until it sees 'DONE'.
    Handles 'artifact' messages to upload atoms/atom_indices/metadata.
    """
    wandb.init(
        id=id,
        entity=entity,
        project=project,
        config=config,
        name=config["wandb_name"],
        tags=tags,
    )
    while True:
        try:
            log = log_queue.get(timeout=1)
        except Empty:
            continue

        # Sentinel to terminate
        if log == "DONE":
            break

        if isinstance(log, dict):
            if log.get("type") == "artifact":
                # Create artifact with the relevant files
                artifact_name = f"{config['wandb_name']}_layer_{config['layer']}_final"
                artifact = wandb.Artifact(name=artifact_name, type="model")

                if "atoms_file" in log and os.path.exists(log["atoms_file"]):
                    artifact.add_file(log["atoms_file"], name="atoms.pt")
                if "atom_indices_file" in log and os.path.exists(
                    log["atom_indices_file"]
                ):
                    artifact.add_file(log["atom_indices_file"], name="atom_indices.pt")
                if "metadata_file" in log and os.path.exists(log["metadata_file"]):
                    artifact.add_file(log["metadata_file"], name="metadata.yaml")

                wandb.log_artifact(artifact)
            else:
                # Normal W&B metrics
                wandb.log(log)
    wandb.finish()


def mp_encode(D, x, n_nonzero_coefs):
    """
    Simple Orthogonal Matching Pursuit-like procedure (aka Matching Pursuit).
    Args:
        D: Dictionary of shape [num_atoms, activation_dim].
        x: Activations of shape [batch_size, activation_dim].
        n_nonzero_coefs: int, how many nonzero coefficients to keep in OMP.
    Returns:
        coefficients: [batch_size, num_atoms].
    """
    batch_size = x.size(0)
    num_dict_atoms = D.size(0)

    residuals = x.clone()
    coefficients = torch.zeros(batch_size, num_dict_atoms, device=x.device)

    for _ in range(n_nonzero_coefs):
        correlations = torch.matmul(residuals, D.T)
        best_atoms = torch.argmax(torch.abs(correlations), dim=1)
        coeff_vals = correlations[torch.arange(batch_size), best_atoms]
        coefficients[torch.arange(batch_size), best_atoms] += coeff_vals
        residuals -= coeff_vals.unsqueeze(1) * D[best_atoms]

    return coefficients


class ITDA(nn.Module):
    """
    Core Iterative Dictionary Adder (ITDA) with an OMP-like encoding step.
    """

    def __init__(
        self, atoms: torch.Tensor, atom_indices: torch.Tensor, k: int, cfg=None
    ):
        super().__init__()
        self.k = k
        self.atoms = atoms
        self.atom_indices = atom_indices
        self.cfg = cfg

        self.dict_size = self.atoms.size(0)
        self.activation_dim = self.atoms.size(1) if self.atoms.size(0) > 0 else 0

    def encode(self, x: torch.Tensor):
        x = x.to(dtype=self.atoms.dtype)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        activations = mp_encode(self.atoms, x, self.k)
        return activations.view(*shape[:-1], -1)

    def decode(self, acts: torch.Tensor):
        return torch.matmul(acts.to(dtype=self.atoms.dtype), self.atoms)

    def forward(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    def normalize_decoder(self):
        norms = torch.norm(self.atoms, dim=1).clamp_min(1e-9)
        self.atoms = self.atoms / norms.unsqueeze(1)

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None) -> "ITDA":
        """
        Load a pre-trained ITDA instance (atoms, atom_indices, and metadata)
        from a directory produced by this training script.
        """
        metadata_path = os.path.join(path, "metadata.yaml")
        atoms_path = os.path.join(path, "atoms.pt")
        atom_indices_path = os.path.join(path, "atom_indices.pt")

        # Load metadata
        with open(metadata_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Load atoms and atom_indices
        atoms = torch.load(atoms_path, map_location=device)
        atom_indices = torch.load(atom_indices_path, map_location=device)

        # Pull 'k' from config
        k = cfg.get("k", 40)

        itda = cls(
            atoms=atoms,
            atom_indices=atom_indices,
            k=k,
            cfg=cfg,
        )
        if device is not None:
            itda.atoms = itda.atoms.to(device)
            itda.atom_indices = itda.atom_indices.to(device)

        return itda

    @property
    def W_dec(self):
        return self.atoms

    def __call__(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    @property
    def W_enc(self):
        """
        Required for certain external APIs (like SAE-bench).
        This is not used for OMP in practice, so we just return zeros.
        """
        return torch.zeros(
            (self.atoms.size(1), self.atoms.size(0)), device=self.atoms.device
        )

    @property
    def device(self):
        return self.atoms.device

    @property
    def dtype(self):
        return self.atoms.dtype

    def to(self, device=None, dtype=None):
        if device:
            self.atoms = self.atoms.to(device)
            self.atom_indices = self.atom_indices.to(device)
        if dtype:
            self.atoms = self.atoms.to(dtype)
        return self

    def normalize_decoder(self):
        norms = torch.norm(self.atoms, dim=1).clamp_min(1e-9)
        self.atoms = self.atoms / norms.unsqueeze(1)
        return self


class ITDATrainer:
    """
    Single-layer trainer that internally manages one ITDA instance.
    Handles the dictionary-building logic for 'layer' of a model.
    """

    def __init__(
        self,
        layer: int,
        steps: int,
        activation_dim: int,
        k: int,
        loss_threshold: float,
        lm_name: str,
        dataset: str,
        seq_len: int,
        batch_size: int,
        device: Optional[str] = None,
        wandb_name: str = "ITDA",
        submodule_name: Optional[str] = None,
        seed: Optional[int] = None,
        max_dict_size: int = 0,  # <--- New parameter
    ):
        self.layer = layer
        self.steps = steps
        self.activation_dim = activation_dim
        self.k = k
        self.loss_threshold = loss_threshold
        self.lm_name = lm_name
        self.dataset = dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_name = wandb_name
        self.submodule_name = submodule_name
        self.seed = seed

        # New: max dictionary size we allow
        self.max_dict_size = max_dict_size

        # ITDA instance for this layer
        self.itda = ITDA(
            atoms=torch.empty((0, activation_dim), device=self.device),
            atom_indices=torch.empty((0, 2), dtype=torch.long, device=self.device),
            k=self.k,
        )

    @property
    def config(self) -> dict:
        """
        Returns a dictionary suitable for logging/metadata.
        """
        return {
            "wandb_name": self.wandb_name,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "device": self.device,
            "submodule_name": self.submodule_name,
            "steps": self.steps,
            "activation_dim": self.activation_dim,
            "k": self.k,
            "loss_threshold": self.loss_threshold,
            "dataset": self.dataset,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "max_dict_size": self.max_dict_size,  # keep track of it for logging
        }

    @property
    def full(self) -> bool:
        """
        Returns True if we have a max_dict_size > 0 and the dictionary
        has reached (or exceeded) that size. Once 'full' is True,
        we skip further updates (i.e., no-op).
        """
        if self.max_dict_size <= 0:
            return False
        return self.itda.atoms.size(0) >= self.max_dict_size

    def update(self, step: int, x: torch.Tensor) -> float:
        """
        Update the single-layer dictionary using the activations x (shape [B, S, D]).
        Returns the mean error (float) for logging.

        If we're already 'full', do a no-op and return 0.0 (or some sentinel).
        """
        # If trainer is already full, skip any further updates
        if self.full:
            return 0.0

        B, S, D = x.shape
        flatten_x = x.reshape(-1, D).to(self.itda.dtype)

        # If dictionary is empty, fix shape
        if self.itda.atoms.size(0) == 0:
            self.itda.atoms = torch.empty((0, D), device=self.device)
            self.itda.activation_dim = D

        # Possibly expand dictionary if not full rank yet
        current_num_atoms = self.itda.atoms.size(0)
        dict_dim = D if current_num_atoms == 0 else self.itda.atoms.size(1)
        if current_num_atoms < dict_dim:
            n_missing = dict_dim - current_num_atoms

            unique_vals, inv_idx, counts = torch.unique(
                flatten_x, dim=0, return_inverse=True, return_counts=True
            )
            sorted_counts, sorted_idx = torch.sort(counts, descending=True)
            n_to_take = min(n_missing, unique_vals.size(0))
            new_rows = unique_vals[sorted_idx[:n_to_take]]

            new_atom_indices = []
            for row_idx in sorted_idx[:n_to_take]:
                match_positions = torch.nonzero(inv_idx == row_idx, as_tuple=True)[0]
                if len(match_positions) > 0:
                    mp_idx = match_positions[0].item()
                    batch_i = mp_idx // S
                    token_i = mp_idx % S
                    global_seq_idx = step * B + batch_i
                    new_atom_indices.append([global_seq_idx, token_i])
                else:
                    new_atom_indices.append([-1, -1])

            new_atom_indices = torch.tensor(
                new_atom_indices, dtype=torch.long, device=self.device
            )
            # Concat
            updated_atoms = torch.cat([self.itda.atoms, new_rows], dim=0)
            updated_indices = torch.cat([self.itda.atom_indices, new_atom_indices], dim=0)

            self.itda.atoms = updated_atoms
            self.itda.atom_indices = updated_indices
            self.itda.dict_size = self.itda.atoms.size(0)
            self.itda.activation_dim = self.itda.atoms.size(1)
            self.itda.normalize_decoder()

        # OMP reconstruction
        recon = self.itda(flatten_x)

        # Normalized MSE
        eps = 1e-9
        norm_x = flatten_x.norm(dim=1, keepdim=True).clamp_min(eps)
        norm_recon = recon.norm(dim=1, keepdim=True).clamp_min(eps)
        normalized_x = flatten_x / norm_x
        normalized_recon = recon / norm_recon
        errors = (normalized_x - normalized_recon).pow(2).mean(dim=1)

        # Add new atoms for items above threshold
        to_add = torch.nonzero(errors > self.loss_threshold, as_tuple=True)[0]
        new_atoms_list = []
        new_indices_list = []
        for idx_item in to_add.tolist():
            v = flatten_x[idx_item].unsqueeze(0)
            new_atoms_list.append(v)
            batch_i = idx_item // S
            token_i = idx_item % S
            global_seq_idx = step * B + batch_i
            new_indices_list.append([global_seq_idx, token_i])

        if len(new_atoms_list) > 0:
            new_atoms = torch.cat(new_atoms_list, dim=0).to(self.itda.dtype)
            new_atom_indices = torch.tensor(
                new_indices_list, dtype=torch.long, device=self.device
            )
            updated_atoms = torch.cat([self.itda.atoms, new_atoms], dim=0)
            updated_indices = torch.cat(
                [self.itda.atom_indices, new_atom_indices], dim=0
            )

            self.itda.atoms = updated_atoms
            self.itda.atom_indices = updated_indices
            self.itda.dict_size = self.itda.atoms.size(0)
            self.itda.activation_dim = self.itda.atoms.size(1)
            self.itda.normalize_decoder()

        # If we've exceeded max_dict_size, crop immediately
        if (self.max_dict_size > 0) and (self.itda.atoms.size(0) > self.max_dict_size):
            self.itda.atoms = self.itda.atoms[: self.max_dict_size].clone()
            self.itda.atom_indices = self.itda.atom_indices[: self.max_dict_size].clone()
            self.itda.dict_size = self.itda.atoms.size(0)
            self.itda.activation_dim = self.itda.atoms.size(1)
            self.itda = ITDA(
                atoms=self.itda.atoms,
                atom_indices=self.itda.atom_indices,
                k=self.itda.k,
            ).to(device=self.device, dtype=self.itda.dtype)

        # Reinitialize to refresh internal state if needed
        else:
            self.itda = ITDA(
                atoms=self.itda.atoms,
                atom_indices=self.itda.atom_indices,
                k=self.itda.k,
            ).to(device=self.device, dtype=self.itda.dtype)

        # Mean error for logging
        return errors.mean().item()


def run_training_loop(
    trainers: List[ITDATrainer],
    data_stream,
    tokenizer,
    model: HookedTransformer,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    device: str,
    wandb_project: str,
    wandb_entity: str = "",
    wandb_tags: Optional[List[str]] = None,
):
    """
    Main training loop for multiple single-layer trainers.
      - Spawns a separate W&B process+queue for each trainer.
      - Each trainer logs to its own W&B run.
      - Saves final artifacts to 'artifacts/runs/{run_id}' for each trainer.
      - Early-stops if all *capable* trainers (those with a .full property) are full.
    """
    # Prepare W&B processes
    layer_log_queues = {}
    wandb_processes = {}
    layer_run_dirs = {}

    # One W&B process per trainer
    for trainer in trainers:
        layer = trainer.layer
        run_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        run_dir = os.path.join("artifacts", "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        layer_run_dirs[layer] = run_dir

        layer_config = dict(trainer.config)
        layer_config["wandb_name"] = f"{trainer.wandb_name}_layer_{layer}"

        log_queue = mp.Queue()
        process = mp.Process(
            target=new_wandb_process,
            args=(
                run_id,
                layer_config,
                log_queue,
                wandb_entity,
                wandb_project,
                wandb_tags,
            ),
        )
        process.start()

        layer_log_queues[layer] = log_queue
        wandb_processes[layer] = process

    # Prepare caching hooks for only the layers we need
    layers_of_interest = [t.layer for t in trainers]
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers_of_interest]

    progress = tqdm(range(max_steps), desc="Training", unit="step")
    for step in progress:
        # Get a batch of text from the stream
        batch = []
        for _ in range(batch_size):
            try:
                text = next(data_stream)
            except StopIteration:
                print("Data stream exhausted.")
                break
            batch.append(text)
        if not batch:
            break

        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        ).to(device)

        # Forward pass with caching
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                stop_at_layer=max(layers_of_interest) + 1,
                names_filter=hook_names,
            )

        # Update each single-layer trainer unless it's already full
        for trainer in trainers:
            if hasattr(trainer, "full") and trainer.full:
                # Already full => skip
                continue

            layer_idx = trainer.layer
            x = cache[f"blocks.{layer_idx}.hook_resid_post"]  # [B, S, D]
            loss_val = trainer.update(step, x)

            # Log metrics to W&B queue
            log_data = {
                "step": step,
                "loss": loss_val,
                "dict_size": trainer.itda.atoms.size(0),
            }
            layer_log_queues[layer_idx].put(log_data)

        # Check if all trainers that define 'full' are indeed full
        trainers_with_full = [t for t in trainers if hasattr(t, "full")]
        if len(trainers_with_full) > 0:
            if all(t.full for t in trainers_with_full):
                print("All 'full'-capable trainers reached max dict size. Early stopping.")
                break

    # Save artifacts and finish W&B for each trainer
    for trainer in trainers:
        layer_idx = trainer.layer
        run_dir = layer_run_dirs[layer_idx]
        atoms_path = os.path.join(run_dir, "atoms.pt")
        atom_indices_path = os.path.join(run_dir, "atom_indices.pt")
        metadata_path = os.path.join(run_dir, "metadata.yaml")

        # Save final dictionary data
        torch.save(trainer.itda.atoms, atoms_path)
        torch.save(trainer.itda.atom_indices, atom_indices_path)

        # Save metadata
        layer_cfg = dict(trainer.config)
        with open(metadata_path, "w") as f:
            yaml.dump(layer_cfg, f)

        # Log final dict_size
        final_dict_size = trainer.itda.atoms.size(0)
        layer_log_queues[layer_idx].put({"dict_size": final_dict_size})

        # Notify W&B process of artifacts
        layer_log_queues[layer_idx].put(
            {
                "type": "artifact",
                "atoms_file": atoms_path,
                "atom_indices_file": atom_indices_path,
                "metadata_file": metadata_path,
            }
        )

        # Done
        layer_log_queues[layer_idx].put("DONE")
        wandb_processes[layer_idx].join()

    print("Training complete.")
    for trainer in trainers:
        print(f" Layer {trainer.layer} artifacts: {layer_run_dirs[trainer.layer]}")


def parse_args():
    """
    Example usage:
      python dl_train.py \
        --model_name EleutherAI/pythia-70m-deduped \
        --layers 0,1,2 \
        --dataset_name monology/pile-uncopyrighted \
        --seq_len 128 \
        --batch_size 4 \
        --loss_threshold 0.1 \
        --k 40 \
        --total_sequences 400 \
        --wandb_project "itda_separate_runs" \
        --wandb_entity "my_entity" \
        --wandb_tags tag1,tag2 \
        --device cuda \
        --crop_dict_size 100
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-layer ITDA Trainer (spawn multiple trainers for multiple layers)."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--dataset_name", type=str, default="monology/pile-uncopyrighted"
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Comma-separated layer indices (e.g. '0,1,2')",
    )
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--loss_threshold", type=float, required=True)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument(
        "--total_sequences",
        type=int,
        required=True,
        help="Total number of training sequences (will be divided by batch_size to get total steps).",
    )
    parser.add_argument("--wandb_project", type=str, default="itda")
    parser.add_argument("--wandb_entity", type=str, default="patrickaaleask")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--submodule_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated tags, e.g. 'tag1,tag2'.",
    )
    # Renamed or re-used argument. We'll store it as 'max_dict_size' inside the trainer.
    parser.add_argument(
        "--crop_dict_size",
        type=int,
        default=0,
        help="If > 0, trainer stops updating dictionary once this size is reached.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 1. Parse arguments
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 2. Compute number of steps
    max_steps = args.total_sequences // args.batch_size
    if max_steps <= 0:
        raise ValueError(
            "The computed number of steps is 0 or negative. Check total_sequences and batch_size."
        )

    # 3. Prepare data stream from Hugging Face dataset
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)

    # 4. Load model and tokenizer
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Get activation dimension
    activation_dim = model.cfg.d_model

    # 6. Build single-layer ITDA trainers
    layers_list = [int(x.strip()) for x in args.layers.split(",")]
    trainers = []
    for layer_idx in layers_list:
        trainer = ITDATrainer(
            layer=layer_idx,
            steps=max_steps,
            activation_dim=activation_dim,
            k=args.k,
            loss_threshold=args.loss_threshold,
            batch_size=args.batch_size,
            lm_name=args.model_name,
            dataset=args.dataset_name,
            seq_len=args.seq_len,
            device=device,
            submodule_name=args.submodule_name,
            seed=args.seed,
            # Here's where we attach the desired max dict size
            max_dict_size=args.crop_dict_size,  
        )
        trainers.append(trainer)

    # 7. Optional W&B tags
    wandb_tags = None
    if args.wandb_tags:
        wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]

    # 8. Run the (updated) training loop
    run_training_loop(
        trainers=trainers,
        data_stream=data_stream,
        tokenizer=tokenizer,
        model=model,
        max_steps=max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=wandb_tags,
    )
