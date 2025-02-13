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

        # Check sentinel
        if log == "DONE":
            break

        # Check if the message is an artifact or normal metric
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
                # Treat as normal W&B metrics
                wandb.log(log)
        # else: handle other message types if needed...

    wandb.finish()


def mp_encode(D, x, n_nonzero_coefs):
    """
    Simple Orthogonal Matching Pursuit-like procedure (aka Matching Pursuit).
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
    Iterative Dictionary Adder (ITDA) with OMP-like encoding.
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
        self.activation_dim = self.atoms.size(1)

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

    @property
    def device(self):
        return self.atoms.device

    @property
    def dtype(self):
        return self.atoms.dtype

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
        atoms = torch.load(atoms_path, map_location=device, weights_only=True)
        atom_indices = torch.load(
            atom_indices_path, map_location=device, weights_only=True
        )

        # Pull 'k' from config
        k = cfg.get("k", 40)

        itda = cls(
            atoms=atoms,
            atom_indices=atom_indices,
            k=k,
            cfg=cfg,
        )

        # Move internal tensors to device if requested
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
        # necessary for running core evals with sae bench
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
        if dtype:
            self.atoms = self.atoms.to(dtype)
        return self

    def normalize_decoder(self):
        norms = torch.norm(self.atoms, dim=1)
        self.atoms /= norms[:, None]
        return self


class MultiLayerITDATrainer:
    """
    Manages multiple ITDAs (one per layer). Each layer is updated from its own activations.
    """

    def __init__(
        self,
        steps: int,
        layers: List[int],
        activation_dim: int,
        k: int,
        loss_threshold: float,
        lm_name: str,
        dataset: str,
        seq_len: int,
        batch_size: int,
        device: Optional[str] = None,
        wandb_name: str = "MultiLayerITDA",
        submodule_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.steps = steps
        self.layers = layers
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

        # Maintain a separate ITDA per layer
        self.itdas = {}
        for layer_idx in layers:
            itda = ITDA(
                atoms=torch.empty((0, activation_dim), device=self.device),
                atom_indices=torch.empty((0, 2), dtype=torch.long, device=self.device),
                k=self.k,
            )
            self.itdas[layer_idx] = itda

    def update(
        self, step: int, activations_dict: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Updates each layer's ITDA using the corresponding activations.
        Returns a dict of {layer_idx -> scalar loss}.
        """
        all_losses = {}
        for layer_idx, x in activations_dict.items():
            itda = self.itdas[layer_idx]
            B, S, D = x.shape
            flatten_x = x.reshape(-1, D).to(itda.dtype)

            # If dictionary is empty, fix shape
            if itda.atoms.size(0) == 0:
                itda.atoms = torch.empty((0, D), device=self.device)
                itda.activation_dim = D

            # Possibly expand dictionary if not full rank yet
            current_num_atoms = itda.atoms.size(0)
            dict_dim = itda.atoms.size(1) if current_num_atoms > 0 else D
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
                    match_positions = torch.nonzero(inv_idx == row_idx, as_tuple=True)[
                        0
                    ]
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
                updated_atoms = torch.cat([itda.atoms, new_rows], dim=0)
                updated_indices = torch.cat(
                    [itda.atom_indices, new_atom_indices], dim=0
                )

                itda.atoms = updated_atoms
                itda.atom_indices = updated_indices
                itda.dict_size = itda.atoms.size(0)
                itda.activation_dim = itda.atoms.size(1)
                itda.normalize_decoder()

            # OMP reconstruction
            recon = itda(flatten_x)

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
                new_atoms = torch.cat(new_atoms_list, dim=0).to(itda.dtype)
                new_atom_indices = torch.tensor(
                    new_indices_list, dtype=torch.long, device=self.device
                )
                updated_atoms = torch.cat([itda.atoms, new_atoms], dim=0)
                updated_indices = torch.cat(
                    [itda.atom_indices, new_atom_indices], dim=0
                )

                itda.atoms = updated_atoms
                itda.atom_indices = updated_indices
                itda.dict_size = itda.atoms.size(0)
                itda.activation_dim = itda.atoms.size(1)
                itda.normalize_decoder()

            # Reinitialize the ITDA object to refresh any internal state
            self.itdas[layer_idx] = ITDA(
                atoms=itda.atoms, atom_indices=itda.atom_indices, k=itda.k
            ).to(device=self.device, dtype=itda.dtype)

            # Mean error for logging
            all_losses[layer_idx] = errors.mean().item()

        return all_losses

    @property
    def config(self):
        return {
            "wandb_name": self.wandb_name,
            "layers": self.layers,
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
        }


def run_training_loop(
    trainer: MultiLayerITDATrainer,
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
    Main training loop:
      - Spawns a separate W&B process+queue per layer (separate run for each).
      - Each layer logs to its own W&B run.
      - Saves final artifacts to 'artifacts/runs/{run_id}' for each layer.
    """

    # 1. Prepare W&B processes & local artifact dirs for each layer
    layer_log_queues = {}
    wandb_processes = {}
    layer_run_dirs = {}

    for layer_idx in trainer.layers:
        # Generate a unique run_id for this layer
        run_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        run_dir = os.path.join("artifacts", "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        layer_run_dirs[layer_idx] = run_dir

        # Make a config just for that layer's run
        layer_config = dict(trainer.config)
        layer_config["layer"] = layer_idx
        # Give each layer a unique run name
        layer_config["wandb_name"] = f"{trainer.wandb_name}_layer_{layer_idx}"

        # Create queue & start W&B process
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

        layer_log_queues[layer_idx] = log_queue
        wandb_processes[layer_idx] = process

    # 2. Build a list of hook names for the chosen layers
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in trainer.layers]

    # 3. Run training steps
    progress = tqdm(range(max_steps), desc="Training", unit="step")
    for step in progress:
        batch = []
        for _ in range(batch_size):
            try:
                text = next(data_stream)
            except StopIteration:
                break
            batch.append(text)
        if not batch:
            print("Data stream exhausted.")
            break

        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # Forward pass with caching up to the max layer we need
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                stop_at_layer=max(trainer.layers) + 1,
                names_filter=hook_names,
            )

        # Gather each layer’s activation from the cache
        activations_dict = {
            layer_idx: cache[f"blocks.{layer_idx}.hook_resid_post"]
            for layer_idx in trainer.layers
        }

        # Update ITDA for each layer
        layer_losses = trainer.update(step=step, activations_dict=activations_dict)

        # Log metrics to each layer's W&B queue
        for layer_idx in trainer.layers:
            log_data = {
                "step": step,
                "loss": layer_losses[layer_idx],
                "dict_size": trainer.itdas[layer_idx].atoms.size(0),
            }
            layer_log_queues[layer_idx].put(log_data)

    # 4. Finalize: save artifacts for each layer and signal W&B runs to close
    for layer_idx in trainer.layers:
        run_dir = layer_run_dirs[layer_idx]
        atoms_path = os.path.join(run_dir, "atoms.pt")
        atom_indices_path = os.path.join(run_dir, "atom_indices.pt")
        metadata_path = os.path.join(run_dir, "metadata.yaml")

        # Save final dictionary data
        torch.save(trainer.itdas[layer_idx].atoms, atoms_path)
        torch.save(trainer.itdas[layer_idx].atom_indices, atom_indices_path)

        # Save metadata
        layer_cfg = dict(trainer.config)
        layer_cfg["layer"] = layer_idx
        with open(metadata_path, "w") as f:
            yaml.dump(layer_cfg, f)

        # Notify W&B process of these artifacts
        layer_log_queues[layer_idx].put(
            {
                "type": "artifact",
                "atoms_file": atoms_path,
                "atom_indices_file": atom_indices_path,
                "metadata_file": metadata_path,
            }
        )

        # Signal that we're done
        layer_log_queues[layer_idx].put("DONE")
        wandb_processes[layer_idx].join()

    print("Training complete.")
    for lidx, rdir in layer_run_dirs.items():
        print(f" Layer {lidx} artifacts: {rdir}")


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
        --max_steps 100 \
        --wandb_project "itda_separate_runs" \
        --wandb_entity "my_entity" \
        --wandb_tags tag1,tag2 \
        --device cuda
    """
    parser = argparse.ArgumentParser(
        description="Multi-layer ITDA Trainer with separate W&B runs per layer."
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
    parser.add_argument("--max_steps", type=int, default=1000)
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
    return parser.parse_args()


def get_activation_dim(model_name):
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    return model.cfg.d_model


if __name__ == "__main__":
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare data stream from huggingface
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)
    activation_dim = get_activation_dim(args.model_name)

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers_list = [int(x.strip()) for x in args.layers.split(",")]

    # Build trainer
    trainer = MultiLayerITDATrainer(
        steps=args.max_steps,
        layers=layers_list,
        activation_dim=activation_dim,
        k=args.k,
        batch_size=args.batch_size,
        loss_threshold=args.loss_threshold,
        lm_name=args.model_name,
        dataset=args.dataset_name,
        seq_len=args.seq_len,
        device=device,
        submodule_name=args.submodule_name,
        seed=args.seed,
    )

    # Optional W&B tags
    wandb_tags = None
    if args.wandb_tags:
        wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]

    # Run training
    run_training_loop(
        trainer=trainer,
        data_stream=data_stream,
        tokenizer=tokenizer,
        model=model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_tags=wandb_tags,
    )
