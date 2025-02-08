import os
import argparse
import yaml
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from queue import Empty
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb


def new_wandb_process(config, log_queue, entity, project, tags=None):
    """
    Spawns a new wandb run for a given layer.
    Continuously pulls logs from log_queue until it sees 'DONE'.
    """
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"], tags=tags)
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            # You may want to ensure each log dict has a 'step' key, or do wandb.log(log, step=log["step"])
            wandb.log(log)
        except Empty:
            continue
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


class MultiLayerITDATrainer:
    """
    Manages multiple ITDAs (one per layer). Each layer is updated from its activations.
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_name = wandb_name
        self.submodule_name = submodule_name
        self.seed = seed

        # Maintain a separate ITDA per layer
        # Initialize them with empty (0, D) atoms
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
            # Flatten
            flatten_x = x.reshape(-1, D).to(itda.dtype)

            # If dictionary is empty (atoms size (0, D)), fix shape
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

            # Add new atoms for those above threshold
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

                # De-duplicate
                updated_atoms, unique_idx = torch.unique(
                    updated_atoms, return_inverse=True, dim=0
                )
                updated_indices = updated_indices[unique_idx]

                itda.atoms = updated_atoms
                itda.atom_indices = updated_indices
                itda.dict_size = itda.atoms.size(0)
                itda.activation_dim = itda.atoms.size(1)
                itda.normalize_decoder()

            # Reconstruct the ITDA object (to refresh internal state)
            self.itdas[layer_idx] = ITDA(
                atoms=itda.atoms, atom_indices=itda.atom_indices, k=itda.k
            ).to(device=self.device, dtype=itda.dtype)

            # Save the mean error for logging
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
      - Spawns a separate wandb process+queue per layer
      - Single forward pass for each step
      - Logs each layer to its own wandb run
    """

    # 1. Prepare separate wandb processes & queues
    layer_log_queues = {}
    wandb_processes = {}
    for layer_idx in trainer.layers:
        # Make a config just for that layer's run
        layer_config = dict(trainer.config)
        layer_config["layer"] = layer_idx
        layer_config["wandb_name"] = f"{trainer.wandb_name}_layer_{layer_idx}"

        # Create queue & start process
        log_queue = mp.Queue()
        process = mp.Process(
            target=new_wandb_process,
            args=(layer_config, log_queue, wandb_entity, wandb_project, wandb_tags),
        )
        process.start()

        layer_log_queues[layer_idx] = log_queue
        wandb_processes[layer_idx] = process

    # 2. Build a list of hooks for all layers
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in trainer.layers]

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

        # 3. One forward pass with caching
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                stop_at_layer=max(trainer.layers) + 1,
                names_filter=hook_names,
            )

        # 4. Gather all activations in a dict
        activations_dict = {}
        for layer_idx in trainer.layers:
            activations_dict[layer_idx] = cache[f"blocks.{layer_idx}.hook_resid_post"]

        # 5. Update all layers at once
        layer_losses = trainer.update(step=step, activations_dict=activations_dict)

        # 6. Log each layer's results to its queue
        for layer_idx in trainer.layers:
            log = {
                "step": step,
                "loss": layer_losses[layer_idx],
                "dict_size": trainer.itdas[layer_idx].atoms.size(0),
            }
            # You can add more metrics if desired
            layer_log_queues[layer_idx].put(log)

    # 7. Finish: Save final artifacts and signal W&B processes to stop
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    for layer_idx in trainer.layers:
        # Save each layer's dictionary
        layer_dir = os.path.join(artifacts_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        torch.save(trainer.itdas[layer_idx].atoms, os.path.join(layer_dir, "atoms.pt"))
        torch.save(
            trainer.itdas[layer_idx].atom_indices,
            os.path.join(layer_dir, "atom_indices.pt"),
        )

        # Save metadata
        layer_cfg = dict(trainer.config)
        layer_cfg["layer"] = layer_idx
        with open(os.path.join(layer_dir, "metadata.yaml"), "w") as f:
            yaml.dump(layer_cfg, f)

        # Signal the wandb process to end
        layer_log_queues[layer_idx].put("DONE")
        wandb_processes[layer_idx].join()  # Wait for that process to finish


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
        description="Multi-layer ITDA Trainer with separate W&B runs"
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
    parser.add_argument("--wandb_project", type=str, default="itda_separate_runs")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--submodule_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated tags for the W&B runs, e.g. 'tag1,tag2'",
    )
    return parser.parse_args()


# Hard-coded example of known model dims. Adjust or remove as needed.
activation_dims = {
    "EleutherAI/pythia-70m-deduped": 512,
    "EleutherAI/pythia-160m-deduped": 768,
    "google/gemma-2-2b": 2304,
}

if __name__ == "__main__":
    # If you get pickling errors on some platforms, you may need:
    # mp.set_start_method("spawn", force=True)
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data stream
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)

    # Load model and tokenizer
    if args.model_name not in activation_dims:
        raise ValueError(
            f"Unknown activation_dim for model {args.model_name}. Please update `activation_dims` dict."
        )
    activation_dim = activation_dims[args.model_name]

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers_list = [int(x.strip()) for x in args.layers.split(",")]

    # Build the trainer
    trainer = MultiLayerITDATrainer(
        steps=args.max_steps,
        layers=layers_list,
        activation_dim=activation_dim,
        k=args.k,
        loss_threshold=args.loss_threshold,
        lm_name=args.model_name,
        dataset=args.dataset_name,
        seq_len=args.seq_len,
        device=device,
        submodule_name=args.submodule_name,
        seed=args.seed,
    )

    # Parse wandb_tags into a list (if provided)
    wandb_tags = None
    if args.wandb_tags is not None:
        wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    # Run the multi-process training loop
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