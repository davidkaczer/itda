import argparse
import random
import string
import os
import argparse
import uuid
from queue import Empty
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from datasets import load_dataset
from tqdm import tqdm

# For W&B logging
import wandb

# Transformer Lens is optional; only import if we need it
try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

# Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM


def capture_hidden_states_ndif(model, prompts, layers):
    hidden = {}
    with model.trace(prompts, remote=True) as run:
        for l in layers:
            hidden[l] = model.model.layers[l].output.save()
    return {k: v for k, v in hidden.items()}


def capture_hidden_states_hf(model, tokens, layers_of_interest):
    """
    Currently only works for ~llama
    """
    hidden_states_dict = {}
    hooks = []

    def hook_factory(idx):
        def hook(module, input, output):
            # We'll store the post-layer hidden state in hidden_states_dict
            if idx in layers_of_interest:
                hidden_states_dict[idx] = output[0].detach()

        return hook

    # Register hooks only on the layers we care about
    for i, layer_module in enumerate(model.model.layers):
        hooks.append(layer_module.register_forward_hook(hook_factory(i)))

    # Do a forward pass. We don't need output_hidden_states=True.
    _ = model(**tokens)

    # Remove hooks to avoid leaking references
    for h in hooks:
        h.remove()

    return hidden_states_dict


###############################################################################
#                               WANDB PROCESS                                  #
###############################################################################


def new_wandb_process(id, config, log_queue, entity, project, tags=None):
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


###############################################################################
#                          MATCHING PURSUIT ENCODER                           #
###############################################################################


def mp_encode(D, x, n_nonzero_coefs):
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


###############################################################################
#                         ITERATIVE DICTIONARY ADDER                          #
###############################################################################


class ITDA(nn.Module):
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
        metadata_path = os.path.join(path, "metadata.yaml")
        atoms_path = os.path.join(path, "atoms.pt")
        atom_indices_path = os.path.join(path, "atom_indices.pt")

        with open(metadata_path, "r") as f:
            cfg = yaml.safe_load(f)

        atoms = torch.load(atoms_path, map_location=device)
        atom_indices = torch.load(atom_indices_path, map_location=device)
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


###############################################################################
#                               SINGLE-LAYER TRAINER                          #
###############################################################################


class ITDATrainer:
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
        max_dict_size: int = 0,
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
        self.max_dict_size = max_dict_size

        self.itda = ITDA(
            atoms=torch.empty((0, activation_dim), device=self.device),
            atom_indices=torch.empty((0, 2), dtype=torch.long, device=self.device),
            k=self.k,
        )

    @property
    def config(self) -> dict:
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
            "max_dict_size": self.max_dict_size,
        }

    @property
    def full(self) -> bool:
        if self.max_dict_size <= 0:
            return False
        return self.itda.atoms.size(0) >= self.max_dict_size

    def update(self, step: int, x: torch.Tensor) -> float:
        if self.full:
            return 0.0

        B, S, D = x.shape
        flatten_x = x.reshape(-1, D).to(self.itda.dtype)

        if self.itda.atoms.size(0) == 0:
            self.itda.atoms = torch.empty((0, D), device=self.device)
            self.itda.activation_dim = D

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
            updated_atoms = torch.cat(
                [self.itda.atoms.to(x.device), new_rows.to(x.device)], dim=0
            )
            updated_indices = torch.cat(
                [self.itda.atom_indices, new_atom_indices], dim=0
            )

            self.itda.atoms = updated_atoms
            self.itda.atom_indices = updated_indices
            self.itda.dict_size = self.itda.atoms.size(0)
            self.itda.activation_dim = self.itda.atoms.size(1)
            self.itda.normalize_decoder()

        recon = self.itda(flatten_x)

        eps = 1e-9
        norm_x = flatten_x.norm(dim=1, keepdim=True).clamp_min(eps)
        norm_recon = recon.norm(dim=1, keepdim=True).clamp_min(eps)
        normalized_x = flatten_x / norm_x
        normalized_recon = recon / norm_recon
        errors = (normalized_x - normalized_recon).pow(2).mean(dim=1)

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

        if (self.max_dict_size > 0) and (self.itda.atoms.size(0) > self.max_dict_size):
            self.itda.atoms = self.itda.atoms[: self.max_dict_size].clone()
            self.itda.atom_indices = self.itda.atom_indices[
                : self.max_dict_size
            ].clone()
            self.itda.dict_size = self.itda.atoms.size(0)
            self.itda.activation_dim = self.itda.atoms.size(1)
            self.itda = ITDA(
                atoms=self.itda.atoms,
                atom_indices=self.itda.atom_indices,
                k=self.itda.k,
            ).to(device=self.device, dtype=self.itda.dtype)
        else:
            self.itda = ITDA(
                atoms=self.itda.atoms,
                atom_indices=self.itda.atom_indices,
                k=self.itda.k,
            ).to(device=self.device, dtype=self.itda.dtype)

        return errors.mean().item()


###############################################################################
#                           MAIN TRAINING LOOP                                #
###############################################################################


def run_training_loop(
    trainers: List[ITDATrainer],
    data_stream,
    tokenizer,
    model,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    device: str,
    wandb_project: str,
    wandb_entity: str = "",
    wandb_tags: Optional[List[str]] = None,
    use_huggingface: bool = False,
    use_ndif: bool = False,
    use_wandb: bool = True,  # new
):
    layer_log_queues, wandb_processes, layer_run_dirs = {}, {}, {}

    for tr in trainers:
        layer = tr.layer
        run_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        run_dir = os.path.join("artifacts", "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        layer_run_dirs[layer] = run_dir

        if use_wandb:
            cfg = dict(tr.config, wandb_name=f"{tr.wandb_name}_layer_{layer}")
            q = mp.Queue()
            p = mp.Process(
                target=new_wandb_process,
                args=(run_id, cfg, q, wandb_entity, wandb_project, wandb_tags),
            )
            p.start()
            layer_log_queues[layer], wandb_processes[layer] = q, p
        else:
            layer_log_queues[layer] = None  # sentinel

    layers_of_interest = [t.layer for t in trainers]
    if not use_huggingface:
        hook_names = [f"blocks.{l}.hook_resid_post" for l in layers_of_interest]

    for step in tqdm(range(max_steps), desc="Training", unit="step"):
        batch = [next(data_stream) for _ in range(batch_size)]
        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        ).to(device)

        if use_huggingface:
            with torch.no_grad():
                hidden_states_dict = capture_hidden_states_hf(
                    model, tokens, layers_of_interest
                )
        elif use_ndif:
            hidden_states_dict = capture_hidden_states_ndif(
                model, tokens, layers_of_interest
            )
        else:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens["input_ids"],
                    stop_at_layer=max(layers_of_interest) + 1,
                    names_filter=hook_names,
                )

        for tr in trainers:
            if tr.full:
                continue
            if use_huggingface:
                x = hidden_states_dict[tr.layer]
            elif use_ndif:
                x = hidden_states_dict[tr.layer][0].to(device)
            else:
                x = cache[f"blocks.{tr.layer}.hook_resid_post"]
            loss_val = tr.update(step, x)

            if use_wandb:
                layer_log_queues[tr.layer].put(
                    {"step": step, "loss": loss_val, "dict_size": tr.itda.atoms.size(0)}
                )

        if all(getattr(t, "full", False) for t in trainers):
            break

    for tr in trainers:
        layer = tr.layer
        run_dir = layer_run_dirs[layer]
        atoms_path = os.path.join(run_dir, "atoms.pt")
        atom_idx_path = os.path.join(run_dir, "atom_indices.pt")
        meta_path = os.path.join(run_dir, "metadata.yaml")

        torch.save(tr.itda.atoms, atoms_path)
        torch.save(tr.itda.atom_indices, atom_idx_path)
        with open(meta_path, "w") as f:
            yaml.dump(dict(tr.config), f)

        if use_wandb:
            q = layer_log_queues[layer]
            q.put({"dict_size": tr.itda.atoms.size(0)})
            q.put(
                {
                    "type": "artifact",
                    "atoms_file": atoms_path,
                    "atom_indices_file": atom_idx_path,
                    "metadata_file": meta_path,
                }
            )
            q.put("DONE")
            wandb_processes[layer].join()

    print("Training complete.")
    for tr in trainers:
        print(f" Layer {tr.layer} artifacts: {layer_run_dirs[tr.layer]}")


###############################################################################
#                                ARG PARSING                                  #
###############################################################################


def parse_layer_ranges(layers_str: str):
    """
    Parses a string like "0,1,2-4,10" into a list [0,1,2,3,4,10].
    Raises ValueError if the format is invalid.
    """
    all_layers = []
    chunks = [chunk.strip() for chunk in layers_str.split(",")]
    for chunk in chunks:
        if "-" in chunk:
            # Range format like "2-5"
            parts = chunk.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range chunk: '{chunk}'")
            start_str, end_str = parts
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise ValueError(f"Invalid range chunk: '{chunk}'")
            if start > end:
                raise ValueError(f"Range start must not exceed end: '{chunk}'")
            all_layers.extend(range(start, end + 1))
        else:
            # Single index
            try:
                layer = int(chunk)
            except ValueError:
                raise ValueError(f"Invalid layer index: '{chunk}'")
            all_layers.append(layer)
    return list(sorted(set(all_layers)))  # Sort and deduplicate if you like


def parse_args():
    """
    Example usage:
      python dl_train.py \
        --model_name EleutherAI/pythia-70m-deduped \
        --layers 0,1,2-5 \
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
        --crop_dict_size 100 \
        --use_huggingface \
        --offload_to_disk \
        --offload_folder ./offload_weights
    """
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
        help="Comma-separated layer indices or ranges, e.g. '0,1,2-5'.",
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
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        default=False,
        help="Use a Hugging Face model instead of Transformer Lens.",
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
    parser.add_argument(
        "--crop_dict_size",
        type=int,
        default=0,
        help="If > 0, trainer stops updating dictionary once this size is reached.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log to W&B (default: False).",
    )
    parser.add_argument(
        "--use_ndif",
        action="store_true",
        default=False,
        help="Use NDIF and nnsight to get the activations.",
    )
    parser.add_argument(
        "--ndif_api_key",
        type=str,
        default="",
        help="API key for NDIF (only used if --use_ndif is set).",
    )
    parser.add_argument(
        "--offload_to_disk",
        action="store_true",
        default=False,
        help="If set, will offload model weights to disk (only valid with --use_huggingface).",
    )
    parser.add_argument(
        "--offload_folder",
        type=str,
        default="offload",
        help="Folder path for offloaded weights (only used if --offload_to_disk is set).",
    )

    return parser.parse_args()


###############################################################################
#                                  MAIN                                       #
###############################################################################

if __name__ == "__main__":
    # 1. Parse arguments
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Compute number of steps
    max_steps = args.total_sequences // args.batch_size
    if max_steps <= 0:
        raise ValueError(
            "The computed number of steps is 0 or negative. Check total_sequences and batch_size."
        )

    # 3. Prepare data stream from Hugging Face dataset
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    data_stream = (item["text"] for item in dataset)

    # 4. Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_huggingface:
        print("Loading Hugging Face model...")
        if args.offload_to_disk:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                offload_folder=args.offload_folder,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        # Activation dimension from HF config
        activation_dim = model.config.hidden_size
    elif args.use_ndif:
        from nnsight import LanguageModel, CONFIG

        model = LanguageModel(
            args.model_name,
            device_map="auto",
        )
        activation_dim = model.config.hidden_size
        CONFIG.API.APIKEY = args.ndif_api_key
        host = CONFIG.API.HOST
        CONFIG.API.SSL = True
    else:
        print("Loading Transformer Lens (HookedTransformer)...")
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(args.model_name, device=device)
        activation_dim = model.cfg.d_model

    # 5. Expand the layers argument
    layers_list = parse_layer_ranges(args.layers)

    # 6. Build single-layer ITDA trainers
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
            max_dict_size=args.crop_dict_size,
        )
        trainers.append(trainer)

    # 7. Optional W&B tags
    wandb_tags = None
    if args.wandb_tags:
        wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]

    # 8. Run the training loop
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
        use_huggingface=args.use_huggingface,
        use_ndif=args.use_ndif,
        use_wandb=args.use_wandb,
    )
