"""
Implementations of inference time optimization (ITO) algorithms for dictionary
learning. Also implements a function for constructing a dictionary of atoms
using these methods.
"""

import argparse
import contextlib
import gc
import os
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from sae_lens import SAE
from sae_lens.load_model import load_model as sae_lens_load_model
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

import wandb

OMP_BATCH_SIZE = 512
OMP_L0 = 40

GPT2 = "gpt2"
GEMMA2 = "gemma2"

OMP = "omp"
GP = "gp"

SEQ_LEN = 128
DATASET = "NeelNanda/pile-10k"


@dataclass
class ITO_SAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    dtype: str

    context_size: int = None
    dtype: str = ""
    device: str = ""

    model_from_pretrained_kwargs: Dict = field(default_factory=dict)
    hook_head_index: Optional[int] = None
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple = (None,)


def load_feature_splitting_saes(device="cpu", saes_idxs=list(range(1, 9))):
    saes = []
    api = wandb.Api()

    class BackwardsCompatibleUnpickler(pickle.Unpickler):
        """
        An Unpickler that can load files saved before the "sae_lens" package namechange
        """

        def find_class(self, module: str, name: str):
            if name == "LanguageModelSAERunnerConfig":
                return super().find_class("sae_lens.config", name)
            return super().find_class(module, name)

    class BackwardsCompatiblePickleClass:
        Unpickler = BackwardsCompatibleUnpickler

    for i in saes_idxs:
        wandb_link = f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{768 * 2**(i-1)}:v9"

        artifact = api.artifact(
            wandb_link,
            type="model",
        )
        artifact_dir = artifact.download()
        file = os.listdir(artifact_dir)[0]

        state_dict = torch.load(
            os.path.join(artifact_dir, file),
            pickle_module=BackwardsCompatiblePickleClass,
        )
        state_dict["cfg"].activation_fn_kwargs = None
        state_dict["cfg"].model_kwargs = None
        state_dict["cfg"].model_from_pretrained_kwargs = None
        state_dict["cfg"].sae_lens_version = None
        state_dict["cfg"].sae_lens_training_version = None
        state_dict["cfg"].activation_fn_str = "relu"
        state_dict["cfg"].dtype = "torch.float32"
        state_dict["cfg"].finetuning_scaling_factor = 1.0
        state_dict["cfg"].hook_name = "blocks.8.hook_resid_pre"
        state_dict["cfg"].hook_layer = 8
        instance = SAE(cfg=state_dict["cfg"])
        instance.finetuning_scaling_factor = nn.Parameter(torch.tensor(1.0))
        state_dict["state_dict"]["finetuning_scaling_factor"] = nn.Parameter(
            torch.tensor(1.0)
        )
        instance.load_state_dict(state_dict["state_dict"], strict=True)
        instance.to(device)

        saes.append(instance)

    model = sae_lens_load_model("HookedTransformer", "gpt2-small", device=device)

    dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=False,
        add_bos_token=saes[0].cfg.prepend_bos,
    )

    return model, saes, token_dataset


def load_gemma_sae(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_3/width_16k/canonical",
    dataset="NeelNanda/c4-10k",
    device="cuda",
):

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()

    model = HookedTransformer.from_pretrained_no_processing(
        cfg_dict["model_name"], device=device
    )
    model.eval()

    dataset = load_dataset(
        path=dataset,
        split="train",
        streaming=False,
    )

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer,
        streaming=False,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    return model, [sae], token_dataset


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


def to_nonnegative_activations(activations):
    positive_activations = activations.clamp(min=0.0)
    negative_activations = activations.clamp(max=0.0)
    activations = torch.cat([positive_activations, negative_activations], dim=-1)
    return activations


def to_unbounded_activations(activations):
    half_dim = activations.size(-1) // 2
    positive_activations = activations[..., :half_dim]
    negative_activations = activations[..., half_dim:]
    original_activations = positive_activations + negative_activations
    return original_activations


class ITO_SAE:
    def __init__(self, atoms, l0=8, cfg=None):
        self.atoms = atoms
        self.l0 = l0
        self.cfg = cfg

    def encode(self, x):
        shape = x.size()
        x = x.view(-1, shape[-1])
        activations = omp_incremental_cholesky_with_fallback(self.atoms, x, self.l0)
        activations = to_nonnegative_activations(activations)
        return activations.view(*shape[:-1], -1)

    def decode(self, activations):
        original_activations = to_unbounded_activations(activations)
        original_device = original_activations.device
        original_activations = original_activations.to(self.atoms.device)
        return torch.matmul(original_activations, self.atoms).to(original_device)

    @property
    def W_dec(self):
        return torch.cat([self.atoms, -self.atoms], dim=0)

    def __call__(self, x):
        acts = self.encode(x)
        return self.decode(acts)

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


def svd_min_norm_solution(A, B, rcond=None):
    m, n = A.shape[-2], A.shape[-1]
    U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver="gesvda")
    if rcond is None:
        rcond = torch.finfo(S.dtype).eps * max(m, n)
    max_singular = S.max(dim=-1, keepdim=True).values
    tol = rcond * max_singular
    large_singular = S > tol
    S_inv = torch.zeros_like(S)
    S_inv[large_singular] = 1.0 / S[large_singular]
    S_inv = S_inv.unsqueeze(-1)
    U_T_B = torch.matmul(U.transpose(-2, -1), B)
    S_inv_U_T_B = S_inv * U_T_B
    X = torch.matmul(Vh.transpose(-2, -1), S_inv_U_T_B)
    return X


def omp_pytorch(D, x, n_nonzero_coefs):
    batch_size, n_features = x.shape
    n_atoms = D.shape[0]
    indices = torch.zeros(
        (batch_size, n_nonzero_coefs), device=D.device, dtype=torch.long
    )
    residual = x.clone()
    selected_atoms = torch.zeros(
        (batch_size, n_nonzero_coefs, n_features), device=D.device
    )
    available_atoms = torch.ones(
        (batch_size, n_atoms), dtype=torch.bool, device=D.device
    )
    batch_indices = torch.arange(batch_size, device=D.device)

    for k in range(n_nonzero_coefs):
        correlations = torch.matmul(residual, D.T)
        abs_correlations = torch.abs(correlations)
        abs_correlations[~available_atoms] = 0
        idx = torch.argmax(abs_correlations, dim=1)
        indices[:, k] = idx
        available_atoms[batch_indices, idx] = False
        selected_atoms[:, k, :] = D[idx]
        A = selected_atoms[:, : k + 1, :].transpose(1, 2)
        B = x.unsqueeze(2)
        try:
            coef = svd_min_norm_solution(A, B)
        except RuntimeError as e:
            coef = torch.zeros(batch_size, k + 1, 1, device=D.device)
        coef = coef.squeeze(2)
        invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
        if invalid_coefs.any():
            coef[invalid_coefs] = 0.0
        recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
        residual = x - recon
    activations = torch.zeros((x.size(0), n_atoms), device=x.device)
    activations.scatter_(1, indices, coef)
    return activations


def omp_incremental_cholesky(D, x, n_nonzero_coefs, device=None):
    if device is None:
        device = D.device
    D = D.to(device)
    x = x.to(device)
    batch_size, n_features = x.shape
    n_atoms = D.shape[0]

    indices = torch.zeros(
        (batch_size, n_nonzero_coefs), device=device, dtype=torch.long
    )
    residual = x.clone()
    selected_atoms = torch.zeros(
        (batch_size, n_nonzero_coefs, n_features), device=device, dtype=D.dtype
    )
    available_atoms = torch.ones((batch_size, n_atoms), dtype=torch.bool, device=device)

    L = None
    b = None
    Beta = None

    for k in range(n_nonzero_coefs):
        correlations = torch.matmul(residual, D.T)
        abs_correlations = torch.abs(correlations)
        abs_correlations[~available_atoms] = 0
        idx = torch.argmax(abs_correlations, dim=1)
        indices[:, k] = idx
        available_atoms[torch.arange(batch_size, device=device), idx] = False

        a_k = D[idx, :]
        selected_atoms[:, k, :] = a_k
        a_k_x = torch.sum(a_k * x, dim=-1, keepdim=True)

        if k == 0:
            norm_a_k2 = torch.sum(a_k * a_k, dim=-1, keepdim=True)
            L = torch.sqrt(norm_a_k2).unsqueeze(-1)
            b = a_k_x.unsqueeze(-1)
            Beta = b / (L * L)
        else:
            prev_atoms = selected_atoms[:, :k, :]
            c = torch.sum(prev_atoms * a_k.unsqueeze(1), dim=-1)
            y = torch.linalg.solve_triangular(
                L, c.unsqueeze(-1), upper=False, unitriangular=False
            )
            y_norm2 = torch.sum(y.squeeze(-1) ** 2, dim=-1)

            norm_a_k2 = torch.sum(a_k * a_k, dim=-1)
            diff = norm_a_k2 - y_norm2
            diff = torch.clamp(diff, min=0.0)
            r_kk = torch.sqrt(diff + 1e-12)

            L_new = torch.zeros(
                (batch_size, k + 1, k + 1), dtype=L.dtype, device=device
            )
            L_new[:, :k, :k] = L
            L_new[:, k, :k] = y.squeeze(-1)
            L_new[:, k, k] = r_kk
            L = L_new

            b_new = torch.cat([b, a_k_x.unsqueeze(-1)], dim=1)
            b = b_new
            Beta = torch.cholesky_solve(b, L, upper=False)

        A_cur = selected_atoms[:, : k + 1, :]
        recon = torch.bmm(A_cur.transpose(1, 2), Beta).squeeze(-1)
        residual = x - recon

    Beta_final = Beta.squeeze(-1)
    activations = torch.zeros((batch_size, n_atoms), device=device, dtype=D.dtype)
    activations.scatter_(1, indices, Beta_final)
    return activations, Beta_final, residual, indices, selected_atoms


def omp_incremental_cholesky_with_fallback(D, x, n_nonzero_coefs, device=None):
    if device is None:
        device = D.device

    activations, Beta_final, residual, indices, selected_atoms = (
        omp_incremental_cholesky(D, x, n_nonzero_coefs, device=device)
    )

    invalid_mask = torch.isnan(Beta_final) | torch.isinf(Beta_final)
    residual_norm = torch.norm(residual, dim=-1)
    high_error_mask = residual_norm > 1e8
    fallback_mask = invalid_mask.any(dim=-1) | high_error_mask

    if fallback_mask.any():
        fallback_indices = torch.where(fallback_mask)[0]
        D_fallback = D
        x_fallback = x[fallback_indices]
        fallback_acts = omp_pytorch(D_fallback, x_fallback, n_nonzero_coefs)
        activations[fallback_indices] = fallback_acts

    return activations



def update_plot(atoms, losses, atom_start_size, ratio_history, run_dir):
    window = 1000
    np_losses = np.array(losses)
    np_ratios = np.array(ratio_history)

    # Smooth the losses if we have enough data points
    if len(np_losses) >= window:
        smoothed_losses = np.convolve(np_losses, np.ones(window) / window, mode="valid")
    else:
        smoothed_losses = np_losses

    # Smooth the ratio history if we have enough data points
    if len(np_ratios) >= window:
        smoothed_ratios = np.convolve(np_ratios, np.ones(window) / window, mode="valid")
    else:
        smoothed_ratios = np_ratios

    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot smoothed losses
    axes[0].set_title(f"Added {len(atoms)} of {len(losses) + atom_start_size} atoms")
    axes[0].plot(smoothed_losses)
    axes[0].set_ylabel("Smoothed Losses")
    axes[0].set_xlabel("Iterations")

    # Plot smoothed ratio history
    axes[1].set_title("Ratio of len(atoms) / (len(losses) + atom_start_size)")
    axes[1].plot(smoothed_ratios)
    axes[1].set_ylabel("Smoothed Ratio")
    axes[1].set_xlabel("Plot Update Interval")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "losses_and_ratio.png"))
    plt.close()


def construct_atoms(
    activations,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    target_loss=2.0,
    plot_update_interval=10,
    run_dir=".",
):
    atoms = torch.cat(
        [activations[:, 0].unique(dim=0), activations[:, 1].unique(dim=0)], dim=0
    ).to(activations.device)
    atom_start_size = len(atoms)

    atom_indices = torch.arange(atom_start_size).tolist()

    remaining_activations = (
        activations[:, 2:].permute(1, 0, 2).reshape(-1, activations.size(-1))
    )
    losses = []
    ratio_history = []

    pbar = tqdm(total=remaining_activations.size(0))
    batch_counter = 0
    while remaining_activations.size(0) > 0:
        batch_activations = remaining_activations[:batch_size]
        remaining_activations = remaining_activations[batch_size:]

        sae = ITO_SAE(atoms, l0)
        recon = sae(batch_activations)
        loss = ((batch_activations - recon) ** 2).mean(dim=1)
        losses.extend(loss[loss < 100.0].cpu().tolist())

        mask = loss > target_loss
        new_atoms = batch_activations[mask]
        if new_atoms.size(0) > 0:
            atoms = torch.cat([atoms, new_atoms], dim=0)
            new_indices = torch.arange(
                len(atom_indices), len(atom_indices) + new_atoms.size(0)
            ).tolist()
            atom_indices.extend(new_indices)

        pbar.update(len(batch_activations))
        batch_counter += 1

        ratio = new_atoms.size(0) / batch_size
        ratio_history.append(ratio)

        if batch_counter % plot_update_interval == 0:
            update_plot(
                atoms,
                losses,
                atom_start_size,
                ratio_history,
                run_dir=run_dir,
            )

    update_plot(
        atoms,
        losses,
        atom_start_size,
        ratio_history,
        run_dir=run_dir,
    )

    pbar.close()

    return atoms, atom_indices, losses


def evaluate(sae, model_activations, batch_size=32):
    losses = []
    for batch in tqdm(torch.split(model_activations, batch_size)):
        batch = batch.flatten(end_dim=1).to(sae.device)
        recon = sae(batch)
        loss = (batch - recon) ** 2
        losses.extend(loss.detach().cpu())
    return torch.stack(losses)


def get_atom_indices(atoms, activations, batch_size: int = 256):
    flattened_activations = activations.view(-1, activations.size(-1))
    num_atoms = atoms.size(0)
    flattened_idxs = torch.empty(num_atoms, dtype=torch.long, device=atoms.device)
    for start_idx in tqdm(range(0, flattened_activations.size(0), batch_size)):
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


# TODO: separate this for gemma and gpt2 as the parameters are totally different
def load_model(
    model_name,
    layer=12,
    width="16k",
    target_l0=40,
    gpt2_saes=list(range(1, 2)),
    device="cpu",
):
    pretrained_saes = get_pretrained_saes_directory()
    if model_name == GPT2:
        model, saes, token_dataset = load_feature_splitting_saes(
            device=device,
            saes_idxs=gpt2_saes,
        )
    elif model_name == GEMMA2:
        release = "gemma-scope-2b-pt-res"
        sae_ids = pretrained_saes[release].saes_map.keys()
        sae_ids = [s for s in sae_ids if s.startswith(f"layer_{layer}")]
        sae_ids = [s for s in sae_ids if f"width_{width}" in s]
        l0s = [int(s.split("_")[-1]) for s in sae_ids]
        # get index of l0 nearest to target_l0
        l0 = l0s.index(min(l0s, key=lambda x: abs(x - target_l0)))
        sae_id = sae_ids[l0]

        model, saes, token_dataset = load_gemma_sae(
            release=release,
            sae_id=sae_id,
            device=device,
            dataset=DATASET,
        )
    else:
        raise ValueError("Invalid model")

    return model, saes, token_dataset


def load_model_activations(args, device):
    # Load the model and dataset
    model, saes, token_dataset = load_model(
        args.model, layer=args.layer, device=device, gpt2_saes=list(range(1, 2))
    )
    hook_name = saes[0].cfg.hook_name
    del saes

    tokens = token_dataset["tokens"][:, : args.seq_len].to(device)
    activations_path = f"data/{args.model}/model_activations.pt"

    # Try to load precomputed activations
    try:
        model_activations = torch.load(activations_path, weights_only=True)
    except FileNotFoundError:
        # Need to generate activations
        model_activations = []

        def add_activations(acts, hook=None):
            model_activations.append(acts.cpu())

        model.remove_all_hook_fns()
        # Add a forward hook to capture the activations at the specified hook_name
        model.add_hook(hook_name, add_activations)

        # Run the model over the dataset and record activations
        for batch in tqdm(torch.split(tokens, args.batch_size), desc="Model"):
            model(batch)

        model.remove_all_hook_fns()

        # Concatenate all recorded activations
        model_activations = torch.cat(model_activations)
        # Save them for future use
        os.makedirs(f"data/{args.model}", exist_ok=True)
        torch.save(model_activations, activations_path)

    # Clean up references
    del model, token_dataset, tokens
    return model_activations


def filter_runs(base_dir="runs", **criteria):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=GPT2)
    parser.add_argument("--batch_size", type=int, default=OMP_BATCH_SIZE)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--target_loss", type=float, default=2.0)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    matching_runs = filter_runs(
        base_dir="runs",
        model=args.model,
        batch_size=args.batch_size,
        l0=args.l0,
        layer=args.layer,
        seq_len=args.seq_len,
        target_loss=args.target_loss,
    )

    if len(matching_runs) > 0:
        # Use the first matching run directory
        run_dir = matching_runs[0]
        print(f"Found existing run with these parameters: {run_dir}")
        # Load metadata to confirm atom count, etc.
        with open(os.path.join(run_dir, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)

        # Load previously computed atoms
        atoms_path = os.path.join(run_dir, "atoms.pt")
        if not os.path.exists(atoms_path):
            raise FileNotFoundError(
                f"Matched run at {run_dir} does not have atoms.pt, something is wrong."
            )
        atoms = torch.load(atoms_path, weights_only=True)

        # Load model activations (for validation)
        model_activations = load_model_activations(args, device)
        test_activations = model_activations[-1000:].to(device)
        ito_sae = ITO_SAE(atoms.to(device), l0=args.l0)
        eval_losses = evaluate(
            ito_sae, test_activations.to(device), batch_size=args.batch_size
        )

        plt.hist(torch.log(eval_losses.mean(-1)).cpu().numpy(), bins=100)
        plt.savefig(os.path.join(run_dir, "loss_hist.png"))
        plt.close()

        print(
            "Mean ITO loss:",
            eval_losses.mean().item(),
            "on",
            len(eval_losses),
            "samples",
        )

    else:
        # Create a new run if no existing run matches
        run_id = str(uuid.uuid4())
        run_dir = os.path.join("runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save arguments and metadata in metadata.yaml (initial)
        metadata = {
            "model": args.model,
            "batch_size": args.batch_size,
            "l0": args.l0,
            "layer": args.layer,
            "seq_len": args.seq_len,
            "target_loss": args.target_loss,
            "device": str(device),
        }
        with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
            yaml.safe_dump(metadata, f)

        model_activations = load_model_activations(args, device)

        train_size = int(model_activations.size(0) * 0.7) // 4
        train_activations = model_activations[:train_size].to(device)

        # Construct atoms
        atoms_path = os.path.join(run_dir, "atoms.pt")
        losses_path = os.path.join(run_dir, "losses.pkl")
        indices_path = os.path.join(run_dir, "atom_indices.pt")

        if not os.path.exists(atoms_path):
            atoms, atom_indices, losses = construct_atoms(
                train_activations,
                batch_size=args.batch_size,
                l0=args.l0,
                target_loss=args.target_loss,
                run_dir=run_dir,
            )
            torch.save(atoms, atoms_path)
            with open(losses_path, "wb") as f:
                pickle.dump(losses, f)

            atom_indices = get_atom_indices(
                atoms.to(device), model_activations, batch_size=1024
            ).cpu()
            torch.save(atom_indices, indices_path)

            # Update metadata with the number of atoms
            metadata["num_atoms"] = atoms.size(0)
            with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
                yaml.safe_dump(metadata, f)
        else:
            # If already present, just load them
            atoms = torch.load(atoms_path, weights_only=True)
            metadata["num_atoms"] = atoms.size(0)
            with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
                yaml.safe_dump(metadata, f)

        # Evaluate
        test_activations = model_activations[-1000:].to(device)
        ito_sae = ITO_SAE(atoms.to(device), l0=args.l0)
        eval_losses = evaluate(
            ito_sae, test_activations.to(device), batch_size=args.batch_size
        )

        plt.hist(torch.log(eval_losses.mean(-1)).cpu().numpy(), bins=100)
        plt.savefig(os.path.join(run_dir, "loss_hist.png"))
        plt.close()

        print(
            "Mean ITO loss:",
            eval_losses.mean().item(),
            "on",
            len(eval_losses),
            "samples",
        )

        # Example usage of filter_runs
        matched = filter_runs(base_dir="runs", model="gpt2", layer=8)
        print("Matching runs with model=gpt2, layer=8:", matched)
