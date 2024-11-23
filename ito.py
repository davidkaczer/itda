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

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm

from meta_saes.sae import load_feature_splitting_saes, load_gemma_sae

OMP_BATCH_SIZE = 512
OMP_L0 = 40

GPT2 = "gpt2"
GEMMA2 = "gemma2"

OMP = "omp"
GP = "gp"

SEQ_LEN = 128
DATASET = "NeelNanda/pile-10k"


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


# TODO: Save the L0 and config alongside the the atoms.
class ITO_SAE:
    def __init__(self, atoms, l0=8, cfg=None):
        self.atoms = atoms
        self.l0 = l0
        self.cfg = cfg

    def encode(self, x):
        # XXX: Janky af but required for sae bench I think?
        original_device = x.device
        x = x.to(self.atoms.device)
        shape = x.size()
        x = x.view(-1, shape[-1])
        activations = omp_pytorch(self.atoms, x, self.l0)
        return activations

    def decode(self, acts):
        original_device = acts.device
        acts = acts.to(self.atoms.device)
        return torch.matmul(acts, self.atoms).to(original_device)

    @property
    def W_dec(self):
        return self.atoms

    def __call__(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    @property
    def device(self):
        return self.atoms.device
    
    @property
    def dtype(self):
        return self.atoms.dtype


def omp_pytorch(D, x, n_nonzero_coefs):
    """
    Naive implementation of Orthogonal Matching Pursuit (OMP) in PyTorch.

    In particular, this is O(k^3) where k is the number of nonzero coefficients.
    Consider more performance implementations from
    https://www.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569580549.pdf.
    """
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
            coef = torch.linalg.lstsq(A, B, driver="gelsd").solution
        except RuntimeError as e:
            print(f"Least squares solver failed at iteration {k}: {e}")
            coef = torch.zeros(batch_size, k + 1, 1, device=D.device)
        coef = coef.squeeze(2)
        invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
        if invalid_coefs.any():
            coef[invalid_coefs] = 0.0
        recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
        residual = x - recon
    activations = torch.zeros((x.size(0), n_atoms), device=x.device)
    activations.scatter_(1, indices, coef)
    activations = activations.view(*x.shape[:-1], -1)
    return activations


def update_plot(
    atoms,
    losses,
    atom_start_size,
    ratio_history,
    model_name,
):
    """
    Update the loss plot and the ratio plot.
    """
    window = 100
    np_losses = np.array(losses)
    smoothed_losses = np.convolve(np_losses, np.ones(window) / window, mode="valid")

    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].set_title(f"Added {len(atoms)} of {len(losses) + atom_start_size} atoms")
    axes[0].plot(smoothed_losses)
    axes[0].set_ylabel("Smoothed Losses")
    axes[0].set_xlabel("Iterations")

    axes[1].set_title("Ratio of len(atoms) / (len(losses) + atom_start_size)")
    axes[1].plot(ratio_history)
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel("Plot Update Interval")

    plt.tight_layout()
    plt.savefig(f"data/{model_name}/losses_and_ratio.png")
    plt.close()


def construct_atoms(
    activations,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    model_name=GPT2,
    target_loss=2.0,
    plot_update_interval=10,
):
    atoms = torch.cat(
        [
            activations[:, 0].unique(dim=0),
            activations[:, 1].unique(dim=0),
        ],
        dim=0,
    ).to(activations.device)
    atom_start_size = len(atoms)

    # Initialize atom_indices to track which activations are added
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
                model_name,
            )

    update_plot(
        atoms,
        losses,
        atom_start_size,
        ratio_history,
        model_name,
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


def get_model_name(model_name, layer, l0):
    return f"{model_name}_layer_{layer}_l0_{l0}"


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
    parser.parse_args()
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    model, saes, token_dataset = load_model(
        args.model, layer=args.layer, device=device, gpt2_saes=list(range(1, 2))
    )
    tokens = token_dataset["tokens"][:, : args.seq_len].to(device)

    # don't need the saes for this
    hook_name = saes[0].cfg.hook_name
    del saes

    model_name = get_model_name(args.model, args.layer, args.l0)

    os.makedirs(f"data/{model_name}", exist_ok=True)

    try:
        model_activations = torch.load(f"data/{model_name}/model_activations.pt")
    except FileNotFoundError:
        model_activations = []

        def add_activations(acts, hook=None):
            model_activations.append(acts.cpu())

        model.remove_all_hook_fns()
        model.add_hook(hook_name, add_activations)

        for batch in tqdm(torch.split(tokens, args.batch_size), desc="Model"):
            model(batch)

        model.remove_all_hook_fns()

        model_activations = torch.cat(model_activations)
        torch.save(model_activations, f"data/{model_name}/model_activations.pt")

    del model, token_dataset, tokens

    train_size = int(model_activations.size(0) * 0.7) // 4
    train_activations = model_activations[:train_size].cpu()

    try:
        atoms = torch.load(f"data/{model_name}/atoms.pt")
        atom_indices = torch.load(f"data/{model_name}/atom_indices.pt")
        print("Found atoms so not training")
    except FileNotFoundError:
        atoms, atom_indices, losses = construct_atoms(
            train_activations.cpu(),
            batch_size=args.batch_size,
            l0=args.l0,
            model_name=model_name,
        )
        print(f"Trained a model with {len(atoms)} atoms")

        torch.save(atoms, f"data/{model_name}/atoms.pt")
        torch.save(atom_indices, f"data/{model_name}/atom_indices.pt")
        with open(f"data/{model_name}/losses.pkl", "wb") as f:
            pickle.dump(losses, f)

    del train_activations
    model_activations = torch.load(f"data/{model_name}/model_activations.pt")
    test_activations = model_activations[-100:].cpu()
    del model_activations

    model, saes, token_dataset = load_model(
        args.model, layer=args.layer, device=device, gpt2_saes=list(range(1, 9))
    )
    del model, token_dataset

    saes_losses = []
    for sae in saes:
        losses = evaluate(
            sae,
            test_activations,
            batch_size=4,
        )
        saes_losses.append(losses)
        print(
            f"{sae.W_dec.size(0)} SAE loss:",
            losses.mean().item(),
            "on",
            len(losses),
            "samples",
        )

    ito_sae = ITO_SAE(saes[-3].W_dec.cpu(), l0=args.l0)
    print(f"Evaluating ITO W_dec SAE with {saes[-3].W_dec.size(0)} atoms")
    del saes
    losses = evaluate(
        ito_sae,
        test_activations.cpu(),
        batch_size=args.batch_size,
    )
    print(
        "Mean ITO W_dec loss:",
        losses.mean().item(),
        "on",
        len(losses),
        "samples",
    )

    print(f"Evaluating ITO SAE with {atoms.size(0)} atoms")
    ito_sae = ITO_SAE(atoms.cpu(), l0=args.l0)
    losses = evaluate(
        ito_sae,
        test_activations.cpu(),
        batch_size=args.batch_size,
    )
    print(
        "Mean ITO loss:",
        losses.mean().item(),
        "on",
        len(losses),
        "samples",
    )
