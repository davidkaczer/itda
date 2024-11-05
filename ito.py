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


import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

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
    gpu_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
    total_memory_bytes = sum(tensor.element_size() * tensor.numel() for tensor in gpu_tensors)
    total_memory_mb = total_memory_bytes / (1024 ** 2)
    print(f"Total memory used by all GPU tensors: {total_memory_mb:.2f} MB")
    for tensor in gpu_tensors:
        memory_mb = tensor.element_size() * tensor.numel() / (1024 ** 2)
        print(f"Tensor: {tensor.shape}, Memory: {memory_mb:.2f} MB")


class ITO_SAE:
    def __init__(self, atoms, l0=8):
        self.atoms = atoms
        self.l0 = l0

    def encode(self, x):
        norm = x.norm(dim=1).unsqueeze(1)

        x = x / norm
        shape = x.size()

        x = x.view(-1, shape[-1])
        coefs, indices = omp_pytorch(self.atoms, x, self.l0)
        expanded = torch.zeros((x.size(0), self.atoms.size(0)), device=x.device)
        expanded.scatter_(1, indices, coefs)
        expanded = expanded.view(*shape[:-1], -1)
        return expanded

    def decode(self, x, acts):
        return torch.mm(acts, self.atoms) * x.norm(dim=1).unsqueeze(1)

    @property
    def W_dec(self):
        return self.atoms
    
    def __call__(self, x):
        acts = self.encode(x)
        return self.decode(x, acts)


# def omp_pytorch(D, x, n_nonzero_coefs):
#     """
#     For some reason this runs faster on CPU? Generally much faster than gradient pursuit.
#     """
#     batch_size, n_features = x.shape
#     n_atoms = D.shape[0]
#     indices = torch.zeros(
#         (batch_size, n_nonzero_coefs), device=D.device, dtype=torch.long
#     )
#     residual = x.clone()
#     selected_atoms = torch.zeros(
#         (batch_size, n_nonzero_coefs, n_features), device=D.device
#     )
#     available_atoms = torch.ones(
#         (batch_size, n_atoms), dtype=torch.bool, device=D.device
#     )
#     batch_indices = torch.arange(batch_size, device=D.device)

#     for k in range(n_nonzero_coefs):
#         correlations = torch.matmul(residual, D.T)
#         abs_correlations = torch.abs(correlations)
#         abs_correlations[~available_atoms] = 0
#         idx = torch.argmax(abs_correlations, dim=1)
#         indices[:, k] = idx
#         available_atoms[batch_indices, idx] = False
#         selected_atoms[:, k, :] = D[idx]
#         A = selected_atoms[:, : k + 1, :].transpose(1, 2)
#         B = x.unsqueeze(2)
#         try:
#             coef = torch.linalg.lstsq(A, B).solution
#         except RuntimeError as e:
#             print(f"Least squares solver failed at iteration {k}: {e}")
#             coef = torch.zeros(batch_size, k + 1, 1, device=D.device)
#         coef = coef.squeeze(2)
#         invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
#         if invalid_coefs.any():
#             coef[invalid_coefs] = 0.0
#         recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
#         residual = x - recon
#     return coef, indices

# def omp_pytorch(D, x, n_nonzero_coefs):
#     batch_size, n_features = x.shape
#     n_atoms = D.shape[0]
#     indices = torch.zeros(
#         (batch_size, n_nonzero_coefs), device=D.device, dtype=torch.long
#     )
#     residual = x.clone()
#     selected_atoms = torch.zeros(
#         (batch_size, n_nonzero_coefs, n_features), device=D.device, dtype=D.dtype
#     )
#     available_atoms = torch.ones(
#         (batch_size, n_atoms), dtype=torch.bool, device=D.device
#     )
#     batch_indices = torch.arange(batch_size, device=D.device)

#     D_T = D.T

#     for k in range(n_nonzero_coefs):
#         correlations = torch.matmul(residual, D_T)
#         abs_correlations = torch.abs(correlations)
#         abs_correlations[~available_atoms] = float("-inf")
#         idx = torch.argmax(abs_correlations, dim=1)
#         indices[:, k] = idx
#         available_atoms[batch_indices, idx] = False
#         selected_atoms[:, k, :] = D[idx]

#         A = selected_atoms[:, : k + 1, :].transpose(1, 2)
#         B = x.unsqueeze(2)

#         G = torch.bmm(A.transpose(1, 2), A)
#         y = torch.bmm(A.transpose(1, 2), B)

#         try:
#             L = torch.linalg.cholesky(G)
#             coef = torch.cholesky_solve(y, L).squeeze(2)
#         except RuntimeError as e:
#             # print(f"Cholesky solver failed at iteration {k}: {e}")
#             coef = torch.zeros(batch_size, k + 1, device=D.device)

#         invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
#         if invalid_coefs.any():
#             coef[invalid_coefs] = 0.0

#         coef = coef.to(A.dtype)
#         recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
#         residual = x - recon

#     return coef, indices


def omp_pytorch(D, x, n_nonzero_coefs, eps=1e-10):
    """
    Modernized and optimized implementation of Orthogonal Matching Pursuit using PyTorch.
    
    Args:
        D (torch.Tensor): Dictionary matrix of shape (n_atoms, n_features)
        x (torch.Tensor): Input signals of shape (batch_size, n_features)
        n_nonzero_coefs (int): Number of non-zero coefficients to select
        eps (float): Small constant for numerical stability
    
    Returns:
        tuple: (coefficients, selected indices)
        - coefficients: Sparse representation coefficients
        - indices: Indices of selected atoms
    """
    batch_size, n_features = x.shape
    n_atoms = D.shape[0]
    device, dtype = D.device, D.dtype
    
    # Pre-compute dictionary products and norms
    D_T = D.T.contiguous()
    gram_matrix = torch.mm(D, D_T)  # Pre-compute full Gram matrix
    D_norms = torch.diagonal(gram_matrix)
    D_normalized = D / (torch.sqrt(D_norms).unsqueeze(1) + eps)
    D_normalized_T = D_normalized.T.contiguous()
    
    # Initialize tensors
    indices = torch.zeros((batch_size, n_nonzero_coefs), device=device, dtype=torch.long)
    available_atoms = torch.ones((batch_size, n_atoms), dtype=torch.bool, device=device)
    batch_indices = torch.arange(batch_size, device=device)
    residual = x
    
    # Pre-compute initial correlations
    correlations = torch.matmul(residual, D_normalized_T)
    
    for k in range(n_nonzero_coefs):
        # Find highest correlations
        abs_correlations = torch.abs(correlations)
        abs_correlations[~available_atoms] = float('-inf')
        idx = torch.argmax(abs_correlations, dim=1)
        indices[:, k] = idx
        available_atoms[batch_indices, idx] = False
        
        # Extract selected atoms
        selected_atoms = D[indices[:, :k+1]]  # Shape: [batch_size, k+1, n_features]
        
        try:
            # Compute Gram matrix for selected atoms
            A = selected_atoms.transpose(1, 2)  # Shape: [batch_size, n_features, k+1]
            G = torch.bmm(A.transpose(1, 2), A)  # Shape: [batch_size, k+1, k+1]
            
            # Add small diagonal perturbation for stability
            G = G + eps * torch.eye(k + 1, device=device, dtype=dtype).unsqueeze(0)
            
            # Solve using Cholesky
            L = torch.linalg.cholesky(G)
            y = torch.bmm(A.transpose(1, 2), x.unsqueeze(2))
            
            # First solve Ly = b
            coef_temp = torch.linalg.solve_triangular(L, y, upper=False)
            # Then solve L^T x = y
            coef = torch.linalg.solve_triangular(L.transpose(-2, -1), coef_temp, upper=True)
            coef = coef.squeeze(2)
            
            # Handle numerical instabilities
            invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
            coef[invalid_coefs] = 0.0
            
            # Update residual
            recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
            residual = x - recon
            
            # Update correlations
            correlations = torch.matmul(residual, D_normalized_T)
            
            # Early stopping
            if torch.norm(residual) < eps:
                break
                
        except RuntimeError:
            # Fallback to QR decomposition
            Q, R = torch.linalg.qr(A)
            y = torch.bmm(Q.transpose(1, 2), x.unsqueeze(2))
            # Use solve_triangular instead of triangular_solve
            coef = torch.linalg.solve_triangular(R, y, upper=True)
            coef = coef.squeeze(2)
            invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
            coef[invalid_coefs] = 0.0
            recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
            residual = x - recon
            correlations = torch.matmul(residual, D_normalized_T)
    
    return coef, indices


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


def construct_atoms(
    normed_activations,
    batch_size=OMP_BATCH_SIZE,
    l0=OMP_L0,
    model_name=GPT2,
    quantile=0.01,
    plot_update_interval=10,
):
    atoms = torch.cat(
        [
            normed_activations[:, 0].unique(dim=0),
            normed_activations[:, 1].unique(dim=0),
        ],
        dim=0,
    ).to(device)
    atom_start_size = len(atoms)

    # Initialize atom_indices to track which activations are added
    atom_indices = torch.arange(atom_start_size).tolist()

    remaining_activations = (
        normed_activations[:, 2:]
        .permute(1, 0, 2)
        .reshape(-1, normed_activations.size(-1))
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
        losses.extend(loss[loss < 100.].cpu().tolist())

        mask = (100. > loss) & (loss > torch.quantile(loss, quantile))
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


def evaluate(
    sae, model_activations, batch_size=32
):
    losses = []
    for batch in tqdm(torch.split(model_activations, batch_size)):
        batch = batch.flatten(end_dim=1)
        recon = sae(batch)
        loss = ((batch - recon) ** 2).mean(dim=1)
        # XXX: can't reconstruct some? removing for now to investigate later
        # loss = loss[loss < 1]
        losses.extend(loss.tolist())
    return losses


# TODO: separate this for gemma and gpt2 as the parameters are totally different
def load_model(model_name, layer=12, width="16k", target_l0=40, gpt2_saes=list(range(1, 2)), device="cpu"):
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


def get_model_name(model_name, layer):
    if model_name == GPT2:
        return f"gpt2_layer_{layer}"
    elif model_name == GEMMA2:
        return f"gemma2_layer_{layer}"
    else:
        raise ValueError("Invalid model")


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

    model, saes, token_dataset = load_model(args.model, layer=args.layer, device=device, gpt2_saes=list(range(1, 2)))
    # TODO: Make this code work for larger datasets
    tokens = token_dataset['tokens'][:int(1_000 / 0.7), :args.seq_len].to(device)

    # don't need the saes for this
    hook_name = saes[0].cfg.hook_name
    del saes

    model_name = get_model_name(args.model, args.layer)

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

    # del model, token_dataset, tokens

    train_size = int(model_activations.size(0) * 0.7)

    # # XXX: This will use too much memory on larger datasets
    train_activations = model_activations[:train_size].to(device)
    # constraint to 8GB of memory so I can fit onto 4090
    element_size = train_activations.element_size()
    remaining_dimensions = train_activations.shape[1:]
    remaining_elements = torch.prod(torch.tensor(remaining_dimensions)).item()
    max_elements = (8 * 1024**3) // (element_size * remaining_elements)

    if train_activations.shape[0] > max_elements:
        train_activations = train_activations[:max_elements]

    try:
        atoms = torch.load(f"data/{model_name}/atoms.pt")
        atom_indices = torch.load(f"data/{model_name}/atom_indices.pt")
        print("Found atoms so not training")
    except FileNotFoundError:
        atoms, atom_indices, losses = construct_atoms(
            train_activations,
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
    test_activations = model_activations[-1000:].to(device)
    test_activations = model_activations[:100].to(device)
    del model_activations

    model, saes, token_dataset = load_model(args.model, layer=args.layer, device=device, gpt2_saes=list(range(1, 9)))
    del model, token_dataset

    saes_losses = []
    for sae in saes:
        losses = evaluate(
            sae,
            # test_activations.to(device),
            test_activations,
            batch_size=8,
        )
        saes_losses.append(losses)
        print(f"{sae.W_dec.size(0)} SAE loss:", torch.tensor(losses).mean().item())

    ito_sae = ITO_SAE(saes[-3].W_dec, l0=args.l0)
    del saes
    print(f"Evaluating ITO W_dec SAE with {atoms.size(0)} atoms")
    losses = evaluate(
        ito_sae,
        # test_activations.to(device),
        test_activations,
        batch_size=16,
    )
    print("Mean ITO W_dec loss:", torch.tensor(losses).mean().item())

    print(f"Evaluating ITO SAE with {atoms.size(0)} atoms")
    ito_sae = ITO_SAE(atoms, l0=args.l0)
    losses = evaluate(
        ito_sae,
        # test_activations.to(device),
        test_activations,
        batch_size=16,
    )
    print("Mean ITO loss:", torch.tensor(losses).mean().item())
