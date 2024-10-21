"""
Implementations of inference time optimization (ITO) algorithms for dictionary
learning. Also implements a function for constructing a dictionary of atoms
using these methods.
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from meta_saes.sae import load_feature_splitting_saes, load_gemma_sae

OMP_BATCH_SIZE = 512
OMP_L0 = 40


def gp_pytorch(D, x, n_nonzero_coefs, lr=1e-2, n_iterations=100):
    """
    Gradient Pursuit implementation in PyTorch.

    Args:
        D (torch.Tensor): Dictionary matrix (n_atoms, n_features).
        x (torch.Tensor): Input signal matrix (batch_size, n_features).
        n_nonzero_coefs (int): Number of non-zero coefficients to select.
        lr (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for gradient descent.

    Returns:
        coef (torch.Tensor): Coefficients for the selected atoms (batch_size, n_nonzero_coefs).
        indices (torch.Tensor): Indices of the selected atoms (batch_size, n_nonzero_coefs).
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

    coef = torch.zeros((batch_size, n_nonzero_coefs), device=D.device)

    for k in range(n_nonzero_coefs):
        correlations = torch.matmul(residual, D.T)
        abs_correlations = torch.abs(correlations)
        abs_correlations[~available_atoms] = 0
        idx = torch.argmax(abs_correlations, dim=1)
        indices[:, k] = idx
        available_atoms[batch_indices, idx] = False
        selected_atoms[:, k, :] = D[idx]
        for _ in range(n_iterations):
            A = selected_atoms[:, : k + 1, :].transpose(1, 2)
            recon = torch.bmm(A, coef[:, : k + 1].unsqueeze(2)).squeeze(2)
            residual = x - recon
            gradient = -2 * torch.bmm(A.transpose(1, 2), residual.unsqueeze(2)).squeeze(
                2
            )
            coef[:, : k + 1] -= lr * gradient
        residual = x - torch.bmm(A, coef[:, : k + 1].unsqueeze(2)).squeeze(2)

    return coef, indices


def omp_pytorch(D, x, n_nonzero_coefs):
    """
    For some reason this runs faster on CPU? Generally much faster than gradient pursuit.
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
            coef = torch.linalg.lstsq(A, B).solution
        except RuntimeError as e:
            print(f"Least squares solver failed at iteration {k}: {e}")
            coef = torch.zeros(batch_size, k + 1, 1, device=D.device)
        coef = coef.squeeze(2)
        invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
        if invalid_coefs.any():
            coef[invalid_coefs] = 0.0
        recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
        residual = x - recon
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
    normed_activations, ito_fn=omp_pytorch, batch_size=OMP_BATCH_SIZE, l0=OMP_L0
):
    atoms = torch.cat(
        [
            normed_activations[:, 0].unique(dim=0),
            normed_activations[:, 1].unique(dim=0),
        ],
        dim=0,
    )
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
    plot_update_interval = 1
    batch_counter = 0
    while remaining_activations.size(0) > 0:
        batch_activations = remaining_activations[:batch_size]
        remaining_activations = remaining_activations[batch_size]

        coefs, indices = ito_fn(atoms.float(), batch_activations.float(), l0)
        selected_atoms = atoms[indices]
        recon = torch.bmm(coefs.unsqueeze(1), selected_atoms.float()).squeeze(1)
        loss = ((batch_activations - recon) ** 2).sum(dim=1)
        loss = torch.clamp(loss, 0, 1)
        losses.extend(loss.cpu().tolist())

        mask = loss > 0.3
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
                plot_update_interval,
                batch_counter,
            )

    update_plot(
        atoms,
        losses,
        atom_start_size,
        ratio_history,
        plot_update_interval,
        batch_counter,
    )

    pbar.close()

    return atoms, atom_indices, losses


def evaluate(
    ito_fn, atoms, model_activations, l0, batch_size=32, return_activations=False
):
    losses = []
    indices = []
    activations = []

    for batch in tqdm(torch.split(model_activations, batch_size), desc="Evaluating..."):
        flattened_batch = batch.flatten(end_dim=1)
        omp_batch = flattened_batch / flattened_batch.norm(dim=1).unsqueeze(1)

        coefs, indices = ito_fn(atoms, omp_batch, l0)
        selected_atoms = atoms[indices]
        omp_recon = torch.bmm(coefs.unsqueeze(1), selected_atoms).squeeze(1)
        loss = ((omp_batch - omp_recon) ** 2).mean(dim=1)
        # XXX: can't reconstruct some? removing for now to investigate later
        loss = loss[loss < 1]
        losses.extend(loss.tolist())

        if return_activations:
            activations.extend(omp_batch.cpu().tolist())
            indices.extend(indices.cpu().tolist())

    if return_activations:
        return losses, activations, indices
    return losses


GPT2 = "gpt2"
GEMMA2 = "gemma2"

OMP = "omp"
GP = "gp"

SEQ_LEN = 128
DATASET = "NeelNanda/pile-10k"


def load_model(model_name, device='cpu'):
    if model_name == GPT2:
        model, saes, token_dataset = load_feature_splitting_saes(
            device=device,
            saes_idxs=list(range(1, 2)),
        )
    elif model_name == GEMMA2:
        model, saes, token_dataset = load_gemma_sae(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_12/width_16k/average_l0_41",
            device=device,
            dataset=DATASET,
        )
    else:
        raise ValueError("Invalid model")

    return model, saes, token_dataset


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=GPT2)
    parser.add_argument("--batch_size", type=int, default=OMP_BATCH_SIZE)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--ito_fn", type=str, default=OMP)
    parser.parse_args()
    args = parser.parse_args()

    # TODO: Loading the SAEs is unnecessary

    torch.set_grad_enabled(False)

    model, saes, token_dataset = load_model(args.model)

    try:
        model_activations = torch.load(f"data/{args.model}/model_activations.pt")
    except FileNotFoundError:
        model_activations = []
        tokens = torch.stack([s["tokens"] for s in token_dataset])[
            :, : args.seq_len
        ].to(device)

        def add_activations(acts, hook=None):
            model_activations.append(acts.cpu())

        model.remove_all_hook_fns()
        model.add_hook(saes[0].cfg.hook_name, add_activations)

        for batch in tqdm(torch.split(tokens, args.batch_size), desc="Model"):
            model(batch)

        model.remove_all_hook_fns()

        model_activations = torch.cat(model_activations)
        torch.save(model_activations, f"data/{args.model}/model_activations.pt")

    train_size = int(model_activations.size(0) * 0.7)

    train_activations = model_activations[:train_size].cpu()
    test_activations = model_activations[train_size:].cpu()
    normed_activations = train_activations / train_activations.norm(dim=2).unsqueeze(2)

    try:
        atoms = torch.load(f"data/{args.model}/atoms.pt")
        atom_indices = torch.load(f"data/{args.model}/atom_indices.pt")
        print("Found atoms so not training")
    except FileNotFoundError:
        atoms, atom_indices, losses = construct_atoms(
            normed_activations,
            ito_fn=omp_pytorch if args.ito_fn == OMP else gp_pytorch,
            batch_size=args.batch_size,
            l0=args.l0,
        )
        print(f"Trained a model with {len(atoms)} atoms")

        torch.save(atoms, f"data/{args.model}/atoms.pt")
        torch.save(atom_indices, f"data/{args.model}/atom_indices.pt")
        with open(f"data/{args.model}/losses.pkl", "wb") as f:
            pickle.dump(losses, f)

    try:
        losses = torch.load(f"data/{args.model}/{args.ito_fn}_losses.pt")
        activations = torch.load(f"data/{args.model}/{args.ito_fn}_activations.pt")
        indices = torch.load(f"data/{args.model}/{args.ito_fn}_indices.pt")
    except FileNotFoundError:
        atoms = atoms.to(normed_activations.device)
        losses, atoms, indices = evaluate(
            omp_pytorch if args.ito_fn == OMP else gp_pytorch,
            atoms,
            test_activations,
            args.l0,
            batch_size=32,
            return_activations=True,
        )
        torch.save(losses, f"data/{args.model}/{args.ito_fn}_losses.pt")
        torch.save(activations, f"data/{args.model}/{args.ito_fn}_activations.pt")
        torch.save(indices, f"data/{args.model}/{args.ito_fn}_indices.pt")
