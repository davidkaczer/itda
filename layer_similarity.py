# %%
import argparse
import os
from itertools import product
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from get_model_activations import get_activations_hf
from tqdm import tqdm
from train import train_ito_saes

import wandb

PYTHIA_MODELS = [
    "pleask/pythia-14mish-seed_1234",
    "pleask/pythia-14mish-seed_1337",
    "pleask/pythia-14mish-seed_2023",
    "pleask/pythia-14mish-seed_42",
    "pleask/pythia-14mish-seed_9999",
]
PYTHIA_LAYERS = 6
PYTHIA_TARGET_LOSS = 0.006

GPT2_MODELS = [
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
]
GPT2_LAYERS = 12
GPT2_TARGET_LOSS = 0.0005

USE_GPT2 = True
if USE_GPT2:
    MODELS = GPT2_MODELS
    NUM_LAYERS = GPT2_LAYERS
    TARGET_LOSS = GPT2_TARGET_LOSS
else:
    MODELS = PYTHIA_MODELS
    NUM_LAYERS = PYTHIA_LAYERS
    TARGET_LOSS = PYTHIA_TARGET_LOSS

DATASET = "NeelNanda/pile-10k"

# %%


def get_layered_runs_for_models(
    model_names, layer_indices, entity="your-entity", project="example_saes"
):
    """
    Query W&B for runs that have the tag 'layer_convergence' in the specified
    entity/project. Return a dictionary:
        model_name -> layer -> run
    containing the latest (highest-step) finished run for each model-layer pair.

    Args:
        model_names (list of str): List of model name strings to match.
        layer_indices (list of int): List of layer indices to match.
        entity (str): The W&B entity (team/user).
        project (str): The W&B project name.

    Returns:
        dict: A nested dictionary of the form:
              {
                  model_name: {
                      layer_index: run,
                      ...
                  },
                  ...
              }
    """
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}", filters={"tags": "layer_similarity"})

    # Initialize a structure to store the best (highest-step) run for each (model_name, layer)
    runs_dict = {
        model_name: {layer: None for layer in layer_indices}
        for model_name in model_names
    }

    for run in runs:
        # Only consider runs that finished successfully
        if run.state != "finished":
            continue

        # Extract layer and model name from run.config
        layer = run.config.get("layer")
        run_model_name = run.config.get("model", "")

        # We only care about the specified layer_indices and model_names
        if layer not in layer_indices:
            continue
        if run_model_name not in model_names:
            continue

        runs_dict[run_model_name][layer] = run

    return runs_dict


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    existing_itdas = get_layered_runs_for_models(
        MODELS,
        list(range(1, NUM_LAYERS)),
        entity="patrickaaleask",
        project="example_saes",
    )

    for model in MODELS:
        if not os.path.exists(f"artifacts/data/{model}"):
            get_activations_hf(
                model_name=model,
                dataset_name="NeelNanda/pile-10k",
                activations_path=f"artifacts/data",
                seq_len=128,
                batch_size=256,
                device=device,
                num_examples=10_000,
                revision=None,
                tokenizer_name="openai-community/gpt2",
            )

        for layer in range(1, NUM_LAYERS):
            if (model in existing_itdas) and existing_itdas[model][layer]:
                print(f"Skipping layer {layer} for model {model}.")
                continue

            args_dict = {
                "method": "ito",
                "model": model,
                "dataset": "NeelNanda/pile-10k",
                "layer": layer,
                "batch_size": 32,
                "target_loss": TARGET_LOSS,
                "max_sequences": None,
                "load_run_id": None,
                "l0": 40,
                "seq_len": 128,
                "skip_validation": True,
                "max_atoms": 200_000,
            }
            args = argparse.Namespace(**args_dict)

            wandb.init(
                project="example_saes", config=vars(args), tags=["layer_similarity"]
            )
            train_ito_saes(args, device)
            wandb.finish()

# %%


def get_similarity_measure(ai1, ai2):
    ai1s = set([tuple(r) for r in ai1.tolist()])
    ai2s = set([tuple(r) for r in ai2.tolist()])

    return len(ai1s.intersection(ai2s)) / len(ai1s.union(ai2s))


def compute_similarity_task(task):
    mi, mj, li, lj, atom_indices = task
    return (
        mi,
        mj,
        li,
        lj,
        get_similarity_measure(atom_indices[mi][li], atom_indices[mj][lj]),
    )


if __name__ == "__main__":
    atom_indices = []
    for model in MODELS:
        atom_indices.append([])
        for layer in range(1, NUM_LAYERS):
            run = existing_itdas[model][layer]
            if not run:
                continue

            atom_indices[-1].append(
                torch.load(
                    f"artifacts/runs/{run.id}/atom_indices.pt", weights_only=True
                )
                .to("cpu")
                .numpy()
            )

    itda_similarities = np.zeros(
        (len(MODELS), len(MODELS), NUM_LAYERS - 1, NUM_LAYERS - 1)
    )

    # Create a list of tasks to distribute across processes
    tasks = [
        (mi, mj, li, lj, atom_indices)
        for mi, mj, li, lj in product(
            range(len(MODELS)),
            range(len(MODELS)),
            range(NUM_LAYERS - 1),
            range(NUM_LAYERS - 1),
        )
    ]

    # Use multiprocessing to compute in parallel
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(compute_similarity_task, tasks),
                total=len(tasks),
                desc="Calculating similarities",
            )
        )

    # Populate the similarities array with results
    for mi, mj, li, lj, similarity in results:
        itda_similarities[mi, mj, li, lj] = similarity


# %%

if __name__ == "__main__":
    # Plot heatmap of the similarities
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    heatmap = itda_similarities.mean(axis=(0, 1))
    im = ax.imshow(heatmap, cmap="viridis")

    # # Add values to the heatmap with dynamic text color
    # for i in range(heatmap.shape[0]):
    #     for j in range(heatmap.shape[1]):
    #         value = heatmap[i, j]
    #         # Use white text for dark colors and black for light colors
    #         text_color = "white" if value < 0.5 else "black"
    #         ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity", rotation=-90, va="bottom")
    cbar.set_ticks([heatmap.min(), heatmap.max()])

    # Label axes
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")

    # Update ticks to start from 1
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_yticks(np.arange(heatmap.shape[0]))
    ax.set_xticklabels(np.arange(1, heatmap.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, heatmap.shape[0] + 1))

    plt.show()

# %%


def load_activation_dataset(model, layer):
    activations_path = f"artifacts/data/{model}/{DATASET}"

    store = zarr.DirectoryStore(activations_path)
    zarr_group = zarr.group(store=store)
    zarr_activations = zarr_group[f"layer_{layer}"]

    return torch.from_numpy(zarr_activations[:].reshape(-1, zarr_activations.shape[-1]))


def linear_cka(X, Y, center_data=True):
    """
    Compute the linear CKA (Centered Kernel Alignment) between two sets of activations.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, d_x)
        Activations for one layer.
    Y : np.ndarray, shape (n_samples, d_y)
        Activations for another layer.
    center_data : bool
        If True, subtract the mean from each feature in X and Y before computing CKA.

    Returns
    -------
    cka_value : float
        The scalar CKA similarity between X and Y, typically in [0, 1].
    """
    # Check that the first dimension (number of samples) matches
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

    # Optionally mean-center each feature
    if center_data:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

    # Numerator: || X^T Y ||_F^2
    numerator = np.linalg.norm(X.T @ Y, ord="fro") ** 2

    # Denominator: || X^T X ||_F * || Y^T Y ||_F
    denom_x = np.linalg.norm(X.T @ X, ord="fro")
    denom_y = np.linalg.norm(Y.T @ Y, ord="fro")

    # To avoid divide-by-zero in degenerate cases, add a small epsilon if needed
    eps = 1e-12
    denominator = max(denom_x * denom_y, eps)

    cka_value = numerator / denominator
    return cka_value


def linear_cka_torch(X, Y, center_data=True, eps=1e-12):
    """
    Compute Linear CKA between two sets of activations, using PyTorch.
    This version will run on the GPU if X and Y are already on a CUDA device.

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, d_x)
        Activations for one layer.
    Y : torch.Tensor, shape (n_samples, d_y)
        Activations for another layer.
    center_data : bool
        If True, subtract the mean from each feature in X and Y before computing CKA.
    eps : float
        Small constant added to denominator to avoid divide-by-zero.

    Returns
    -------
    cka_value : float
        The scalar CKA similarity between X and Y, typically in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

    if center_data:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

    # Numerator: || X^T Y ||_F^2
    numerator = (X.T @ Y).norm(p="fro").pow(2)

    # Denominator: || X^T X ||_F * || Y^T Y ||_F
    denom_x = (X.T @ X).norm(p="fro")
    denom_y = (Y.T @ Y).norm(p="fro")
    denominator = torch.maximum(denom_x * denom_y, torch.tensor(eps, device=X.device))

    cka_value = numerator / denominator
    return cka_value.item()  # return as a Python float


def svcca(X, Y, num_components=20):
    """
    Compute SVCCA similarity between two sets of activations X and Y.
    1) Reduce dimensionality via SVD to 'num_components'.
    2) Perform canonical correlation analysis.
    3) Return the mean of the canonical correlations.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, d_x)
        Activations for one layer.
    Y : np.ndarray, shape (n_samples, d_y)
        Activations for another layer.
    num_components : int
        Number of principal components to keep before CCA.

    Returns
    -------
    svcca_value : float
        The mean canonical correlation across the 'num_components' directions.
        Typically in [0, 1].
    """
    # 1. Mean-center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # 2. Truncate to top principal components with SVD
    #    We'll do a truncated SVD by taking the top 'num_components' from the full SVD.
    Ux, Sx, Vx = np.linalg.svd(X, full_matrices=False)  # X = Ux * diag(Sx) * Vx
    Uy, Sy, Vy = np.linalg.svd(Y, full_matrices=False)  # Y = Uy * diag(Sy) * Vy

    # Keep top 'num_components'
    Ux_k = Ux[:, :num_components] * Sx[:num_components]
    Uy_k = Uy[:, :num_components] * Sy[:num_components]

    # Now X_reduced and Y_reduced are shape (n_samples, num_components)
    X_reduced = Ux_k
    Y_reduced = Uy_k

    # 3. Perform classical CCA on the reduced data.
    #    We'll compute the canonical correlations from the covariance matrices.
    #    The typical formula for the squared canonical correlations is the eigenvalues of:
    #         (X^T X)^-1 (X^T Y) (Y^T Y)^-1 (Y^T X)

    # Covariance-like matrices (note: these are "uncentered" because X_reduced, Y_reduced
    # are already centered).
    Cxx = X_reduced.T @ X_reduced
    Cyy = Y_reduced.T @ Y_reduced
    Cxy = X_reduced.T @ Y_reduced

    # Invert Cxx and Cyy (add small ridge if necessary to avoid singularities)
    eps = 1e-12
    Cxx_inv = np.linalg.pinv(Cxx + eps * np.eye(Cxx.shape[0]))
    Cyy_inv = np.linalg.pinv(Cyy + eps * np.eye(Cyy.shape[0]))

    # Matrix whose eigenvalues give us the squared canonical correlations
    M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T

    # Compute eigenvalues
    eigvals, _ = np.linalg.eigh(M)
    # Sort eigenvalues in descending order (largest first)
    eigvals = np.sort(eigvals)[::-1]

    # Canonical correlations are the sqrt of eigenvalues
    # (clip to avoid small negative numerical errors)
    canonical_corrs = np.sqrt(np.clip(eigvals, a_min=0.0, a_max=None))

    # The "number of canonical correlations" is limited by the smaller subspace dimension
    # We might only consider the top 'num_components' or the number that fits both subspaces.
    # Typically min(num_components, X_reduced.shape[1], Y_reduced.shape[1])
    # but X_reduced.shape[1] = num_components by construction, so:
    k = min(num_components, len(canonical_corrs))
    top_corrs = canonical_corrs[:k]

    # 4. Average the canonical correlations to get the SVCCA score
    svcca_value = np.mean(top_corrs)
    return svcca_value


def svcca_torch(X, Y, num_components=20, eps=1e-12):
    # 1. Mean-center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # 2. Partial SVD (low-rank SVD)
    Ux, Sx, Vx = torch.svd_lowrank(X, q=num_components)
    Uy, Sy, Vy = torch.svd_lowrank(Y, q=num_components)

    # Construct the reduced data
    # Ux: (n_samples, q), Sx: (q), Vx: (d_x, q)
    # X ≈ Ux @ diag(Sx) @ Vx.T
    X_reduced = Ux * Sx  # shape (n_samples, q)
    Y_reduced = Uy * Sy  # shape (n_samples, q)

    # 3. CCA on the reduced data
    Cxx = X_reduced.T @ X_reduced
    Cyy = Y_reduced.T @ Y_reduced
    Cxy = X_reduced.T @ Y_reduced

    I_x = eps * torch.eye(Cxx.shape[0], device=X.device)
    I_y = eps * torch.eye(Cyy.shape[0], device=Y.device)

    Cxx_inv = torch.linalg.pinv(Cxx + I_x)
    Cyy_inv = torch.linalg.pinv(Cyy + I_y)

    M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T

    # Eigenvalues => squared canonical correlations
    eigvals, _ = torch.linalg.eigh(M)
    eigvals = eigvals.relu()

    # Sort descending, take square roots
    eigvals_sorted, _ = torch.sort(eigvals, descending=True)
    canonical_corrs = torch.sqrt(eigvals_sorted)

    k = min(num_components, canonical_corrs.shape[0])
    svcca_value = canonical_corrs[:k].mean()

    return svcca_value.item()


CKA = "cka"
SVCCA = "svcca"
ITDA = "itda"

if __name__ == "__main__":
    similarities = {
        ITDA: itda_similarities,
        CKA: np.zeros((len(MODELS), len(MODELS), NUM_LAYERS - 1, NUM_LAYERS - 1)),
        SVCCA: np.zeros((len(MODELS), len(MODELS), NUM_LAYERS - 1, NUM_LAYERS - 1)),
    }

    measures = [SVCCA, CKA]
    for mi, mj, li, lj, measure in tqdm(
        product(
            range(len(MODELS)),
            range(len(MODELS)),
            range(1, NUM_LAYERS),
            range(1, NUM_LAYERS),
            measures,
        ),
        desc="Calculating similarities",
        total=len(MODELS) ** 2 * (NUM_LAYERS - 1) ** 2 * len(measures),
    ):
        ai1 = load_activation_dataset(MODELS[mi], li).to(device)
        ai2 = load_activation_dataset(MODELS[mj], lj).to(device)

        if measure == SVCCA:
            similarity = svcca_torch(ai1, ai2)
        elif measure == CKA:
            similarity = linear_cka_torch(ai1, ai2)
        else:
            raise ValueError(f"Unknown similarity measure: {measure}")

        similarities[measure][mi, mj, li - 1, lj - 1] = similarity

# %%

if __name__ == "__main__":
    target = np.arange(5).reshape(1, 1, 5) + np.zeros((5, 5, 1), dtype=int)
    for measure in [ITDA, SVCCA, CKA]:
        sims = similarities[measure].argmax(axis=-1)
        print(f"Accuracy for {measure}", (target == sims).sum() / target.size)
