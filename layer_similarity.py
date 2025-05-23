"""
Get layer similarity results using ITDAs and other similarity measures.
"""

import argparse
import os
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import zarr
from datasets import load_dataset
from dictionary_learning.dictionary import Dictionary
from example_saes.train import ITDATrainer, get_activation_dim, run_training_loop
from get_model_activations import get_activations_hf
from get_model_activations_transformerlens import get_activations_tl
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

import wandb

# Define GPT-2 model configurations
GPT2_MODELS = {
    "small": {
        "models": [
            "stanford-crfm/alias-gpt2-small-x21",
            "stanford-crfm/battlestar-gpt2-small-x49",
            "stanford-crfm/caprica-gpt2-small-x81",
            "stanford-crfm/darkmatter-gpt2-small-x343",
            "stanford-crfm/expanse-gpt2-small-x777",
        ],
        "layers": 12,
        "target_loss": 0.0004,
        "activation_dim": 768,
        "batch_size": 128,
    },
    "medium": {
        "models": [
            "stanford-crfm/arwen-gpt2-medium-x21",
            "stanford-crfm/beren-gpt2-medium-x49",
            "stanford-crfm/celebrimbor-gpt2-medium-x81",
            "stanford-crfm/durin-gpt2-medium-x343",
            "stanford-crfm/eowyn-gpt2-medium-x777",
        ],
        "layers": 24,
        "target_loss": 0.0004,
        "activation_dim": 1024,
        "batch_size": 64,
    },
}


def get_layered_runs_for_models(
    model_names,
    layer_indices,
    entity="your-entity",
    project="example_saes",
    tag="layer_similarity",
):
    """Fetch finished W&B runs for each (model, layer) combination."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="+created_at")

    runs_dict = {
        model_name: {layer: None for layer in layer_indices}
        for model_name in model_names
    }

    for run in runs:
        layer = run.config.get("layer")
        run_model_name = run.config.get("lm_name", "")

        if tag not in run.tags:
            continue
        if run.state != "finished":
            continue
        if layer not in layer_indices:
            continue
        if run_model_name not in model_names:
            continue

        runs_dict[run_model_name][layer] = run

    return runs_dict


def get_similarity_measure(ai1, ai2):
    """
    A simplistic measure that treats each row as a "tuple"
    and computes intersection/union of them. Not a typical
    measure, but included as in the original code.
    """
    ai1s = set([tuple(r) for r in ai1.tolist()])
    ai2s = set([tuple(r) for r in ai2.tolist()])

    return len(ai1s.intersection(ai2s)) / len(ai1s.union(ai2s))


def load_activation_dataset(
    model,
    layer,
    activations_base_path,
    dataset_name,
    seq_len,
    batch_size,
    device,
    num_examples,
    num_layers,
):
    """
    Load or generate the activation dataset from disk. If it doesn't exist,
    generate using `get_activations_tl()`.
    """
    activations_path = f"{activations_base_path}/{model}/{dataset_name}"
    layers = [str(l) for l in range(num_layers)]
    if not os.path.exists(activations_path):
        get_activations_tl(
            model_name=model,
            dataset_name=dataset_name,
            activations_path=activations_base_path,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            num_examples=num_examples,
            layers_str=", ".join(layers),
            tokenizer_name=model,
        )

    store = zarr.DirectoryStore(activations_path)
    zarr_group = zarr.group(store=store)
    zarr_activations = zarr_group[f"layer_{layer}"]

    # Flatten to (N, D) shape
    return torch.from_numpy(zarr_activations[:].reshape(-1, zarr_activations.shape[-1]))


def linear_cka_torch(X, Y, center_data=True, eps=1e-12):
    """
    Compute the linear CKA similarity between two sets of activations X and Y.
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."
    if center_data:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        X = X / (X.norm(dim=0, keepdim=True) + eps)
        Y = Y / (Y.norm(dim=0, keepdim=True) + eps)

    numerator = (X.T @ Y).norm(p="fro").pow(2)
    denom_x = (X.T @ X).norm(p="fro")
    denom_y = (Y.T @ Y).norm(p="fro")
    denominator = torch.maximum(denom_x * denom_y, torch.tensor(eps, device=X.device))

    cka_value = numerator / denominator
    return cka_value.item()


def svcca_torch(X, Y, num_components=20, eps=1e-12):
    """
    Compute an SVCCA similarity between two sets of activations X and Y.
    """
    # 1. Mean-center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # 2. Low-rank SVD to reduce dimensionality
    Ux, Sx, Vx = torch.svd_lowrank(X, q=num_components)
    Uy, Sy, Vy = torch.svd_lowrank(Y, q=num_components)

    X_red = Ux * Sx
    Y_red = Uy * Sy

    # 3. CCA on reduced data
    Cxx = X_red.T @ X_red
    Cyy = Y_red.T @ Y_red
    Cxy = X_red.T @ Y_red

    I_x = eps * torch.eye(Cxx.shape[0], device=X.device)
    I_y = eps * torch.eye(Cyy.shape[0], device=Y.device)

    Cxx_inv = torch.linalg.pinv(Cxx + I_x)
    Cyy_inv = torch.linalg.pinv(Cyy + I_y)

    M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
    eigvals, _ = torch.linalg.eigh(M)
    eigvals = eigvals.relu()

    eigvals_sorted, _ = torch.sort(eigvals, descending=True)
    canonical_corrs = torch.sqrt(eigvals_sorted)

    k = min(num_components, canonical_corrs.shape[0])
    svcca_value = canonical_corrs[:k].mean()
    return svcca_value.item()


def linear_regression_r2_torch(X, Y, eps=1e-12):
    """
    Compute how well Y can be linearly predicted from X (single global R^2).
    X: (N, d_X) tensor
    Y: (N, d_Y) tensor
    Returns: scalar float R^2 in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples."

    # Solve least squares: W = pinv(X) * Y
    X_pinv = torch.linalg.pinv(X)  # shape (d_X, N)
    W = X_pinv @ Y  # shape (d_X, d_Y)
    Y_pred = X @ W  # shape (N, d_Y)

    # SSE = sum of squared errors
    SSE = (Y - Y_pred).pow(2).sum()

    # TSS = total sum of squares around mean(Y)
    Y_mean = Y.mean(dim=0, keepdim=True)
    TSS = (Y - Y_mean).pow(2).sum()

    if TSS < eps:
        # If Y is constant (or near-constant), define R^2 = 1 if SSE is also near 0, else 0
        return 1.0 if SSE < eps else 0.0

    R2 = 1.0 - (SSE / TSS)
    return R2.item()


def get_atoms_from_wandb_run(run, wandb_project):
    api = wandb.Api()
    artifact_refs = [a.name for a in run.logged_artifacts()]
    artifact_refs = [a for a in artifact_refs if "ITDA" in a]
    if len(artifact_refs) != 1:
        raise ValueError("Expected exactly one artifact in the run.")
    artifact_refs = f"{wandb_project}/{artifact_refs[0]}"
    artifact = api.artifact(artifact_refs, type="model")
    artifact_dir = artifact.download()

    atom_indices_path = os.path.join(artifact_dir, "atom_indices.pt")
    indices = torch.load(atom_indices_path, weights_only=True).to("cpu").numpy()

    atoms_path = os.path.join(artifact_dir, "atoms.pt")
    atoms = torch.load(atoms_path, weights_only=True).to("cpu").numpy()

    return atoms, indices


def load_itda_similarities(
    models,
    existing_itdas,
    wandb_project,
    model_group,
    num_layers,
    itda_label="itda",
):
    """
    Load or compute ITDA-based similarities (intersection-over-union of learned atom indices).
    If the .npy file is found, load it; otherwise compute from W&B artifacts.
    """
    import wandb

    itda_path = f"artifacts/similarities/{model_group}_{itda_label}.npy"
    if os.path.exists(itda_path):
        print(f"Loading ITDA similarities from {itda_path}...")
        return np.load(itda_path)

    # Not found; compute from W&B artifacts
    print("Computing ITDA similarities from W&B artifacts...")

    atom_indices = []
    for model in models:
        atom_indices.append([])
        for layer in range(1, num_layers):
            run = existing_itdas[model][layer]
            if not run:
                atom_indices[-1].append(None)
                continue

            try:
                loaded_indices = get_atoms_from_wandb_run(run, wandb_project)[1]
            except ValueError as e:
                raise ValueError(
                    f"Error loading atom indices for {model} layer {layer}: {str(e)}"
                )
            atom_indices[-1].append(loaded_indices)

    # Shape: (len(models), len(models), num_layers-1, num_layers-1)
    itda_sims = np.zeros((len(models), len(models), num_layers - 1, num_layers - 1))
    for mi, mj, li, lj in product(
        range(len(models)),
        range(len(models)),
        range(num_layers - 1),
        range(num_layers - 1),
    ):
        # If either is None, similarity is 0 or unknown
        if (atom_indices[mi][li] is None) or (atom_indices[mj][lj] is None):
            itda_sims[mi, mj, li, lj] = 0.0
        else:
            itda_sims[mi, mj, li, lj] = get_similarity_measure(
                atom_indices[mi][li],
                atom_indices[mj][lj],
            )

    os.makedirs("artifacts/similarities", exist_ok=True)
    np.save(itda_path, itda_sims)
    return itda_sims


def relative_similarity(X, Y, num_anchors=300, seed=42, batch_size=512):
    """
    Compute latent space similarity using relative representations in a batched manner
    to avoid out-of-memory (OOM) issues.

    Args:
        X (torch.Tensor): Embeddings from model X [num_samples, embedding_dim].
        Y (torch.Tensor): Embeddings from model Y [num_samples, embedding_dim].
        num_anchors (int): Number of anchors to sample.
        seed (int): Random seed for reproducibility.
        batch_size (int): Chunk size for batched computation.

    Returns:
        float: Mean cosine similarity between relative embeddings.
    """
    assert X.shape == Y.shape, "X and Y must have the same shape."

    # Set the random generator for reproducible anchor sampling
    generator = torch.Generator().manual_seed(seed)
    anchor_indices = torch.randperm(X.shape[0], generator=generator)[:num_anchors].to(
        X.device
    )

    # Extract the anchor points
    anchors_X = X[anchor_indices]
    anchors_Y = Y[anchor_indices]

    total_sim_sum = 0.0
    total_count = 0

    # Process X and Y in chunks
    for i in range(0, X.shape[0], batch_size):
        X_chunk = X[i : i + batch_size]
        Y_chunk = Y[i : i + batch_size]

        # Cosine similarity of each chunk to the anchors
        X_chunk_rel = F.cosine_similarity(
            X_chunk[:, None, :], anchors_X[None, :, :], dim=-1
        )
        Y_chunk_rel = F.cosine_similarity(
            Y_chunk[:, None, :], anchors_Y[None, :, :], dim=-1
        )

        # Now compute the similarity between these relative representations
        chunk_sim = F.cosine_similarity(X_chunk_rel, Y_chunk_rel, dim=-1)

        total_sim_sum += chunk_sim.sum().item()
        total_count += chunk_sim.size(0)

    return total_sim_sum / total_count


def compute_or_load_measure(
    measure,
    models,
    num_layers,
    dataset,
    device,
    batch_size,
    seq_len,
    num_examples,
    model_group,
    activations_base_path,
):
    """
    Compute or load the similarity measure (CKA, SVCCA, or linreg R^2) for each
    (model_i, model_j, layer_i, layer_j). Returns a 4D numpy array of shape:
    (len(models), len(models), num_layers-1, num_layers-1).

    If a cached .npy file is found, loads it. Otherwise, computes from scratch
    using the activation datasets on disk (or generating them if missing).
    """
    measure_path = f"artifacts/similarities/{model_group}_{measure}.npy"
    if os.path.exists(measure_path):
        print(f"Loading {measure.upper()} from {measure_path}...")
        return np.load(measure_path)

    print(f"Computing {measure.upper()} similarities...")

    # Prepare an empty array for this measure
    measure_array = np.zeros((len(models), len(models), num_layers - 1, num_layers - 1))

    # Calculate total iterations for tqdm (mi, mj, li, lj)
    total_iterations = len(models) * len(models) * (num_layers - 1) * (num_layers - 1)

    with tqdm(total=total_iterations, desc=f"Computing {measure.upper()}") as pbar:
        for mi, model_i in enumerate(models):
            for mj, model_j in enumerate(models):
                for li in range(1, num_layers):
                    # Load only the current layer li for model_i
                    ai_cpu = load_activation_dataset(
                        model_i,
                        li,
                        activations_base_path=activations_base_path,
                        dataset_name=dataset,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        device=device,
                        num_examples=num_examples,
                        num_layers=num_layers,
                    )
                    ai_gpu = ai_cpu.to(device)[:100000]

                    for lj in range(1, num_layers):
                        # Load only the current layer lj for model_j
                        aj_cpu = load_activation_dataset(
                            model_j,
                            lj,
                            activations_base_path=activations_base_path,
                            dataset_name=dataset,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            device=device,
                            num_examples=num_examples,
                            num_layers=num_layers,
                        )
                        aj_gpu = aj_cpu.to(device)[:100000]

                        # Compute the requested similarity measure
                        if measure == SVCCA:
                            sim = svcca_torch(ai_gpu, aj_gpu)
                        elif measure == CKA:
                            sim = linear_cka_torch(ai_gpu, aj_gpu)
                        elif measure == LINREG:
                            sim = linear_regression_r2_torch(ai_gpu, aj_gpu)
                        elif measure == RELATIVE:
                            sim = relative_similarity(ai_gpu, aj_gpu)
                        else:
                            raise ValueError(f"Unknown measure: {measure}")

                        # Store the result
                        measure_array[mi, mj, li - 1, lj - 1] = sim

                        # Free memory for this layer comparison
                        del aj_gpu, aj_cpu
                        torch.cuda.empty_cache()

                        # Update the progress bar
                        pbar.update(1)

                    # Free model_i's layer li activation
                    del ai_gpu, ai_cpu
                    torch.cuda.empty_cache()

                # (Optional) Save partial results after finishing model_j
                # in case the run is long and you want to checkpoint
                os.makedirs("artifacts/similarities", exist_ok=True)
                np.save(measure_path, measure_array)

    return measure_array


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(
        description="Run the ITDA and similarity measurements."
    )
    parser.add_argument(
        "--model_group",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Choose which GPT-2 model group to use ('small' or 'medium').",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="patrickaaleask",
        help="Entity (user or team) under which the W&B project lives.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="itda",
        help="Name of the W&B project for logging or fetching runs.",
    )
    parser.add_argument(
        "--activations_base_path",
        type=str,
        default="artifacts/data",
        help="Base path for storing/loading activation data (zarr).",
    )
    args = parser.parse_args()

    # Select model config based on the chosen group
    config = GPT2_MODELS[args.model_group]

    MODELS = config["models"]
    NUM_LAYERS = config["layers"]
    TARGET_LOSS = config["target_loss"]
    ACTIVATION_DIM = config["activation_dim"]
    BATCH_SIZE = config["batch_size"]

    # Hard-coded dataset and other hyperparams (from original notebook)
    DATASET = "NeelNanda/pile-10k"
    SEQ_LEN = 128
    NUM_EXAMPLES = 10_000
    WANDB_PROJECT = args.wandb_project  # e.g. "itda"

    # 1) Check existing runs (ITDA) on W&B
    existing_itdas = get_layered_runs_for_models(
        MODELS,
        list(range(1, NUM_LAYERS)),
        entity=args.wandb_entity,
        project=WANDB_PROJECT,
    )

    # 2) Train new ITDA dictionaries if needed
    print("==== Checking existing ITDA runs and training if needed... ====")
    for model_name in MODELS:
        if model_name not in existing_itdas:
            train_layers = list(range(1, NUM_LAYERS))
        else:
            train_layers = [
                layer
                for layer in range(1, NUM_LAYERS)
                if not existing_itdas[model_name][layer]
            ]
            if len(train_layers) == 0:
                continue

        print(f"Training ITDA for model={model_name}, layers={train_layers} ...")
        trainers = []
        for layer_idx in train_layers:
            trainer = ITDATrainer(
                layer=layer_idx,
                steps=NUM_EXAMPLES // BATCH_SIZE,
                activation_dim=model.cfg.d_model,
                k=40,
                loss_threshold=TARGET_LOSS,
                device=device,
                dataset=DATASET,
                seq_len=SEQ_LEN,
                batch_size=BATCH_SIZE,
                seed=0,
            )
            trainers.append(trainer)

        dataset = load_dataset(DATASET, split="train", streaming=True)
        data_stream = (item["text"] for item in dataset)

        model = HookedTransformer.from_pretrained(model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        run_training_loop(
            trainers=trainers,
            data_stream=data_stream,
            tokenizer=tokenizer,
            model=model,
            max_steps=NUM_EXAMPLES // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
            wandb_project=WANDB_PROJECT,
            wandb_entity=args.wandb_entity,
            wandb_tags=["layer_similarity"],
        )

    print("==== Finished (or skipped) all needed ITDA training. ====")

    # 3) Compute ITDA-based similarities if needed
    ITDA = "itda"
    similarities = {}

    similarities[ITDA] = load_itda_similarities(
        models=MODELS,
        existing_itdas=existing_itdas,
        wandb_project=WANDB_PROJECT,
        model_group=args.model_group,
        num_layers=NUM_LAYERS,
        itda_label=ITDA,
    )

    # 4) Compute CKA, SVCCA, and linear reg R^2 similarities if needed
    CKA = "cka"
    SVCCA = "svcca"
    LINREG = "linreg_r2"
    RELATIVE = "relative"

    # Prepare placeholders in a dictionary for each measure
    for measure in [CKA, SVCCA, LINREG, RELATIVE]:
        similarities[measure] = compute_or_load_measure(
            measure=measure,
            models=MODELS,
            num_layers=NUM_LAYERS,
            dataset=DATASET,
            device=device,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_examples=NUM_EXAMPLES,
            model_group=args.model_group,
            activations_base_path=args.activations_base_path,
        )

    # 5) Simple evaluation of alignment across diagonal layers
    print("\n==== Accuracy of layer alignment by argmax similarity ====")
    target = np.arange(NUM_LAYERS - 1).reshape(1, 1, NUM_LAYERS - 1)
    target = target + np.zeros((len(MODELS), len(MODELS), 1), dtype=int)

    for measure in [ITDA, SVCCA, CKA, LINREG, RELATIVE]:
        sims = similarities[measure].argmax(axis=-1)
        accuracy = (target == sims).sum() / target.size
        print(f"{measure.upper()} accuracy: {accuracy:.4f}")

    # 6) Plot the average similarity across (model_i, model_j)
    print("\n==== Plotting average similarity maps ====")
    measure_titles = {
        ITDA: "ITDA",
        SVCCA: "SVCCA",
        CKA: "Linear CKA",
        LINREG: "Linear Regression R²",
        RELATIVE: "Relative Similarity",
    }

    for measure in [SVCCA, CKA, ITDA, LINREG, RELATIVE]:
        heatmap = similarities[measure].mean(axis=(0, 1))
        plt.figure(figsize=(4, 4))
        im = plt.imshow(heatmap, cmap="viridis")
        plt.title(measure_titles[measure])
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Similarity", rotation=270, labelpad=15)

        plt.xlabel("Layer")
        plt.ylabel("Layer")

        ticks = np.arange(heatmap.shape[0])
        plt.xticks(ticks, ticks + 1)
        plt.yticks(ticks, ticks + 1)

        plt.tight_layout()
        plt.savefig(f"artifacts/similarities/{measure}_{args.model_group}.png")
