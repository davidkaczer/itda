#!/usr/bin/env python3

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
from dl_train import MultiLayerITDATrainer, run_training_loop
from get_model_activations_transformerlens import get_activations_tl
from get_model_activations import get_activations_hf
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer

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


def get_layered_runs_for_models(model_names, layer_indices, entity="your-entity", project="example_saes"):
    """Fetch finished W&B runs for each (model, layer) combination."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"tags": "layer_similarity"})

    runs_dict = {
        model_name: {layer: None for layer in layer_indices}
        for model_name in model_names
    }

    for run in runs:
        if run.state != "finished":
            continue

        layer = run.config.get("layer")
        run_model_name = run.config.get("model", "")

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


def load_activation_dataset(model, layer, activations_base_path, dataset_name, seq_len, batch_size, device, num_examples, num_layers):
    """
    Load or generate the activation dataset from disk. If it doesn't exist,
    generate using `get_activations_tl()`.
    """
    activations_path = f"{activations_base_path}/{model}/{dataset_name}"
    if not os.path.exists(activations_path):
        os.makedirs(activations_path, exist_ok=True)
        get_activations_tl(
            model_name=model,
            dataset_name=dataset_name,
            activations_path=activations_path,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            num_examples=num_examples,
            layers_str=list(range(num_layers)),
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
    W = X_pinv @ Y                 # shape (d_X, d_Y)
    Y_pred = X @ W                 # shape (N, d_Y)

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


def main():
    """
    Main entry point for the script. This will:
    1. Parse command-line arguments.
    2. Potentially train new ITDA dictionaries for each layer if not found.
    3. Compute ITDA-based similarities if needed.
    4. Compute CKA, SVCCA, and linear regression R^2 similarities if needed.
    5. Print and plot results.
    """
    parser = argparse.ArgumentParser(description="Run the ITDA and similarity measurements.")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        for layer in range(1, NUM_LAYERS):
            if (model_name in existing_itdas) and existing_itdas[model_name][layer]:
                print(f"Skipping layer {layer} for model {model_name} (already done).")
                continue

            print(f"Training ITDA for model={model_name}, layer={layer} ...")
            trainer_cfg = {
                "activation_dim": ACTIVATION_DIM,
                "k": 40,
                "loss_threshold": TARGET_LOSS,
                "layers": list(range(1, NUM_LAYERS)),
                "lm_name": model_name,
                "device": device,
                "steps": NUM_EXAMPLES // BATCH_SIZE,
                "dataset": DATASET,
                "seq_len": SEQ_LEN,
                "seed": 0,
            }
            trainer = MultiLayerITDATrainer(**trainer_cfg)

            dataset = load_dataset(DATASET, split="train", streaming=True)
            data_stream = (item["text"] for item in dataset)

            model = HookedTransformer.from_pretrained(model_name, device=device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            run_training_loop(
                trainer=trainer,
                data_stream=data_stream,
                tokenizer=tokenizer,
                model=model,
                max_steps=trainer_cfg["steps"],
                batch_size=BATCH_SIZE,
                seq_len=trainer_cfg["seq_len"],
                device=device,
                wandb_project=WANDB_PROJECT,
                wandb_entity=args.wandb_entity,
                wandb_tags=["layer_similarity"],
            )

    print("==== Finished (or skipped) all needed ITDA training. ====")

    # 3) Compute ITDA-based similarities if needed
    ITDA = "itda"
    similarities = {}

    def load_itda_similarities():
        """
        Either load from disk if present or compute from W&B artifacts.
        """
        itda_path = f"artifacts/similarities/{args.model_group}_{ITDA}.npy"
        if os.path.exists(itda_path):
            print(f"Loading ITDA similarities from {itda_path}...")
            return np.load(itda_path)

        # If not found, compute from artifacts
        print("Computing ITDA similarities from W&B artifacts...")
        api = wandb.Api()
        atom_indices = []
        for model in MODELS:
            atom_indices.append([])
            for layer in range(1, NUM_LAYERS):
                run = existing_itdas[model][layer]
                if not run:
                    atom_indices[-1].append(None)
                    continue

                artifact_ref = f"{WANDB_PROJECT}/ito_dictionary_{run.id}:latest"
                artifact = api.artifact(artifact_ref, type="model")
                artifact_dir = artifact.download()
                atom_indices_path = os.path.join(artifact_dir, "atom_indices.pt")
                loaded_indices = torch.load(atom_indices_path, weights_only=True).to("cpu").numpy()
                atom_indices[-1].append(loaded_indices)

        itda_sims = np.zeros((len(MODELS), len(MODELS), NUM_LAYERS - 1, NUM_LAYERS - 1))
        for mi, mj, li, lj in product(
            range(len(MODELS)),
            range(len(MODELS)),
            range(NUM_LAYERS - 1),
            range(NUM_LAYERS - 1),
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

    similarities[ITDA] = load_itda_similarities()

    # 4) Compute CKA, SVCCA, and linear reg R^2 similarities if needed
    CKA = "cka"
    SVCCA = "svcca"
    LINREG = "linreg_r2"

    # Initialize holders
    for measure in [CKA, SVCCA, LINREG]:
        shape = (len(MODELS), len(MODELS), NUM_LAYERS - 1, NUM_LAYERS - 1)
        similarities[measure] = np.zeros(shape)

    # We'll check if each measure is already saved. If so, load it; otherwise compute.
    def compute_or_load_measure(measure):
        out_path = f"artifacts/similarities/{args.model_group}_{measure}.npy"
        if os.path.exists(out_path):
            print(f"Loading {measure.upper()} from {out_path}...")
            return np.load(out_path)

        print(f"Computing {measure.upper()} similarities...")
        # Pre-load activations for model_i to avoid repeated disk access
        for mi, model_i in enumerate(MODELS):
            # Load all layers' activations once for model_i
            model_i_activations = []
            for li in range(1, NUM_LAYERS):
                ai_cpu = load_activation_dataset(
                    model_i, li,
                    activations_base_path=args.activations_base_path,
                    dataset_name=DATASET,
                    seq_len=SEQ_LEN,
                    batch_size=BATCH_SIZE,
                    device=device,
                    num_examples=NUM_EXAMPLES,
                    num_layers=NUM_LAYERS,
                )
                model_i_activations.append(ai_cpu)

            for mj, model_j in enumerate(MODELS):
                # For each layer in j, load on the fly
                for lj in range(1, NUM_LAYERS):
                    aj_cpu = load_activation_dataset(
                        model_j, lj,
                        activations_base_path=args.activations_base_path,
                        dataset_name=DATASET,
                        seq_len=SEQ_LEN,
                        batch_size=BATCH_SIZE,
                        device=device,
                        num_examples=NUM_EXAMPLES,
                        num_layers=NUM_LAYERS,
                    )
                    aj_gpu = aj_cpu.to(device)

                    for li in range(1, NUM_LAYERS):
                        ai_cpu = model_i_activations[li - 1]
                        ai_gpu = ai_cpu.to(device)

                        if measure == SVCCA:
                            sim = svcca_torch(ai_gpu, aj_gpu)
                        elif measure == CKA:
                            sim = linear_cka_torch(ai_gpu, aj_gpu)
                        elif measure == LINREG:
                            sim = linear_regression_r2_torch(ai_gpu, aj_gpu)
                        else:
                            raise ValueError(f"Unknown measure: {measure}")

                        similarities[measure][mi, mj, li - 1, lj - 1] = sim

                        del ai_gpu
                        torch.cuda.empty_cache()

                    del aj_gpu
                    torch.cuda.empty_cache()

            del model_i_activations
            torch.cuda.empty_cache()

            # Save partial results after finishing each model_i
            os.makedirs("artifacts/similarities", exist_ok=True)
            np.save(out_path, similarities[measure])

        return similarities[measure]

    # Actually run the computations for each measure
    for measure in [CKA, SVCCA, LINREG]:
        similarities[measure] = compute_or_load_measure(measure)

    # 5) Simple evaluation of alignment across diagonal layers
    print("\n==== Accuracy of layer alignment by argmax similarity ====")
    target = np.arange(NUM_LAYERS - 1).reshape(1, 1, NUM_LAYERS - 1)
    target = target + np.zeros((len(MODELS), len(MODELS), 1), dtype=int)

    for measure in [ITDA, SVCCA, CKA, LINREG]:
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
    }

    for measure in [SVCCA, CKA, ITDA, LINREG]:
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
        # If you want to save instead of show, do:
        # plt.savefig(f"artifacts/similarities/{measure}_{args.model_group}.png")
        plt.show()


if __name__ == "__main__":
    main()