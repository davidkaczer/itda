# %%
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from rich.text import Text
from sklearn.linear_model import orthogonal_mp
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from tqdm import tqdm

from meta_saes.sae import load_feature_splitting_saes

# %%

MAIN = __name__ == "__main__"

if MAIN:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, saes, token_dataset = load_feature_splitting_saes(
        device=device,
        saes_idxs=list(range(1, 9)),
    )
    torch.set_grad_enabled(False)

# %%
# test the hypothesis that SAE features relate to text patterns requiring a
# certain number of tokens.

BATCH_SIZE = 64
SEQ_LEN = 16


def sort_rows_by_weighted_mean(arr):
    indices = np.arange(arr.shape[1])
    weighted_means = (arr * indices).sum(axis=1)
    sorted_indices = np.argsort(weighted_means)
    return arr[sorted_indices]


if __name__ == "__main__":
    tokens = torch.stack([s["tokens"] for s in token_dataset])[:, :SEQ_LEN].to(device)

    try:
        model_activations = torch.load("data/model_activations.pt")
    except FileNotFoundError:
        model_activations = []
        for batch in tqdm(torch.split(tokens, BATCH_SIZE), desc="Model"):
            model_activations.append(
                model.run_with_cache(batch)[1][saes[0].cfg.hook_name]
            )
        model_activations = torch.cat(model_activations)
        torch.save(model_activations, "data/model_activations.pt")
    print(model_activations.shape)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i, sae in enumerate(saes):
        try:
            mean_acts = torch.load(
                f"data/mean_acts_{sae.W_dec.size(0)}.pt", weights_only=True
            )
        except FileNotFoundError:
            mean_acts = torch.zeros(SEQ_LEN, sae.W_dec.size(0))
            for model_acts in tqdm(
                torch.split(model_activations, BATCH_SIZE), desc=f"SAE {i+1}"
            ):
                sae_acts = (sae.encode(model_acts) > 0).float().sum(dim=0).cpu()
                mean_acts += sae_acts
            mean_acts /= len(tokens)
            torch.save(mean_acts, f"data/mean_acts_{sae.W_dec.size(0)}.pt")

        normed = mean_acts / mean_acts.sum(dim=0)
        ordered_rows = sort_rows_by_weighted_mean(normed.T)

        qt = QuantileTransformer(output_distribution="uniform", random_state=42)
        transformed_data = qt.fit_transform(ordered_rows)

        ax = axs[i // 4, i % 4]
        ax.imshow(transformed_data, aspect="auto")
        ax.set_title(f"SAE {i+1}")

    plt.tight_layout()
    plt.show()

# %%


def omp_pytorch(D, x, n_nonzero_coefs):
    batch_size, n_features = x.shape
    n_atoms = D.shape[0]
    indices = torch.zeros(
        (batch_size, n_nonzero_coefs), dtype=torch.long, device=D.device
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
    plot_update_interval=10,
    batch_counter=0,
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
    plt.savefig("data/losses_and_ratio.png")


if MAIN:
    warnings.filterwarnings("ignore")

    train_activations = model_activations[:10000]
    test_activations = model_activations[10000:]

    omp_l0 = 8

    try:
        atoms = torch.load("data/atoms.pt")
        atom_indices = torch.load("data/atom_indices.pt")
    except FileNotFoundError:
        normed_activations = train_activations / train_activations.norm(
            dim=2
        ).unsqueeze(2)
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
            .reshape(-1, saes[0].W_dec.size(1))
        )

        losses = []
        ratio_history = []

        pbar = tqdm(total=remaining_activations.size(0))
        batch_size = 1024
        plot_update_interval = 1
        batch_counter = 0
        while remaining_activations.size(0) > 0:
            batch_activations = remaining_activations[:batch_size]
            remaining_activations = remaining_activations[batch_size:]

            coefs, indices = omp_pytorch(atoms, batch_activations, omp_l0)
            selected_atoms = atoms[indices]
            recon = torch.bmm(coefs.unsqueeze(1), selected_atoms).squeeze(1)
            loss = ((batch_activations - recon) ** 2).sum(dim=1)
            loss = torch.clamp(loss, 0, 1)
            losses.extend(loss.cpu().tolist())

            mask = loss > 0.2
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

        torch.save(atoms, "data/atoms.pt")
        torch.save(atom_indices, "data/atom_indices.pt")
        with open("data/losses.pkl", "wb") as f:
            pickle.dump(losses, f)


# %%

if MAIN:
    sae = saes[-1]

    try:
        omp_losses = pickle.load(open(f"data/omp_losses.pkl", "rb"))
        sae_losses = pickle.load(open(f"data/sae_losses.pkl", "rb"))
        omp_activations = torch.load(f"data/omp_activations.pt")
        omp_indices = torch.load(f"data/omp_indices.pt")
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        print(e)
        BATCH_SIZE = 32
        omp_losses = []
        omp_indices = []
        omp_activations = []
        sae_losses = []
        for batch in tqdm(torch.split(test_activations, BATCH_SIZE)):
            flattened_batch = batch.flatten(end_dim=1)

            omp_batch = flattened_batch / flattened_batch.norm(dim=1).unsqueeze(1)
            coefs, indices = omp_pytorch(atoms, omp_batch, omp_l0)
            omp_activations.extend(coefs.cpu().tolist())
            omp_indices.extend(indices.cpu().tolist())
            selected_atoms = atoms[indices]
            omp_recon = torch.bmm(coefs.unsqueeze(1), selected_atoms).squeeze(1)

            loss = ((omp_batch - omp_recon) ** 2).mean(dim=1)
            # can't reconstruct some? removing for now to investigate later
            loss = loss[loss < 1]
            omp_losses.extend(loss.cpu().tolist())

            sae_recon = sae(flattened_batch)
            loss = (
                (
                    flattened_batch / flattened_batch.norm(dim=1).unsqueeze(1)
                    - sae_recon / sae_recon.norm(dim=1).unsqueeze(1)
                )
                ** 2
            ).mean(dim=1)
            sae_losses.extend(loss.cpu().tolist())

        omp_activations = np.array(omp_activations)
        omp_indices = np.array(omp_indices)

        with open(f"data/omp_losses.pkl", "wb") as f:
            pickle.dump(omp_losses, f)
        with open(f"data/sae_losses.pkl", "wb") as f:
            pickle.dump(sae_losses, f)
        torch.save(omp_activations, f"data/omp_activations.pt")
        torch.save(omp_indices, f"data/omp_indices.pt")

    omp_loss = np.mean(omp_losses)
    sae_loss = np.mean(sae_losses)
    print(f"OMP Loss: {omp_loss}")
    print(f"SAE Loss: {sae_loss}")

# %%


def highlight_token_to_string(tokens, indices, activations):
    if activations[0] == 0.:
        return
    highlighted_text = Text()

    for i, token in enumerate(tokens):
        token_str = model.to_string([token])
        if i in indices:
            activation_index = indices.index(i)
            activation_value = activations[activation_index]
            if token_str.isspace():
                token_str = "_"
            highlighted_text.append(
                f"<<<{token_str}>>>[{activation_value:.2f}]", style="bold bright_yellow"
            )
        else:
            highlighted_text.append(f"{token_str}")

    print(highlighted_text)

    return highlighted_text.plain


if __name__ == "__main__":
    sae = saes[0]

    # uncomment to use the SAE instead
    # omp_indices = []
    # omp_activations = []
    # for batch in torch.split(test_activations, BATCH_SIZE):
    #     a = sae.encode(batch)
    #     omp_activations.append(a.topk(omp_l0).values)
    #     omp_indices.append(a.topk(omp_l0).indices)
    # omp_activations = torch.cat(omp_activations).cpu().numpy()
    # omp_indices = torch.cat(omp_indices).cpu().numpy()

    reshaped_indices = omp_indices.reshape(
        test_activations.size(0), test_activations.size(1), omp_l0
    )
    reshaped_activations = omp_activations.reshape(
        test_activations.size(0), test_activations.size(1), omp_l0
    )
    result = np.where(reshaped_indices[:, :, :] == 0)
    activations = reshaped_activations[result[0], result[1], result[2]]

    # Sort by activation values in descending order
    sorted_indices = np.argsort(-activations)
    sorted_input_indices = result[0][sorted_indices]
    sorted_token_indices = result[1][sorted_indices]
    sorted_activations = activations[sorted_indices]

    limit = 0
    for input_idx, token_idx, activation in zip(
        sorted_input_indices, sorted_token_indices, sorted_activations
    ):
        highlight_token_to_string(tokens[10_000 + input_idx], [token_idx], [activation])

        limit += 1
        if limit > 10:
            break
