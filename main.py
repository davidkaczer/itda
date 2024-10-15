# %%
import pickle
import random
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.text import Text
from sklearn.linear_model import orthogonal_mp
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from tqdm import tqdm

from meta_saes.sae import load_feature_splitting_saes, load_gemma_sae

# %%

MAIN = __name__ == "__main__"

# model_name = "gemma2"
model_name = "gpt2"

if MAIN:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "gpt2":
        model, saes, token_dataset = load_feature_splitting_saes(
            device=device,
            saes_idxs=list(range(1, 9)),
        )
    elif model_name == "gemma2":
        model, saes, token_dataset = load_gemma_sae(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_12/width_16k/average_l0_41",
            device=device,
        )
    else:
        raise ValueError("Invalid model")

    torch.set_grad_enabled(False)

# %%

# create the data and model directories
if MAIN:

    os.makedirs("data", exist_ok=True)
    os.makedirs(f"data/{model_name}", exist_ok=True)

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
        model_activations = torch.load(f"data/{model_name}/model_activations.pt")
    except FileNotFoundError:
        model_activations = []

        def add_activations(acts, hook=None):
            model_activations.append(acts)

        model.remove_all_hook_fns()
        model.add_hook(saes[0].cfg.hook_name, add_activations)

        for batch in tqdm(torch.split(tokens, BATCH_SIZE), desc="Model"):
            model(batch)

        model.remove_all_hook_fns()

        model_activations = torch.cat(model_activations)
        torch.save(model_activations, f"data/{model_name}/model_activations.pt")

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i, sae in enumerate(saes):
        try:
            mean_acts = torch.load(
                f"data/{model_name}/mean_acts_{sae.W_dec.size(0)}.pt", weights_only=True
            )
        except FileNotFoundError:
            mean_acts = torch.zeros(SEQ_LEN, sae.W_dec.size(0))
            for model_acts in tqdm(
                torch.split(model_activations, BATCH_SIZE), desc=f"SAE {i+1}"
            ):
                sae_acts = (sae.encode(model_acts) > 0).float().sum(dim=0).cpu()
                mean_acts += sae_acts
            mean_acts /= len(tokens)
            torch.save(mean_acts, f"data/{model_name}/mean_acts_{sae.W_dec.size(0)}.pt")

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

if MAIN:
    cs = []
    for sae in tqdm(saes):
        W_dec = F.normalize(sae.W_dec, p=2, dim=-1)
        acts_norm = F.normalize(model_activations, p=2, dim=-1)

        n = W_dec.size(0)
        m = acts_norm.size(0)

        batch_size = 256
        maxes = []
        for i in range(0, n, batch_size):
            W_dec_batch = W_dec[i : i + batch_size]
            m = (
                torch.mm(W_dec_batch, acts_norm.flatten(end_dim=1).t())
                .max(dim=-1)
                .values
            )
            maxes.append(m)
        maxes = torch.cat(maxes)
        cs.append(maxes.cpu().numpy())

    plt.figure(figsize=(10, 6))
    for i, c in enumerate(cs):
        sns.kdeplot(c, bw_adjust=1.0, label=f"SAE {i+1}", alpha=0.3)
    plt.title("Cosine Similarity between SAE Decoder Directions and Model Activations")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
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
    plt.savefig(f"data/{model_name}/losses_and_ratio.png")


TRAIN_SIZE = 10_000
OMP_L0 = 8

if MAIN:
    warnings.filterwarnings("ignore")

    train_activations = model_activations[:TRAIN_SIZE]
    test_activations = model_activations[TRAIN_SIZE:]
    normed_activations = train_activations / train_activations.norm(dim=2).unsqueeze(2)

    try:
        atoms = torch.load(f"data/{model_name}/atoms.pt")
        atom_indices = torch.load(f"data/{model_name}/atom_indices.pt")
    except FileNotFoundError:
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

            coefs, indices = omp_pytorch(atoms, batch_activations, OMP_L0)
            selected_atoms = atoms[indices]
            recon = torch.bmm(coefs.unsqueeze(1), selected_atoms).squeeze(1)
            loss = ((batch_activations - recon) ** 2).sum(dim=1)
            loss = torch.clamp(loss, 0, 1)
            losses.extend(loss.cpu().tolist())

            mask = loss > 0.4
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

        print(f'Trained a model with {len(atoms)} atoms')

        torch.save(atoms, f"data/{model_name}/atoms.pt")
        torch.save(atom_indices, f"data/{model_name}/atom_indices.pt")
        with open(f"data/{model_name}/losses.pkl", "wb") as f:
            pickle.dump(losses, f)


# %%

if MAIN:
    sae = saes[3]

    try:
        raise FileNotFoundError()
        omp_losses = pickle.load(open(f"data/{model_name}/omp_losses.pkl", "rb"))
        sae_losses = pickle.load(open(f"data/{model_name}/sae_losses.pkl", "rb"))
        sae_losses = pickle.load(open(f"data/{model_name}/omp_sae_losses.pkl", "rb"))
        # TODO: get these in a separate loop during evaluation
        omp_activations = torch.load(f"data/{model_name}/omp_activations.pt")
        omp_indices = torch.load(f"data/{model_name}/omp_indices.pt")
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        BATCH_SIZE = 32
        omp_losses = []
        omp_indices = []
        omp_activations = []
        sae_losses = []
        omp_sae_losses = []
        for batch in tqdm(torch.split(test_activations, BATCH_SIZE)):
            flattened_batch = batch.flatten(end_dim=1)
            omp_batch = flattened_batch / flattened_batch.norm(dim=1).unsqueeze(1)

            coefs, indices = omp_pytorch(atoms, omp_batch, OMP_L0)
            omp_activations.extend(coefs.cpu().tolist())
            omp_indices.extend(indices.cpu().tolist())
            selected_atoms = atoms[indices]
            omp_recon = torch.bmm(coefs.unsqueeze(1), selected_atoms).squeeze(1)
            loss = ((omp_batch - omp_recon) ** 2).mean(dim=1)
            # can't reconstruct some? removing for now to investigate later
            loss = loss[loss < 1]
            omp_losses.extend(loss.cpu().tolist())

            coefs, indices = omp_pytorch(sae.W_dec, omp_batch, OMP_L0)
            selected_atoms = sae.W_dec[indices]
            omp_recon = torch.bmm(coefs.unsqueeze(1), selected_atoms).squeeze(1)
            loss = ((omp_batch - omp_recon) ** 2).mean(dim=1)
            # can't reconstruct some? removing for now to investigate later
            loss = loss[loss < 1]
            omp_sae_losses.extend(loss.cpu().tolist())

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

        with open(f"data/{model_name}/omp_losses.pkl", "wb") as f:
            pickle.dump(omp_losses, f)
        with open(f"data/{model_name}/sae_losses.pkl", "wb") as f:
            pickle.dump(sae_losses, f)
        with open(f"data/{model_name}/omp_sae_losses.pkl", "wb") as f:
            pickle.dump(omp_sae_losses, f)
        torch.save(omp_activations, f"data/{model_name}/omp_activations.pt")
        torch.save(omp_indices, f"data/{model_name}/omp_indices.pt")

    omp_loss = np.mean(omp_losses)
    sae_loss = np.mean(sae_losses)
    omp_sae_loss = np.mean(omp_sae_losses)
    print(f"OMP Loss: {omp_loss}")
    print(f"SAE Loss: {sae_loss}")
    print(f"OMP SAE Loss: {omp_sae_loss}")


# %%


def highlight_token_to_string(tokens, index, activation, verbose=True):
    if activation <= 0.01:
        return
    highlighted_text = Text()

    for i, token in enumerate(tokens):
        token_str = model.to_string([token]).replace("\n", "")
        if i == index:
            if token_str.isspace():
                token_str = "_"
            highlighted_text.append(
                f"<<<{token_str}>>>[{activation:.2f}]", style="bold bright_yellow"
            )
        else:
            highlighted_text.append(f"{token_str}")

    if verbose:
        rprint(highlighted_text)

    return highlighted_text.plain


def get_strings(acts):
    input_strings = []
    for i in range(acts.shape[-1]):
        active_inputs = (acts.sum(axis=1)[:, i] > 0).nonzero()[0]
        feature_input_strings = []
        for active_input in active_inputs:
            token_idx = np.where(acts[active_input] > 0)[0][0]
            str = highlight_token_to_string(
                tokens[int(TRAIN_SIZE) + active_input],
                token_idx,
                acts[active_input, token_idx, i],
                verbose=False,
            )
            if str is None:
                continue
            feature_input_strings.append((str, acts[active_input, token_idx, i]))
        # Sort feature_input_strings by activation in descending order
        feature_input_strings = [
            x[0]
            for x in sorted(feature_input_strings, key=lambda x: x[1], reverse=True)
        ]
        input_strings.append(feature_input_strings)
    return input_strings


def highlight_string(tokens, idx):
    str = ""
    for i, token in enumerate(tokens):
        token_str = model.to_string([token]).replace("\n", "")
        if i == idx:
            str += f"<<<{token_str}>>>"
        else:
            str += f"{token_str}"
    return str


def get_strings_activations(acts, tokens, n=20):
    non_zero_indices = np.argwhere(acts != 0)
    zero_indices = np.argwhere(acts == 0)

    num_non_zero = int(n * 0.8)
    num_zero = n - num_non_zero

    try:
        non_zero_idxs = non_zero_indices[
            np.random.choice(
                non_zero_indices.shape[0], size=num_non_zero, replace=False
            )
        ]
        zero_idxs = zero_indices[
            np.random.choice(zero_indices.shape[0], size=num_zero, replace=False)
        ]

        idxs = np.concatenate((non_zero_idxs, zero_idxs))
        np.random.shuffle(idxs)
    except ValueError:
        raise ValueError(
            f"Not enough activations found to sample {n} values. Found {non_zero_indices.shape[0]} activations."
        )

    sample_strs, sample_acts = [], []
    for inp, tok in idxs[:-1]:
        act = acts[inp, tok]

        str = highlight_string(tokens[int(TRAIN_SIZE) + inp], tok)

        sample_strs.append(str)
        sample_acts.append(act)

    test_idx = idxs[-1]
    test_str = highlight_string(tokens[int(TRAIN_SIZE) + test_idx[0]], test_idx[1])
    test_act = acts[test_idx[0], test_idx[1]]

    sample_strs, sample_acts = zip(
        *sorted(zip(sample_strs, sample_acts), key=lambda x: x[1], reverse=True)
    )

    return sample_strs, sample_acts, test_str, test_act


def generate_test_samples(activations, feature_sample_count, tokens, verbose=False):
    tests = []
    for i in range(feature_sample_count):
        try:
            sample_strs, sample_acts, test_str, test_act = get_strings_activations(
                activations[:, :, i], tokens, n=20
            )

            if verbose:
                for a, s in zip(sample_acts, sample_strs):
                    print(f"({a:.2f}) {s}")
                print(f"Test: ({test_act:.2f}) {test_str}")
                print("\n\n\n")

            tests.append((sample_strs, sample_acts, test_str, test_act))
        except ValueError as e:
            if verbose:
                print(e)
            continue
    return tests


# XXX: MAJOR CONCERN: We only use the features where at least 20 activations are
# found. This ignores infrequently activating latents, which could be
# uninterpretable.
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    feature_sample_count = 100
    sae_features = random.sample(range(sae.W_dec.size(0)), feature_sample_count)

    feature, count = np.unique(omp_indices, return_counts=True)
    omp_features = random.sample(range(atoms.size(0)), feature_sample_count)

    sae_activations = []
    for batch in torch.split(test_activations, BATCH_SIZE):
        sae_activations.append(sae.encode(batch)[:, :, sae_features].cpu())
    sae_activations = torch.cat(sae_activations, dim=0).cpu().numpy()
    sae_activations = sae_activations / sae_activations.max()

    omp_indices = omp_indices.reshape(
        test_activations.size(0), test_activations.size(1), OMP_L0
    )
    omp_activations = omp_activations.reshape(
        test_activations.size(0), test_activations.size(1), OMP_L0
    )
    omp_activations = np.clip(omp_activations, -1, 1)
    omp_features_array = np.array(omp_features)
    dense_omp_activations = np.zeros((6957, 16, len(omp_features_array)))
    for i, feature in enumerate(omp_features_array):
        mask = omp_indices == feature
        feature_activations = omp_activations * mask
        dense_omp_activations[:, :, i] = np.sum(feature_activations, axis=-1)

    dense_omp_activations = dense_omp_activations / dense_omp_activations.max()
    dense_omp_activations[dense_omp_activations < 0.0] = 0.0

    verbose = True
    omp_tests = generate_test_samples(
        dense_omp_activations, feature_sample_count, tokens, verbose=verbose
    )
    omp_actual = np.array([test[3] for test in omp_tests])
    sae_tests = generate_test_samples(
        sae_activations, feature_sample_count, tokens, verbose=verbose
    )
    sae_actual = np.array([test[3] for test in sae_tests])

    plt.figure(figsize=(10, 5))
    plt.hist(omp_actual, bins=20, alpha=0.5, label="OMP actual")
    plt.hist(sae_actual, bins=20, alpha=0.5, label="SAE actual")
    plt.title("Distribution of predictions doesn't match")
    fig.show()

# %%

# TODO: this section depends on the specific model, but is a huge pain to update
# as it's all manual. Trust for now I guess?

OMP_SOURCE = 0
SAE_SOURCE = 1

UNINTERPRETABLE = 0
SOMEWHAT_INTERPRETABLE = 1
INTERPRETABLE = 2

if MAIN:
    try:
        with open(f"data/{model_name}/all_tests.pkl", "rb") as f:
            all_tests = pickle.load(f)
        with open(f"data/{model_name}/source.pkl", "rb") as f:
            source = pickle.load(f)
    except FileNotFoundError:
        source = [0] * len(omp_tests) + [1] * len(sae_tests)
        all_tests = omp_tests + sae_tests

        shuffled_tests = list(zip(all_tests, source))
        random.shuffle(shuffled_tests)
        all_tests, source = zip(*shuffled_tests)

        with open(f"data/{model_name}/all_tests.pkl", "wb") as f:
            pickle.dump(all_tests, f)
        with open(f"data/{model_name}/source.pkl", "wb") as f:
            pickle.dump(source, f)

    for i, t in enumerate(all_tests):
        string = "Sample " + str(i) + "\n"
        for a, s in zip(t[1], t[0]):
            string += f"{a:.2f} {s}\n"
        string += f"\n\n"
        print(string)

    rating = {
        0: 2,
        1: 2,
        2: 2,
        3: 0,
        4: 2,
        5: 2,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 0,
        17: 2,
        18: 2,
        19: 2,
        20: 1,
        21: 2,
        22: 2,
        23: 1,
        24: 2,
        25: 2,
        26: 0,
        27: 1,
        28: 1,
        29: 2,
        30: 2,
        31: 2,
        32: 2,
        33: 2,
        34: 1,
        35: 0,
        36: 2,
        37: 0,
        38: 2,
        39: 1,
        40: 0,
        41: 0,
        42: 2,
        43: 2,
        44: 2,
        45: 1,
        46: 1,
        47: 1,
        48: 1,
        49: 2,
        50: 2,
        51: 0,
        52: 2,
        53: 2,
        54: 2,
        55: 2,
        56: 1,
        57: 2,
        58: 2,
        59: 1,
        60: 0,
        61: 2,
        62: 1,
        63: 1,
        64: 2,
        65: 0,
        66: 1,
        67: 2,
        68: 0,
        69: 2,
        70: 2,
        71: 2,
        72: 2,
        73: 2,
        74: 2,
        75: 2,
        76: 0,
        77: 2,
        78: 0,
        79: 2,
        80: 2,
        81: 2,
        82: 2,
        83: 1,
        84: 0,
        85: 0,
        86: 1,
        87: 2,
        88: 2,
        89: 1,
        90: 0,
        91: 0,
        92: 0,
        93: 2,
        94: 2,
        95: 1,
        96: 0,
        97: 0,
        98: 2,
        99: 1,
        100: 2,
        101: 1,
        102: 0,
        103: 1,
        104: 2,
        105: 2,
        106: 0,
        107: 2,
        108: 2,
        109: 2,
        110: 2,
        111: 2,
        112: 0,
        113: 2,
        114: 2,
        115: 2,
        116: 2,
        117: 2,
        117: 2,
    }

    interpretability_counts = {
        OMP_SOURCE: {UNINTERPRETABLE: 0, SOMEWHAT_INTERPRETABLE: 0, INTERPRETABLE: 0},
        SAE_SOURCE: {UNINTERPRETABLE: 0, SOMEWHAT_INTERPRETABLE: 0, INTERPRETABLE: 0},
    }

    # group ratings by source
    ratings_by_source = {OMP_SOURCE: [], SAE_SOURCE: []}
    for i, r in rating.items():
        ratings_by_source[source[i]].append(r)

    for source, values in ratings_by_source.items():
        for value in values:
            interpretability_counts[source][value] += 1

    # Calculate ratios for each source
    ratios = {}
    for source, counts in interpretability_counts.items():
        total = sum(counts.values())
        ratios[source] = {key: counts[key] / total for key in counts}

    # Create a rich table
    console = Console()
    table = Table(title="Interpretability Ratios by Source")

    table.add_column("Source", justify="center", style="cyan", no_wrap=True)
    table.add_column("Uninterpretable Ratio", justify="center", style="magenta")
    table.add_column("Somewhat Interpretable Ratio", justify="center", style="yellow")
    table.add_column("Interpretable Ratio", justify="center", style="green")

    # Add rows to the table
    for source, ratio in ratios.items():
        source_name = "OMP_SOURCE" if source == OMP_SOURCE else "SAE_SOURCE"
        table.add_row(
            source_name,
            f"{ratio[UNINTERPRETABLE]:.2f}",
            f"{ratio[SOMEWHAT_INTERPRETABLE]:.2f}",
            f"{ratio[INTERPRETABLE]:.2f}",
        )

    # Print the table
    console.print(table)

# %%


import ipywidgets as widgets
from IPython.display import display, clear_output
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def match_activation(search_activation, all_activations):
    if all_activations.shape[-1] != search_activation.shape[0]:
        raise ValueError(
            "Target tensor dimensions do not match the last dimension of the larger tensor."
        )

    reshaped_tensor = all_activations.view(-1, all_activations.shape[-1])
    matches = torch.all(reshaped_tensor == search_activation, dim=1)
    matching_indices_flat = torch.nonzero(matches, as_tuple=False).squeeze()

    if matching_indices_flat.numel() > 0:
        try:
            first_match_index = matching_indices_flat[0]
        except IndexError:
            first_match_index = matching_indices_flat
        batch_index = torch.div(
            first_match_index, all_activations.shape[1], rounding_mode="floor"
        )
        sequence_index = first_match_index % all_activations.shape[1]
        return batch_index.item(), sequence_index.item()
    else:
        raise ValueError("No matching vector found in the larger tensor.")


def get_activation_color(value):
    min_value = -1
    max_value = 1
    normalized_value = (
        (value - min_value) / (max_value - min_value) if max_value > min_value else 0
    )
    red = int(255 * (1 - normalized_value))
    green = int(255 * normalized_value)
    return f"rgb({red},{green},0)"


if __name__ == "__main__":
    input_idx = 0

    for token_idx in range(16):
        title = highlight_string(tokens[TRAIN_SIZE + input_idx], token_idx)

        table = Table(title=title)
        table.add_column("Activation", justify="right")
        table.add_column("String")

        indices = omp_indices[input_idx][token_idx]
        activations = omp_activations[input_idx][token_idx]

        sorted_activations_indices = sorted(
            zip(activations, indices), key=lambda x: x[0], reverse=True
        )
        min_activation = min(activations)
        max_activation = max(activations)

        for a, i in sorted_activations_indices:
            atom_input_idx, atom_token_idx = match_activation(
                atoms[i], normed_activations
            )
            highlighted_string = highlight_string(
                tokens[atom_input_idx], atom_token_idx
            )
            color = get_activation_color(a)
            activation_text = Text(f"{a:.2f}", style=color)
            table.add_row(str(atom_input_idx), activation_text, highlighted_string)

        console.print(table)


# %%
# TODO: Move this into a config file
openai.api_key = "..."


# # TODO: This is slow - convert to a batch job?
# def predict_activation(test_cases):
#     predictions = []
#     for test in tqdm(test_cases):
#         question = ""
#         for a, s in zip(test[1], test[0]):
#             question += f"{a:.2f} {s}\n"
#         question += f"\n\n{test[2]}"
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are doing auto-interp on SAE features. You will be presented with 20 activation and string pairs, in the string, a single token will be highlighted with <<<>>> brakcets. You will then be provided with another string with a highlighted token. Please predict the activation of that token using 2 decimal places. Respond only with the activation value.",
#                 },
#                 {"role": "user", "content": question},
#             ],
#             max_tokens=10,
#             temperature=0.0,
#         )
#         answer = response.choices[0].message.content.strip()
#         predictions.append(float(answer))
#     return predictions


# if __name__ == "__main__":
#     # save the tests for later
#     with open(f"data/{model_name}/omp_tests.pkl", "wb") as f:
#         pickle.dump(omp_tests, f)
#     with open(f"data/{model_name}/sae_tests.pkl", "wb") as f:
#         pickle.dump(sae_tests, f)

#     try:
#         omp_predictions = pickle.load(open(f"data/{model_name}/omp_predictions.pkl", "rb"))
#         sae_predictions = pickle.load(open(f"data/{model_name}/sae_predictions.pkl", "rb"))
#     except FileNotFoundError:
#         omp_predictions = predict_activation(omp_tests)
#         sae_predictions = predict_activation(sae_tests)

#         # save the predictions
#         with open(f"data/{model_name}/omp_predictions.pkl", "wb") as f:
#             pickle.dump(omp_predictions, f)
#         with open(f"data/{model_name}/sae_predictions.pkl", "wb") as f:
#             pickle.dump(sae_predictions, f)

#     omp_predictions = np.array(omp_predictions)
#     sae_predictions = np.array(sae_predictions)

# %%

from datasets import load_dataset


class OMPSAE:
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


def load_huggingface_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "bias_in_bios":
        dataset = load_dataset("LabHC/bias_in_bios")
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
    elif dataset_name == "amazon_reviews_all_ratings":
        dataset = load_dataset(
            "canrager/amazon_reviews_mcauley",
            config_name="dataset_all_categories_and_ratings_train1000_test250",
        )
    elif dataset_name == "amazon_reviews_1and5":
        dataset = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
        )
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return train_df, test_df


def ensure_shared_keys(train_data: dict, test_data: dict) -> tuple[dict, dict]:
    # Find keys that are in test but not in train
    test_only_keys = set(test_data.keys()) - set(train_data.keys())

    # Find keys that are in train but not in test
    train_only_keys = set(train_data.keys()) - set(test_data.keys())

    # Remove keys from test that are not in train
    for key in test_only_keys:
        print(f"Removing {key} from test set")
        del test_data[key]

    # Remove keys from train that are not in test
    for key in train_only_keys:
        print(f"Removing {key} from train set")
        del train_data[key]

    return train_data, test_data


POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0

# NOTE: These are going to be hardcoded, and won't change even if the underlying dataset or data labels change.
# This is a bit confusing, but IMO male_professor / female_nurse is a bit easier to understand than e.g. class1_pos_class2_pos / class1_neg_class2_neg
PAIRED_CLASS_KEYS = {
    "male / female": "female_data_only",
    "professor / nurse": "nurse_data_only",
    "male_professor / female_nurse": "female_nurse_data_only",
}

profession_dict = {
    "accountant": 0,
    "architect": 1,
    "attorney": 2,
    "chiropractor": 3,
    "comedian": 4,
    "composer": 5,
    "dentist": 6,
    "dietitian": 7,
    "dj": 8,
    "filmmaker": 9,
    "interior_designer": 10,
    "journalist": 11,
    "model": 12,
    "nurse": 13,
    "painter": 14,
    "paralegal": 15,
    "pastor": 16,
    "personal_trainer": 17,
    "photographer": 18,
    "physician": 19,
    "poet": 20,
    "professor": 21,
    "psychologist": 22,
    "rapper": 23,
    "software_engineer": 24,
    "surgeon": 25,
    "teacher": 26,
    "yoga_teacher": 27,
}
profession_int_to_str = {v: k for k, v in profession_dict.items()}

gender_dict = {
    "male": 0,
    "female": 1,
}

# From the original dataset
amazon_category_dict = {
    "All_Beauty": 0,
    "Toys_and_Games": 1,
    "Cell_Phones_and_Accessories": 2,
    "Industrial_and_Scientific": 3,
    "Gift_Cards": 4,
    "Musical_Instruments": 5,
    "Electronics": 6,
    "Handmade_Products": 7,
    "Arts_Crafts_and_Sewing": 8,
    "Baby_Products": 9,
    "Health_and_Household": 10,
    "Office_Products": 11,
    "Digital_Music": 12,
    "Grocery_and_Gourmet_Food": 13,
    "Sports_and_Outdoors": 14,
    "Home_and_Kitchen": 15,
    "Subscription_Boxes": 16,
    "Tools_and_Home_Improvement": 17,
    "Pet_Supplies": 18,
    "Video_Games": 19,
    "Kindle_Store": 20,
    "Clothing_Shoes_and_Jewelry": 21,
    "Patio_Lawn_and_Garden": 22,
    "Unknown": 23,
    "Books": 24,
    "Automotive": 25,
    "CDs_and_Vinyl": 26,
    "Beauty_and_Personal_Care": 27,
    "Amazon_Fashion": 28,
    "Magazine_Subscriptions": 29,
    "Software": 30,
    "Health_and_Personal_Care": 31,
    "Appliances": 32,
    "Movies_and_TV": 33,
}
amazon_int_to_str = {v: k for k, v in amazon_category_dict.items()}


amazon_rating_dict = {
    1.0: 1.0,
    5.0: 5.0,
}

dataset_metadata = {
    "bias_in_bios": {
        "text_column_name": "hard_text",
        "column1_name": "profession",
        "column2_name": "gender",
        "column2_autointerp_name": "gender",
        "column1_mapping": profession_dict,
        "column2_mapping": gender_dict,
    },
    "amazon_reviews_1and5": {
        "text_column_name": "text",
        "column1_name": "category",
        "column2_name": "rating",
        "column2_autointerp_name": "Amazon Review Sentiment",
        "column1_mapping": amazon_category_dict,
        "column2_mapping": amazon_rating_dict,
    },
}

chosen_classes_per_dataset = {
    "bias_in_bios": ["0", "1", "2", "6", "9"],
    "amazon_reviews_1and5": ["1", "2", "3", "5", "6"],
}

def get_balanced_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    min_samples_per_quadrant: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Returns a dataset of, in the case of bias_in_bios, a key of profession idx,
    and a value of a list of bios (strs) of len min_samples_per_quadrant * 2."""

    text_column_name = dataset_metadata[dataset_name]["text_column_name"]
    column1_name = dataset_metadata[dataset_name]["column1_name"]
    column2_name = dataset_metadata[dataset_name]["column2_name"]

    balanced_df_list = []

    for profession in tqdm(df[column1_name].unique()):
        prof_df = df[df[column1_name] == profession]
        min_count = prof_df[column2_name].value_counts().min()

        unique_groups = prof_df[column2_name].unique()
        if len(unique_groups) < 2:
            continue  # Skip professions with less than two groups

        if min_count < min_samples_per_quadrant:
            continue

        balanced_prof_df = pd.concat(
            [
                group.sample(n=min_samples_per_quadrant, random_state=random_seed)
                for _, group in prof_df.groupby(column2_name)
            ]
        ).reset_index(drop=True)
        balanced_df_list.append(balanced_prof_df)

    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    grouped = balanced_df.groupby(column1_name)[text_column_name].apply(list)

    str_data = {str(key): texts for key, texts in grouped.items()}

    balanced_data = {label: texts for label, texts in str_data.items()}

    for key in balanced_data.keys():
        balanced_data[key] = balanced_data[key][: min_samples_per_quadrant * 2]
        assert len(balanced_data[key]) == min_samples_per_quadrant * 2

    return balanced_data

def get_multi_label_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    train_set_size: int,
    test_set_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Returns a dict of [class_name, list[str]]"""
    # 4 is because male / gender for each profession
    minimum_train_samples_per_quadrant = train_set_size // 4
    minimum_test_samples_per_quadrant = test_set_size // 4

    train_data = get_balanced_dataset(
        train_df,
        dataset_name,
        minimum_train_samples_per_quadrant,
        random_seed=random_seed,
    )
    test_data = get_balanced_dataset(
        test_df,
        dataset_name,
        minimum_test_samples_per_quadrant,
        random_seed=random_seed,
    )

    train_data, test_data = ensure_shared_keys(train_data, test_data)

    return train_data, test_data


def filter_dataset(
    data: dict[str, list[str]], chosen_class_indices: list[str]
) -> dict[str, list[str]]:
    filtered_data = {}
    for class_name in chosen_class_indices:
        filtered_data[class_name] = data[class_name]
    return filtered_data


from transformers import AutoTokenizer


def tokenize_data(
    data: dict[str, list[str]], tokenizer: AutoTokenizer, max_length: int, device: str
) -> dict[str, dict]:
    tokenized_data = {}
    for key, texts in tqdm(data.items(), desc="Tokenizing data"):
        # .data so we have a dict, not a BatchEncoding
        tokenized_data[key] = (
            tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            .to(device)
            .data
        )
    return tokenized_data

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable, Optional
from jaxtyping import Int, Float, jaxtyped, BFloat16
from beartype import beartype
import einops
from transformer_lens import HookedTransformer
from sae_lens import SAE

LLM_NAME_TO_BATCH_SIZE = {
    "gpt2-small": 64,
    "pythia-70m-deduped": 500,
    "gemma-2-2b": 32,
}

LLM_NAME_TO_DTYPE = {
    "gpt2-small": torch.float32,
    "pythia-70m-deduped": torch.float32,
    "gemma-2-2b": torch.bfloat16,
}


@torch.no_grad
def get_all_llm_activations(
    tokenized_inputs_dict: dict[str, dict[str, Int[torch.Tensor, "dataset_size seq_len"]]],
    model: HookedTransformer,
    batch_size: int,
    hook_name: str,
) -> dict[str, Float[torch.Tensor, "dataset_size seq_len d_model"]]:
    """VERY IMPORTANT NOTE: We zero out masked token activations in this function. Later, we ignore zeroed activations."""
    all_classes_acts_BLD = {}

    for class_name in tokenized_inputs_dict:
        all_acts_BLD = []
        tokenized_inputs = tokenized_inputs_dict[class_name]

        for i in tqdm(
            range(0, len(tokenized_inputs["input_ids"]), batch_size),
            desc=f"Collecting activations for class {class_name}",
        ):
            tokens_BL = tokenized_inputs["input_ids"][i : i + batch_size]
            attention_mask_BL = tokenized_inputs["attention_mask"][i : i + batch_size]

            acts_BLD = None

            def activation_hook(resid_BLD: torch.Tensor, hook):
                nonlocal acts_BLD
                acts_BLD = resid_BLD

            model.run_with_hooks(
                tokens_BL, return_type=None, fwd_hooks=[(hook_name, activation_hook)]
            )

            acts_BLD = acts_BLD * attention_mask_BL[:, :, None]
            all_acts_BLD.append(acts_BLD)

        all_acts_BLD = torch.cat(all_acts_BLD, dim=0)

        all_classes_acts_BLD[class_name] = all_acts_BLD

    return all_classes_acts_BLD


@torch.no_grad
def create_meaned_model_activations(
    all_llm_activations_BLD: dict[str, Float[torch.Tensor, "batch_size seq_len d_model"]],
) -> dict[str, Float[torch.Tensor, "batch_size d_model"]]:
    """VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    all_llm_activations_BD = {}
    for class_name in all_llm_activations_BLD:
        acts_BLD = all_llm_activations_BLD[class_name]
        dtype = acts_BLD.dtype

        activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        meaned_acts_BD = einops.reduce(acts_BLD, "B L D -> B D", "sum") / nonzero_acts_B[:, None]
        all_llm_activations_BD[class_name] = meaned_acts_BD

    return all_llm_activations_BD


@torch.no_grad
def get_sae_meaned_activations(
    all_llm_activations_BLD: dict[str, Float[torch.Tensor, "batch_size seq_len d_model"]],
    sae,
    sae_batch_size: int,
    dtype: torch.dtype,
) -> dict[str, Float[torch.Tensor, "batch_size d_sae"]]:
    """VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""
    all_sae_activations_BF = {}
    for class_name in all_llm_activations_BLD:
        all_acts_BLD = all_llm_activations_BLD[class_name]

        all_acts_BF = []

        for i in range(0, len(all_acts_BLD), sae_batch_size):
            acts_BLD = all_acts_BLD[i : i + sae_batch_size]
            acts_BLF = sae.encode(acts_BLD)

            activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
            nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
            nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

            acts_BLF = acts_BLF * nonzero_acts_BL[:, :, None]
            acts_BF = einops.reduce(acts_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None]
            acts_BF = acts_BF.to(dtype=dtype)

            all_acts_BF.append(acts_BF)

        all_acts_BF = torch.cat(all_acts_BF, dim=0)
        all_sae_activations_BF[class_name] = all_acts_BF

    return all_sae_activations_BF

from dataclasses import dataclass, field
from typing import Optional
import torch

import copy
from typing import Optional

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def prepare_probe_data(
    all_activations: dict[str, Float[torch.Tensor, "num_datapoints_per_class ... d_model"]],
    class_name: str,
    spurious_corr: bool = False,
) -> tuple[
    Float[torch.Tensor, "num_datapoints_per_class_x_2 ... d_model"],
    Int[torch.Tensor, "num_datapoints_per_class_x_2"],
]:
    """spurious_corr is for the SHIFT metric. In this case, all_activations has 3 pairs of keys, or 6 total.
    It's a bit unfortunate to introduce coupling between the metrics, but most of the code is reused between them.
    The ... means we can have an optional seq_len dimension between num_datapoints_per_class and d_model.
    """
    positive_acts_BD = all_activations[class_name]
    device = positive_acts_BD.device

    num_positive = len(positive_acts_BD)

    if spurious_corr:
        if class_name in PAIRED_CLASS_KEYS.keys():
            negative_acts = all_activations[PAIRED_CLASS_KEYS[class_name]]
        elif class_name in PAIRED_CLASS_KEYS.values():
            reversed_dict = {v: k for k, v in PAIRED_CLASS_KEYS.items()}
            negative_acts = all_activations[reversed_dict[class_name]]
        else:
            raise ValueError(f"Class {class_name} not found in paired class keys.")
    else:
        # Collect all negative class activations and labels
        negative_acts = []
        for idx, acts in all_activations.items():
            if idx != class_name:
                negative_acts.append(acts)

        negative_acts = torch.cat(negative_acts)

    # Randomly select num_positive samples from negative class
    indices = torch.randperm(len(negative_acts))[:num_positive]
    selected_negative_acts_BD = negative_acts[indices]

    assert selected_negative_acts_BD.shape == positive_acts_BD.shape

    # Combine positive and negative samples
    combined_acts = torch.cat([positive_acts_BD, selected_negative_acts_BD])

    combined_labels = torch.empty(len(combined_acts), dtype=torch.int, device=device)
    combined_labels[:num_positive] = POSITIVE_CLASS_LABEL
    combined_labels[num_positive:] = NEGATIVE_CLASS_LABEL

    # Shuffle the combined data
    shuffle_indices = torch.randperm(len(combined_acts))
    shuffled_acts = combined_acts[shuffle_indices]
    shuffled_labels = combined_labels[shuffle_indices]

    return shuffled_acts, shuffled_labels


def get_top_k_mean_diff_mask(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
) -> Bool[torch.Tensor, "k"]:
    positive_mask_B = labels_B == POSITIVE_CLASS_LABEL
    negative_mask_B = labels_B == NEGATIVE_CLASS_LABEL

    positive_distribution_D = acts_BD[positive_mask_B].mean(dim=0)
    negative_distribution_D = acts_BD[negative_mask_B].mean(dim=0)
    distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
    top_k_indices_D = torch.argsort(distribution_diff_D, descending=True)[:k]

    mask_D = torch.ones(acts_BD.shape[1], dtype=torch.bool, device=acts_BD.device)
    mask_D[top_k_indices_D] = False

    return mask_D


def apply_topk_mask_zero_dims(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()
    masked_acts_BD[:, mask_D] = 0.0

    return masked_acts_BD


def apply_topk_mask_reduce_dim(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()

    masked_acts_BD = masked_acts_BD[:, ~mask_D]

    return masked_acts_BD


@beartype
def train_sklearn_probe(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    max_iter: int = 1000,  # non-default sklearn value, increased due to convergence warnings
    C: float = 1.0,  # default sklearn value
    verbose: bool = False,
    l1_ratio: Optional[float] = None,
) -> tuple[LogisticRegression, float]:
    train_inputs = train_inputs.to(dtype=torch.float32)
    test_inputs = test_inputs.to(dtype=torch.float32)

    # Convert torch tensors to numpy arrays
    train_inputs_np = train_inputs.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_inputs_np = test_inputs.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Initialize the LogisticRegression model
    if l1_ratio is not None:
        # Use Elastic Net regularization
        probe = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            verbose=int(verbose),
        )
    else:
        # Use L2 regularization
        probe = LogisticRegression(penalty="l2", C=C, max_iter=max_iter, verbose=int(verbose))

    # Train the model
    probe.fit(train_inputs_np, train_labels_np)

    # Compute accuracies
    train_accuracy = accuracy_score(train_labels_np, probe.predict(train_inputs_np))
    test_accuracy = accuracy_score(test_labels_np, probe.predict(test_inputs_np))

    if verbose:
        print(f"\nTraining completed.")
        print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}\n")

    return probe, test_accuracy


# Helper function to test the probe
@beartype
def test_sklearn_probe(
    inputs: Float[torch.Tensor, "dataset_size d_model"],
    labels: Int[torch.Tensor, "dataset_size"],
    probe: LogisticRegression,
) -> float:
    inputs = inputs.to(dtype=torch.float32)
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions = probe.predict(inputs_np)
    return accuracy_score(labels_np, predictions)


@torch.no_grad
def test_probe_gpu(
    inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    labels: Int[torch.Tensor, "test_dataset_size"],
    batch_size: int,
    probe: Probe,
) -> float:
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for i in range(0, len(labels), batch_size):
            acts_BD = inputs[i : i + batch_size]
            labels_B = labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            correct_B = (preds_B == labels_B).float()

            all_corrects.append(correct_B)
            corrects_0.append(correct_B[labels_B == 0])
            corrects_1.append(correct_B[labels_B == 1])

            loss = criterion(logits_B, labels_B.to(dtype=probe.net.weight.dtype))
            losses.append(loss)

        accuracy_all = torch.cat(all_corrects).mean().item()
        accuracy_0 = torch.cat(corrects_0).mean().item() if corrects_0 else 0.0
        accuracy_1 = torch.cat(corrects_1).mean().item() if corrects_1 else 0.0
        all_loss = torch.stack(losses).mean().item()

    return accuracy_all


def train_probe_gpu(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    verbose: bool = False,
    l1_penalty: Optional[float] = None,
    early_stopping_patience: int = 10,
) -> tuple[Probe, float]:
    """We have a GPU training function for training on all SAE features, which was very slow (1 minute+) on CPU."""
    device = train_inputs.device
    model_dtype = train_inputs.dtype

    print(f"Training probe with dim: {dim}, device: {device}, dtype: {model_dtype}")

    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_test_accuracy = 0.0
    best_probe = None
    patience_counter = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_inputs))

        for i in range(0, len(train_inputs), batch_size):
            batch_indices = indices[i : i + batch_size]
            acts_BD = train_inputs[batch_indices]
            labels_B = train_labels[batch_indices]
            logits_B = probe(acts_BD)
            loss = criterion(
                logits_B, labels_B.clone().detach().to(device=device, dtype=model_dtype)
            )

            if l1_penalty is not None:
                l1_loss = l1_penalty * torch.sum(torch.abs(probe.net.weight))
                loss += l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = test_probe_gpu(train_inputs, train_labels, batch_size, probe)
        test_accuracy = test_probe_gpu(test_inputs, test_labels, batch_size, probe)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_probe = copy.deepcopy(probe)
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}"
            )

        if patience_counter >= early_stopping_patience:
            print(f"GPU probe training early stopping triggered after {epoch + 1} epochs")
            break

    return best_probe, best_test_accuracy


def train_probe_on_activations(
    train_activations: dict[str, Float[torch.Tensor, "train_dataset_size d_model"]],
    test_activations: dict[str, Float[torch.Tensor, "test_dataset_size d_model"]],
    select_top_k: Optional[int] = None,
    use_sklearn: bool = True,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    verbose: bool = False,
    early_stopping_patience: int = 10,
    spurious_corr: bool = False,
) -> tuple[dict[str, LogisticRegression | Probe], dict[str, float]]:
    """Train a probe on the given activations and return the probe and test accuracies for each profession.
    use_sklearn is a flag to use sklearn's LogisticRegression model instead of a custom PyTorch model.
    We use sklearn by default. probe training on GPU is only for training a probe on all SAE features.
    """
    torch.set_grad_enabled(True)

    probes, test_accuracies = {}, {}

    for profession in train_activations.keys():
        train_acts, train_labels = prepare_probe_data(
            train_activations, profession, spurious_corr
        )
        test_acts, test_labels = prepare_probe_data(
            test_activations, profession, spurious_corr
        )

        if select_top_k is not None:
            activation_mask_D = get_top_k_mean_diff_mask(
                train_acts, train_labels, select_top_k
            )
            train_acts = apply_topk_mask_reduce_dim(train_acts, activation_mask_D)
            test_acts = apply_topk_mask_reduce_dim(test_acts, activation_mask_D)

        activation_dim = train_acts.shape[1]

        print(f"Num non-zero elements: {activation_dim}")

        if use_sklearn:
            probe, test_accuracy = train_sklearn_probe(
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                verbose=False,
            )
        else:
            probe, test_accuracy = train_probe_gpu(
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                dim=activation_dim,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                verbose=verbose,
                early_stopping_patience=early_stopping_patience,
            )

        print(f"Test accuracy for {profession}: {test_accuracy}")

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

    return probes, test_accuracies




def average_test_accuracy(test_accuracies: dict[str, float]) -> float:
    return sum(test_accuracies.values()) / len(test_accuracies)



@dataclass
class EvalConfig:
    random_seed: int = 42

    dataset_names: list[str] = field(
        default_factory=lambda: ["bias_in_bios", "amazon_reviews_1and5"]
    )

    probe_train_set_size: int = 4000
    probe_test_set_size: int = 1000
    context_length: int = 128

    sae_batch_size: int = 16

    ## Uncomment to run Pythia SAEs

    # sae_releases: list[str] = field(
    #     default_factory=lambda: [
    #         "sae_bench_pythia70m_sweep_standard_ctx128_0712",
    #         "sae_bench_pythia70m_sweep_topk_ctx128_0730",
    #     ]
    # )
    model_name: str = "gpt2-small"
    layer: int = 8
    trainer_ids: Optional[list[int]] = field(default_factory=lambda: list(range(20)))
    trainer_ids: Optional[list[int]] = field(default_factory=lambda: [10])
    include_checkpoints: bool = False

    ## Uncomment to run Gemma SAEs

    # sae_releases: list[str] = field(
    #     default_factory=lambda: [
    #         "gemma-scope-2b-pt-res",
    #         "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
    #         "sae_bench_gemma-2-2b_sweep_standard_ctx128_ef8_0824",
    #     ]
    # )
    # model_name: str = "gemma-2-2b"
    # layer: int = 19
    # trainer_ids: Optional[list[int]] = None
    # include_checkpoints: bool = False

    k_values: list[int] = field(default_factory=lambda: [1]) #, 2, 5, 10, 20, 50, 100])

    selected_saes_dict: dict = field(default_factory=lambda: {})

if MAIN:

    dataset_name = "bias_in_bios"
    config = EvalConfig()


    train_df, test_df = load_huggingface_dataset(dataset_name)
    train_data, test_data = get_multi_label_train_test_data(
        train_df,
        test_df,
        dataset_name,
        config.probe_train_set_size,
        config.probe_test_set_size,
        config.random_seed,
    )

    chosen_classes = chosen_classes_per_dataset[dataset_name]

    train_data = filter_dataset(train_data, chosen_classes)
    test_data = filter_dataset(test_data, chosen_classes)

    train_data = tokenize_data(
        train_data, model.tokenizer, config.context_length, device
    )
    test_data = tokenize_data(
        test_data, model.tokenizer, config.context_length, device
    )

    print(f"Running evaluation for layer {config.layer}")
    hook_name = f"blocks.{config.layer}.hook_resid_post"

    llm_batch_size = LLM_NAME_TO_BATCH_SIZE[config.model_name]
    llm_dtype = LLM_NAME_TO_DTYPE[config.model_name]

    all_train_acts_BLD = get_all_llm_activations(
        train_data, model, llm_batch_size, hook_name
    )
    all_test_acts_BLD = get_all_llm_activations(
        test_data, model, llm_batch_size, hook_name
    )

    all_train_acts_BD = create_meaned_model_activations(
        all_train_acts_BLD
    )
    all_test_acts_BD = create_meaned_model_activations(
        all_test_acts_BLD
    )

    results_dict = {}


    llm_probes, llm_test_accuracies = train_probe_on_activations(
        all_train_acts_BD,
        all_test_acts_BD,
        select_top_k=None,
    )

    llm_results = {"llm_test_accuracy": average_test_accuracy(llm_test_accuracies)}

    for k in config.k_values:
        llm_top_k_probes, llm_top_k_test_accuracies = (
            train_probe_on_activations(
                all_train_acts_BD,
                all_test_acts_BD,
                select_top_k=k,
            )
        )
        llm_results[f"llm_top_{k}_test_accuracy"] = average_test_accuracy(
            llm_top_k_test_accuracies
        )

    import gc

    for i, sae in enumerate(tqdm(
        [
            sae,
            OMPSAE(atoms, OMP_L0),
            OMPSAE(sae.W_dec, OMP_L0),
        ],
        desc="Running SAE evaluation on all selected SAEs",
    )):
        gc.collect()
        torch.cuda.empty_cache()

        sae_name = sae.__class__.__name__

        # if "topk" in sae_name:
        #     assert isinstance(sae.activation_fn, TopK)

        all_sae_train_acts_BF = get_sae_meaned_activations(
            all_train_acts_BLD, sae, config.sae_batch_size, llm_dtype
        )
        all_sae_test_acts_BF = get_sae_meaned_activations(
            all_test_acts_BLD, sae, config.sae_batch_size, llm_dtype
        )

        _, sae_test_accuracies = train_probe_on_activations(
            all_sae_train_acts_BF,
            all_sae_test_acts_BF,
            select_top_k=None,
            use_sklearn=False,
            batch_size=250,
            epochs=100,
            lr=1e-2,
        )

        results_dict[sae_name] = {}

        for llm_result_key, llm_result_value in llm_results.items():
            results_dict[sae_name][llm_result_key] = llm_result_value

        results_dict[sae_name]["sae_test_accuracy"] = average_test_accuracy(
            sae_test_accuracies
        )

        for k in config.k_values:
            sae_top_k_probes, sae_top_k_test_accuracies = (
                train_probe_on_activations(
                    all_sae_train_acts_BF,
                    all_sae_test_acts_BF,
                    select_top_k=k,
                )
            )
            results_dict[sae_name][f"sae_top_{k}_test_accuracy"] = (
                average_test_accuracy(sae_top_k_test_accuracies)
            )

        print(results_dict)
