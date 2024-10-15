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
    normed_activations = train_activations / train_activations.norm(
        dim=2
    ).unsqueeze(2)


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

        torch.save(atoms, f"data/{model_name}/atoms.pt")
        torch.save(atom_indices, f"data/{model_name}/atom_indices.pt")
        with open(f"data/{model_name}/losses.pkl", "wb") as f:
            pickle.dump(losses, f)


# %%

if MAIN:
    sae = saes[5]

    try:
        omp_losses = pickle.load(open(f"data/{model_name}/omp_losses.pkl", "rb"))
        sae_losses = pickle.load(open(f"data/{model_name}/sae_losses.pkl", "rb"))
        # TODO: get these in a separate loop during evaluation
        omp_activations = torch.load(f"data/{model_name}/omp_activations.pt")
        omp_indices = torch.load(f"data/{model_name}/omp_indices.pt")
    except (pickle.UnpicklingError, FileNotFoundError) as e:
        BATCH_SIZE = 32
        omp_losses = []
        omp_indices = []
        omp_activations = []
        sae_losses = []
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
        torch.save(omp_activations, f"data/{model_name}/omp_activations.pt")
        torch.save(omp_indices, f"data/{model_name}/omp_indices.pt")

    omp_loss = np.mean(omp_losses)
    sae_loss = np.mean(sae_losses)
    print(f"OMP Loss: {omp_loss}")
    print(f"SAE Loss: {sae_loss}")


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
        raise ValueError("Target tensor dimensions do not match the last dimension of the larger tensor.")
    
    reshaped_tensor = all_activations.view(-1, all_activations.shape[-1])
    matches = torch.all(reshaped_tensor == search_activation, dim=1)
    matching_indices_flat = torch.nonzero(matches, as_tuple=False).squeeze()

    if matching_indices_flat.numel() > 0:
        try:
            first_match_index = matching_indices_flat[0]
        except IndexError:
            first_match_index = matching_indices_flat
        batch_index = torch.div(first_match_index, all_activations.shape[1], rounding_mode='floor')
        sequence_index = first_match_index % all_activations.shape[1]
        return batch_index.item(), sequence_index.item()
    else:
        raise ValueError("No matching vector found in the larger tensor.")


def get_activation_color(value):
    min_value = -1
    max_value = 1
    normalized_value = (value - min_value) / (max_value - min_value) if max_value > min_value else 0
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

        sorted_activations_indices = sorted(zip(activations, indices), key=lambda x: x[0], reverse=True)
        min_activation = min(activations)
        max_activation = max(activations)

        for a, i in sorted_activations_indices:
            atom_input_idx, atom_token_idx = match_activation(atoms[i], normed_activations)
            highlighted_string = highlight_string(tokens[atom_input_idx], atom_token_idx)
            color = get_activation_color(a)
            activation_text = Text(f"{a:.2f}", style=color)
            table.add_row(str(atom_input_idx), activation_text, highlighted_string)

        console.print(table)



# %%
# TODO: Move this into a config file and recycle the key
openai.api_key = "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"


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
