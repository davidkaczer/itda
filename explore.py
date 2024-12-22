# %%
from copy import copy

import torch
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from example_saes.ito_sae import ITO_SAE, to_unbounded_activations
from example_saes.train import (
    get_model_name,
    load_model,
    get_atom_indices,
)
from tqdm import tqdm

# %%

MAIN = __name__ == "__main__"

if MAIN:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    MODEL = "gpt2"
    LAYER = 8
    L0 = 40
    SEQ_LEN = 128
    TARGET_LOSS = 3.0
    model_name = get_model_name(MODEL, LAYER, L0, TARGET_LOSS)

    atoms = torch.load(f"data/{model_name}/atoms.pt")
    model_activations = torch.load(f"data/{model_name}/model_activations.pt")

    model, saes, token_dataset = load_model(
        MODEL, LAYER, device=device, gpt2_saes=list(range(1, 2))
    )
    tokens = token_dataset["tokens"][:, :SEQ_LEN]

    sae = ITO_SAE(atoms, l0=8, cfg=copy(saes[0].cfg))
    # sae = ITO_SAE(atoms, l0=L0, cfg=copy(saes[0].cfg))

# %%

if MAIN:
    try:
        atom_indices = torch.load(f"data/{model_name}/atom_indices.pt")
    except FileNotFoundError:
        atom_indices = get_atom_indices(
            atoms.to(device), model_activations, batch_size=1024
        )
        torch.save(atom_indices, f"data/{model_name}/atom_indices.pt")
    atom_indices = atom_indices.to(device)

# %%


def get_model_activations(strings, model, batch_size=32):
    tokens = model.tokenizer(
        strings, return_tensors="pt", padding=True, truncation=True
    )["input_ids"].to(device)

    model_activations = []

    def add_activations(acts, hook=None):
        model_activations.append(acts.cpu())

    model.remove_all_hook_fns()
    model.add_hook(sae.cfg.hook_name, add_activations)
    for batch in tqdm(torch.split(tokens, batch_size), desc="Model"):
        model(batch)
    model.remove_all_hook_fns()
    acts = torch.stack(model_activations).squeeze(0)
    return acts


def encode_string(string, model, sae, batch_size=32):
    acts = get_model_activations([string], model, batch_size=batch_size)
    return sae.encode(acts)


def print_decomposition(encoding, atom_indices, tokens, tokenizer):
    active_latents = torch.nonzero(encoding).squeeze(-1)
    active_values = encoding[active_latents]
    sorted_indices = torch.argsort(active_values, descending=True)
    active_latents = active_latents[sorted_indices]
    atom_indices = atom_indices[active_latents]

    for act, idxs in zip(active_latents, atom_indices):
        activation_value = encoding[act].item()
        highlighted_string = highlight_string(
            tokens[idxs[0]], idxs[1], tokenizer, crop=10
        )
        combined_html = f"<b>{activation_value:.2f}</b> " + highlighted_string.data
        display(HTML(combined_html))


def highlight_string(tokens, idx, tokenizer, crop=-1):
    str_ = ""

    if crop != -1:
        start_idx = max(0, idx - crop)
        end_idx = min(len(tokens), idx + crop + 1)
        tokens = tokens[start_idx:end_idx]
        idx -= start_idx

    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token]).replace("\n", "")
        if i == idx:
            str_ += f'<span style="background-color: yellow; font-weight: bold;">{token_str}</span>'
        else:
            str_ += f"{token_str}"

    return HTML(str_)


# if MAIN:
#     s = "Adolf Hitler"
#     s = "Winston Churchill"
#     encoding = encode_string(s, model, sae)
#     encoding = to_unbounded_activations(encoding)
#     print("Decomposition of:", s)
#     print_decomposition(encoding[-1], atom_indices, tokens, model.tokenizer)

# Interactive dictionary exploration

dictionary = [
    "happy",
    # "joy",
    "James",
    # "sad",
    "anger"
]

tokens = model.tokenizer(
    dictionary, return_tensors="pt", padding=True, truncation=True
)["input_ids"]
model_activations = get_model_activations(dictionary, model)
mask = tokens != 50256
atoms = model_activations[torch.arange(mask.size(0)), mask.float().argmin(dim=-1) - 1]
sae = ITO_SAE(atoms, l0=len(dictionary), cfg=copy(saes[0].cfg))

s = "I love Mark."
encoding = encode_string(s, model, sae)
acts = get_model_activations([s], model)
recon = sae.decode(encoding)
loss = (acts - recon).pow(2).mean().item()
print("Decomposition of:", s, "into", dictionary)
print("Loss:", loss)

encoding = to_unbounded_activations(encoding).squeeze(0)
print_decomposition(
    encoding[1],
    torch.stack([torch.arange(mask.size(0)), mask.float().argmin(dim=-1) - 1], dim=-1),
    tokens,
    model.tokenizer,
)

# %%


def get_all_activations(sae, model_activations, tokens):
    all_acts = []
    batch_size = 128
    for start_idx in tqdm(range(0, len(model_activations), batch_size)):
        try:
            end_idx = min(len(tokens), start_idx + batch_size)
            acts = sae.encode(model_activations[start_idx:end_idx].to(sae.device))
            sparse = acts.to_sparse().cpu()
            all_acts.append(sparse)
        except torch.OutOfMemoryError:
            print(f"Out of memory at {start_idx}")
            break
    return torch.cat(all_acts, dim=0)


if MAIN:
    try:
        all_acts = torch.load(f"data/{model_name}/all_acts.pt")
    except FileNotFoundError:
        all_acts = get_all_activations(sae, model_activations, tokens)
        torch.save(all_acts.coalesce(), f"data/{model_name}/all_acts.pt")

    if not all_acts.is_coalesced():
        all_acts = all_acts.coalesce()


# %%

def print_max_activating_samples(all_acts, atom_idx, tokens, tokenizer, n=20):
    sparse_active_idxs = (all_acts.indices()[2] == atom_idx) & (all_acts.values() > 0.2)
    active_idxs = all_acts.indices()[:, sparse_active_idxs][:2]
    atom_acts = all_acts.values()[sparse_active_idxs]

    # take a random sample of n activations
    sample_idxs = torch.randperm(len(atom_acts))[:n]
    active_idxs = active_idxs[:, sample_idxs]
    atom_acts = atom_acts[sample_idxs]

    # sort by activation value
    sorted_idxs = torch.argsort(atom_acts, descending=True)
    active_idxs = active_idxs[:, sorted_idxs]
    atom_acts = atom_acts[sorted_idxs]

    for activation_value, idxs in zip(atom_acts, active_idxs.T):
        highlighted_string = highlight_string(
            tokens[idxs[0]], idxs[1], tokenizer, crop=10
        )
        combined_html = f"<b>{activation_value:.2f}</b> " + highlighted_string.data
        display(HTML(combined_html))


atom_idx = 16009
print_max_activating_samples(all_acts, atom_idx, tokens, model.tokenizer)

# %%


def plot_activations_histogram(all_acts, atom_idx):
    sparse_active_idxs = all_acts.indices()[2] == atom_idx
    atom_acts = all_acts.values()[sparse_active_idxs]
    plt.hist(atom_acts.cpu().numpy(), bins=100)


plot_activations_histogram(all_acts, atom_idx)
