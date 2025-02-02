# %%
import os
import random
import yaml
import torch
import zarr
from datasets import load_dataset
from ito_sae import ITO_SAE
from transformers import AutoTokenizer
from IPython.display import HTML, display
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
# === USER CONFIGURATION ===

RUN_ID = "tadbdwqi"

# Highlight colors (dark background, so text is presumably white)
TOKEN_OF_INTEREST_COLOR = "#3B1E6B"  # Indigo-ish
ATOM_ACTIVATION_COLOR = "#144B39"  # Dark greenish
ORIGIN_HIGHLIGHT_COLOR = "#5A3B00"  # Dark brownish

# === END USER CONFIG ===


# %%
def highlight_tokens(
    token_ids,
    active_positions_set,
    special_position=None,
    highlight_color="#144B39",
    special_color="#3B1E6B",
    tokenizer=None,
):
    """
    Return an HTML string highlighting:
      - positions in `active_positions_set` with highlight_color
      - `special_position` (the token of interest) with a separate special_color
    All other tokens are not highlighted.

    `tokenizer` is used to decode tokens. If None, we’ll just show token IDs.
    """
    if tokenizer is None:
        # fallback: just show token IDs
        def decode_fn(tok_id):
            return f"[{tok_id}]"

    else:

        def decode_fn(tok_id):
            text = tokenizer.decode([tok_id], skip_special_tokens=False)
            return text.replace("\n", "\\n")

    html = ""
    for idx, tok_id in enumerate(token_ids):
        decoded_str = decode_fn(tok_id)

        if idx == special_position:
            # highlight with special color
            html += f'<span style="background-color: {special_color}; font-weight: bold;">{decoded_str}</span> '
        elif idx in active_positions_set:
            html += f'<span style="background-color: {highlight_color};">{decoded_str}</span> '
        else:
            html += f"{decoded_str} "
    return HTML(html)


def get_token_activation(zf_layer, seq_idx, token_idx):
    """
    Load the entire activation for sequence `seq_idx` from Zarr.
    Shape: (seq_len, d_model).
    """
    arr = zf_layer[seq_idx][token_idx]  # shape (1, seq_len, d_model)
    return torch.from_numpy(arr)


def gather_atom_origin_snippet(atom_idx, atom_indices, ds, tokenizer, context=5):
    """
    Each dictionary atom was originally added from (some_seq, some_pos).
    We'll show a short snippet around that original token, with the origin token highlighted.
    """
    try:
        origin_seq_idx, origin_pos_idx = atom_indices[atom_idx].tolist()
    except AttributeError:
        origin_seq_idx, origin_pos_idx = atom_indices[atom_idx][0]
    example = ds[origin_seq_idx]
    tok_ids = example["input_ids"]

    start_i = max(0, origin_pos_idx - context)
    end_i = min(len(tok_ids), origin_pos_idx + context + 1)

    snippet_html = ""
    for i in range(start_i, end_i):
        decoded = ''.join(tokenizer.decode([tok_ids[i]], skip_special_tokens=False).replace(
            "\n", "\\n"
        ))
        if i == origin_pos_idx:
            snippet_html += f'<span style="background-color:{ORIGIN_HIGHLIGHT_COLOR}; font-weight:bold">/hl{{{decoded}}}</span> '
        else:
            snippet_html += f"{decoded} "
    return snippet_html


# %%
def match_atoms_in_activations(
    atoms: torch.Tensor, zarr_acts, threshold: float = 1e-6, batch_size: int = 128
) -> dict[int, list[tuple[int, int]]]:
    """
    Naively search a Zarr dataset of activations for each atom’s exact (or near-exact)
    match within a certain threshold. Returns a dictionary mapping:
        { atom_idx: [(seq_idx, pos_idx), ...] }

    Args:
        atoms:        (n_atoms, d_model) dictionary atoms
        zarr_acts:    Zarr array of shape (num_seqs, seq_len, d_model)
        threshold:    L2 distance threshold for considering two vectors a "match"
        batch_size:   How many sequences to process at a time

    Note: This method is O(num_seqs * seq_len * n_atoms * d_model) in the worst case,
          so it can be very slow for large datasets.
    """
    device = atoms.device if atoms.is_cuda else "cpu"
    n_atoms, d_model = atoms.shape

    # We'll store all matching (seq_idx, pos_idx) for each atom
    matches = {atom_idx: [] for atom_idx in range(n_atoms)}

    total_seqs = zarr_acts.shape[0]
    for start_seq in tqdm(range(0, total_seqs, batch_size), desc="Matching atoms"):
        end_seq = min(start_seq + batch_size, total_seqs)

        # Load this chunk of sequences from Zarr
        # shape = (chunk_size, seq_len, d_model)
        chunk = zarr_acts[start_seq:end_seq]
        chunk_acts = torch.from_numpy(chunk).to(device)

        for i in range(chunk_acts.shape[0]):
            seq_idx = start_seq + i
            # shape: (seq_len, d_model)
            seq_acts = chunk_acts[i]

            # For each token in this sequence
            for pos_idx in range(seq_acts.shape[0]):
                token_vec = seq_acts[pos_idx]
                # Compare to all atoms at once
                diffs = torch.norm(atoms - token_vec, dim=1)  # shape: (n_atoms,)

                # If some atom(s) are within threshold, record them
                close_mask = (diffs < threshold).nonzero(as_tuple=True)[0]
                for atom_idx in close_mask.tolist():
                    matches[atom_idx].append((seq_idx, pos_idx))

    return matches


if __name__ == "__main__":
    # 1) Load the run's metadata
    run_dir = os.path.join("artifacts", "runs", RUN_ID)
    meta_path = os.path.join(run_dir, "metadata.yaml")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No metadata.yaml found in {run_dir}")

    with open(meta_path, "r") as f:
        metadata = yaml.safe_load(f)

    model_name = metadata["model"]
    dataset_name = metadata["dataset"]
    layer_idx = metadata["layer"]
    l0 = metadata["l0"]

    # Load the dataset (for reference / text decoding)
    ds = load_dataset(dataset_name, split="train", streaming=False)
    # We might re-tokenize if the dataset had a text column,
    # but presumably your run used a stored version that already has input_ids.
    # Just ensure ds has "input_ids" for indexing. If not, adapt to your situation.
    if "text" in ds.column_names:
        # Re-tokenize for consistency
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_fn(ex):
            return tokenizer(
                ex["text"],
                max_length=metadata["seq_len"],
                truncation=True,
                padding="max_length",
            )

        ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="torch", columns=["input_ids"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 4) Load the correct layer’s activations from zarr
    act_path = os.path.join("artifacts", "data", model_name, dataset_name)
    store = zarr.DirectoryStore(act_path)
    zf = zarr.open_group(store=store, mode="r")

    layer_name = f"layer_{layer_idx}"
    if layer_name not in zf:
        raise ValueError(
            f"Could not find Zarr dataset {layer_name} in {act_path}. "
            f"Available: {list(zf.array_keys())}"
        )
    zf_layer = zf[layer_name]

    atoms_path = os.path.join(run_dir, "atoms.pt")
    atoms = torch.load(atoms_path, weights_only=True)
    atom_indices_path = os.path.join(run_dir, "atom_indices.pt")
    try:
        atom_indices = torch.load(atom_indices_path, weights_only=True)
    except FileNotFoundError:
        atom_indices = match_atoms_in_activations(
            atoms=atoms, zarr_acts=zf_layer, threshold=1e-6, batch_size=128
        )

    ito_sae = ITO_SAE(atoms, l0=l0)

# %%

if __name__ == "__main__":
    SAMPLE_IDX = 110
    TOKEN_IDX = 30
    SAMPLE_IDX = int(0.7 * len(ds)) + SAMPLE_IDX

    if SAMPLE_IDX >= zf_layer.shape[0]:
        raise IndexError(
            f"SAMPLE_IDX={SAMPLE_IDX} out of range for {zf_layer.shape[0]} sequences."
        )

    # 5) Gather partial sequence's activations (0 .. TOKEN_IDX)
    token_acts = get_token_activation(zf_layer, SAMPLE_IDX, TOKEN_IDX).to(
        ito_sae.device
    )

    # 6) Encode using the ITO_SAE
    #    The encode function expects (batch, seq_len, d_in), so unsqueeze(0).
    with torch.no_grad():
        token_acts_batch = token_acts.unsqueeze(0)  # shape (1, partial_len, d_model)
        encoded = ito_sae.encode(token_acts_batch)  # shape (1, partial_len, n_atoms)
        encoded = encoded.squeeze(0)  # shape (partial_len, n_atoms)

    # Sort descending
    top_atom_indices = torch.argsort(encoded, descending=True)[:l0]
    top_atom_values = encoded[top_atom_indices]

    # 8) Display the partial sequence with token of interest highlighted
    example = ds[SAMPLE_IDX]
    token_ids = example["input_ids"]
    partial_token_ids = token_ids[: TOKEN_IDX + 1]

    display(HTML(f"<h2>Sequence #{SAMPLE_IDX} (up to token #{TOKEN_IDX})</h2>"))
    highlight_html = highlight_tokens(
        partial_token_ids,
        active_positions_set=set(),
        special_position=TOKEN_IDX,
        highlight_color=ATOM_ACTIVATION_COLOR,
        special_color=TOKEN_OF_INTEREST_COLOR,
        tokenizer=tokenizer,
    )
    display(highlight_html)
    display(HTML("<hr/>"))

    # 9) Show top dictionary atoms, origin snippet, and highlight where they activate
    #    in the partial sequence. For each top atom, find positions in the partial sequence
    #    where the absolute activation > some threshold, or simply nonzero.
    for rank, (atom_idx, score) in enumerate(
        zip(top_atom_indices, top_atom_values), start=1
    ):
        # Gather positions for partial_acts where activation is big
        threshold = 1e-4
        this_atom_acts = encoded[atom_idx]
        if this_atom_acts.abs() <= threshold:
            continue

        origin_snippet = gather_atom_origin_snippet(
            atom_idx.item(), atom_indices, ds, tokenizer, context=10
        )

        display(
            HTML(
                f"<b>#{rank} Atom {atom_idx.item()} Act {score:.4f}</b>): {origin_snippet}"
            )
        )

# %%

# %%
# In this cell, we add a method to find the top activating samples for a given latent (atom) index.
# Inspired by diff.py, we collect all encoded activations in a single sparse tensor, where the
# sparse indices represent (sequence_idx, token_idx, atom_idx). Then we can retrieve the top
# activating positions for any chosen atom.

import torch
from tqdm import tqdm
from IPython.display import HTML, display


def get_all_activations_sparse(
    sae,
    zf_layer,
    batch_size=128,
):
    """
    Encode the entire dataset's activations into the SAE’s representation
    and store them in a sparse tensor of shape [num_positions, n_atoms],
    with an expanded 3-row indices for (seq_idx, token_idx, atom_idx).

    zf_layer: A Zarr array of shape (num_sequences, seq_len, d_model).
    sae:      An ITO_SAE (or compatible) model with .encode() -> (B, n_atoms).

    Returns: A sparse tensor with:
      - all_acts.indices(): shape [3, nnz], representing (seq_idx, token_idx, atom_idx).
      - all_acts.values(): shape [nnz], the corresponding activation magnitude.

    This function takes inspiration from `diff.py` but adapts to the Zarr-based dataset,
    converting each batch's dense SAE-encoded output to a 3D index layout.
    """
    device = sae.device
    num_sequences, seq_len, d_model = zf_layer.shape

    # We'll accumulate all partial sparse results, then torch.cat them at the end.
    all_sparse_pieces = []
    row_arange = torch.arange(seq_len, device=device)

    for start_idx in tqdm(
        range(0, num_sequences, batch_size), desc="Encoding to sparse"
    ):
        end_idx = min(start_idx + batch_size, num_sequences)
        # (this_batch, seq_len, d_model)
        chunk_np = zf_layer[start_idx:end_idx]
        chunk_acts = torch.from_numpy(chunk_np).to(device)

        # Flatten to [this_batch * seq_len, d_model]
        bsize = chunk_acts.size(0)
        chunk_acts = chunk_acts.view(-1, d_model)  # shape: [bsize * seq_len, d_model]

        # Encode with SAE -> shape: [bsize * seq_len, n_atoms]
        encoded = sae.encode(chunk_acts)  # dense

        # Convert to sparse: indices -> shape [2, nnz], values -> [nnz]
        sparse_encoded = encoded.to_sparse().coalesce()

        # We must expand the row dimension (which is in [0..bsize*seq_len-1]) into (seq_idx, token_idx).
        # row_idx = row_in_flat
        row_idx = sparse_encoded.indices()[0]  # [nnz]
        col_idx = sparse_encoded.indices()[1]  # [nnz], i.e. which atom

        # Convert row_idx into (seq_idx, token_idx)
        # seq_idx = start_idx + row_idx // seq_len
        # tok_idx = row_idx % seq_len
        seq_idx = (row_idx // seq_len) + start_idx
        tok_idx = row_idx % seq_len

        # Build new 3-row indices: [3, nnz]
        new_indices = torch.stack([seq_idx, tok_idx, col_idx], dim=0)

        # Create a new sparse tensor with these 3-row indices
        # shape logically is [num_sequences, seq_len, n_atoms], but we'll keep it 2D in .to_sparse()
        piece_3d = torch.sparse_coo_tensor(
            new_indices,
            sparse_encoded.values(),
            size=(num_sequences, seq_len, encoded.size(-1)),  # n_atoms
            dtype=encoded.dtype,
            device=encoded.device,
        ).coalesce()

        all_sparse_pieces.append(piece_3d)

    # Now concatenate the pieces along the nnz dimension. We can do that by
    # collecting their .indices() and .values(), then building one big coo_tensor.
    # Then we'll coalesce to combine duplicates if any.
    all_indices = []
    all_values = []
    for sp in all_sparse_pieces:
        all_indices.append(sp.indices())
        all_values.append(sp.values())

    cat_indices = torch.cat(all_indices, dim=1)
    cat_values = torch.cat(all_values, dim=0)

    # Build final coalesced result
    all_acts = torch.sparse_coo_tensor(
        cat_indices,
        cat_values,
        size=(num_sequences, seq_len, sae.atoms.size(0)),
        dtype=cat_values.dtype,
        device=device,
    ).coalesce()

    return all_acts


def highlight_string(
    tokens, idx, tokenizer, crop=10, highlight_color="#144B39", prefix=""
):
    """
    Utility to create HTML with a highlighted token at index `idx`.
    We'll decode the region tokens[idx-crop : idx+crop+1], highlight the center token.
    """
    start_i = max(0, idx - crop)
    end_i = min(len(tokens), idx + crop + 1)

    out_html = prefix
    for i in range(start_i, end_i):
        token_str = tokenizer.decode([tokens[i]]).replace("\n", "\\n")
        if i == idx:
            out_html += f'<span style="background-color: {highlight_color}; font-weight: bold;">\hl{{{token_str}}}</span>'
        else:
            out_html += token_str
    return HTML(out_html)


if __name__ == "__main__":
    acts_path = os.path.join(run_dir, "all_acts.pt")
    if os.path.exists(acts_path):
        all_acts = torch.load(acts_path)
    else:
        all_acts = get_all_activations_sparse(ito_sae, zf_layer, batch_size=64)
        torch.save(all_acts, acts_path)

# %%


import random
import torch
from IPython.display import HTML, display


def print_top_activating_samples(
    all_acts,
    atom_idx,
    ds,
    tokenizer,
    n=10,
    val_range=(0.0, float("inf")),
    highlight_color="#144B39",
    random_sample=True,
):
    """
    From a sparse activation tensor `all_acts` with indices [3, nnz]:
      row 0 -> seq_idx
      row 1 -> token_idx
      row 2 -> atom_idx
    and values -> activation magnitudes,
    display up to `n` (seq_idx, token_idx) positions with activations for `atom_idx`
    that fall within the activation range `val_range = (min_val, max_val)`.

    If `random_sample=True`, pick a random subset of size `n` from all matching samples.
    Otherwise, display the top `n` by descending activation.

    Args:
        all_acts:        The sparse encoding tensor (coalesced) of shape
                         [num_seqs, seq_len, n_atoms] in COO format.
        atom_idx (int):  Index of the atom whose activations we want to inspect.
        ds:              The dataset (or subset) containing "input_ids".
        tokenizer:       The tokenizer used to decode integer token IDs.
        n (int):         Number of samples to display.
        val_range (tuple): (min_val, max_val) range for activation values
                           that should be included.
        highlight_color: The background color for highlighting the activated token.
        random_sample:   Whether to take a random sample of the matches or the top n.
    """
    idx_matrix = all_acts.indices()  # shape [3, nnz]
    val_matrix = all_acts.values()  # shape [nnz]

    # Filter only entries corresponding to this atom index
    is_this_atom = idx_matrix[2] == atom_idx

    # Filter by the activation range [min_val, max_val]
    min_val, max_val = val_range
    is_in_range = (val_matrix >= min_val) & (val_matrix <= max_val)

    # Combine both masks
    mask = is_this_atom & is_in_range

    if not mask.any():
        print(f"No activations for atom {atom_idx} within range {val_range}")
        return

    # Filter
    these_indices = idx_matrix[:, mask]  # shape [3, M]
    these_values = val_matrix[mask]  # shape [M]

    # Sort by descending activation so we can pick top ones
    sorted_order = torch.argsort(these_values, descending=True)
    sorted_indices = these_indices[:, sorted_order]
    sorted_values = these_values[sorted_order]

    # Decide how to pick the final subset
    # 1) If random_sample=True, we'll randomly select `n` from all matching
    # 2) Otherwise, just take the top `n`
    if random_sample:
        # Make sure we don't sample more than we have
        total_matches = sorted_indices.shape[1]
        n_to_take = min(n, total_matches)
        random_indices = random.sample(range(total_matches), n_to_take)
        final_indices = sorted_indices[:, random_indices]
        final_values = sorted_values[random_indices]
    else:
        final_indices = sorted_indices[:, :n]
        final_values = sorted_values[:n]

    # For clarity, sort final picks by descending activation again
    # so they appear from highest -> lowest
    final_order = torch.argsort(final_values, descending=True)
    final_indices = final_indices[:, final_order]
    final_values = final_values[final_order]

    # Display results
    snippets = []
    for act_val, (seq_i, tok_i, _) in zip(final_values, final_indices.T):
        seq_i = seq_i.item()
        tok_i = tok_i.item()

        snippet_html = highlight_string(
            ds[seq_i]["input_ids"],
            tok_i,
            tokenizer,
            crop=10,
            highlight_color=highlight_color,
            prefix=f"<b>{act_val:.3f}</b> (Seq {seq_i}, Tok {tok_i}): ",
        )
        snippets.append(snippet_html)

    display(*snippets)


def show_histogram_for_atom(all_acts, atom_idx, bins=50):
    """
    Plots a histogram of all activation values for the given `atom_idx`
    from the sparse tensor `all_acts`, and displays the percentage of
    inputs on which this atom is active.
    """
    # all_acts is a sparse tensor with shape [B, T, A] (Batch, Position, Atom)
    # its .indices() has shape [3, nnz]
    # its .values() has shape [nnz]
    idx_matrix = all_acts.indices()  # shape [3, nnz]
    val_matrix = all_acts.values()   # shape [nnz]

    # --- 1) Identify which entries belong to the requested atom ---
    atom_mask = (idx_matrix[2] == atom_idx)         # which nonzero entries are for this specific atom
    these_values = val_matrix[atom_mask]            # the values for this atom
    these_positions = idx_matrix[:2, atom_mask]     # the (batch, position) pairs for this atom

    # --- 2) Determine how many unique (batch, position) pairs have this atom active ---
    # Convert to NumPy for easy unique operation
    these_positions_np = these_positions.detach().cpu().numpy()  # shape [2, #nonzero_for_atom]
    # unique along axis=1 means we treat each column as a separate (batch, position) pair
    unique_positions_for_atom = np.unique(these_positions_np, axis=1)
    num_unique_positions_for_atom = unique_positions_for_atom.shape[1]

    # --- 3) Total number of (batch, position) pairs in the entire dataset ---
    B, T, A = all_acts.size()  # shape is [B, T, A]
    total_positions = B * T

    # --- 4) Fraction (as a percentage) of inputs where this atom is active ---
    fraction_active = num_unique_positions_for_atom / total_positions
    pct_text = f"Active on {fraction_active * 100:.2f}% of inputs"

    # --- 5) Plot the histogram of activation values ---
    # Convert the selected activation values to NumPy for plotting
    these_values = these_values.detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(these_values, bins=bins, color="skyblue", edgecolor="black")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")

    # Add text in the top-right corner (in Axes coordinates) so it doesn't overlap the histogram
    plt.text(
        0.95, 0.95, pct_text,
        horizontalalignment='right',
        verticalalignment='top',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    atom_idx = 9110

    origin_snippet = gather_atom_origin_snippet(
        atom_idx, atom_indices, ds, tokenizer, context=40
    )
    display(HTML(origin_snippet))
    print_top_activating_samples( 
        all_acts, atom_idx, ds, tokenizer, n=5, val_range=(2, float("inf"))
    )
    print_top_activating_samples( 
        all_acts, atom_idx, ds, tokenizer, n=5, val_range=(0, 2)
    )
    print_top_activating_samples( 
        all_acts, atom_idx, ds, tokenizer, n=5, val_range=(-2., 0)
    )
    print_top_activating_samples( 
        all_acts, atom_idx, ds, tokenizer, n=5, val_range=(float("-inf"), -2.)
    )

    # Finally, call the histogram plotting function
    show_histogram_for_atom(all_acts, atom_idx, bins=50)

# %%
