# %%
import os
import yaml
import torch
import zarr
from datasets import load_dataset
from ito_sae import ITO_SAE
from transformers import AutoTokenizer
from IPython.display import HTML, display
from tqdm import tqdm

# %%
# === USER CONFIGURATION ===

RUN_ID = "jppyqjur"

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
        decoded = tokenizer.decode([tok_ids[i]], skip_special_tokens=False).replace(
            "\n", "\\n"
        )
        if i == origin_pos_idx:
            snippet_html += f'<span style="background-color:{ORIGIN_HIGHLIGHT_COLOR}; font-weight:bold">{decoded}</span> '
        else:
            snippet_html += f"{decoded} "
    return snippet_html


# %%
def match_atoms_in_activations(
    atoms: torch.Tensor,
    zarr_acts,
    threshold: float = 1e-6,
    batch_size: int = 128
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
            atoms=atoms, 
            zarr_acts=zf_layer, 
            threshold=1e-6, 
            batch_size=128
        )

    ito_sae = ITO_SAE(atoms, l0=l0)

# %%

if __name__ == "__main__":
    SAMPLE_IDX = 3
    TOKEN_IDX = 14
    SAMPLE_IDX = int(0.7*len(ds)) + SAMPLE_IDX

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
