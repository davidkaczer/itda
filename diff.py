# %%
import os
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

RUNS_DIR = "artifacts/runs"
BASE_MODEL = "Qwen/Qwen2-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2-0.5B-Instruct"

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = os.listdir(RUNS_DIR)
    metadata = {}
    for run in runs:
        with open(f"{RUNS_DIR}/{run}/metadata.yaml", "r") as f:
            metadata[run] = yaml.safe_load(f)

    # Get a dictionary of model names to a list of runs sorted by layer
    from collections import defaultdict

    model_dict = defaultdict(list)
    for id_, details in metadata.items():
        model = details["model"]
        layer_num = int(details["hook_name"].split(".")[1])
        model_dict[model].append((layer_num, id_))
    sorted_model_dict = {
        model: [id_ for _, id_ in sorted(entries)]
        for model, entries in model_dict.items()
    }

# %%

if __name__ == "__main__":
    # Load atoms.pt for each layer of each model
    import torch

    atoms_dict = {}
    for model, runs in sorted_model_dict.items():
        atoms_dict[model] = []
        for run in runs:
            with open(f"{RUNS_DIR}/{run}/atoms.pt", "rb") as f:
                atoms = torch.load(f).unique(dim=0).cpu()
                atoms_dict[model].append(atoms)

# %%


def batched_pairwise_cosine_similarity(
    tensor1: torch.Tensor, tensor2: torch.Tensor, batch_size: int = 1024
) -> torch.Tensor:
    tensor1_norm = tensor1 / tensor1.norm(dim=1, keepdim=True)
    tensor2_norm = tensor2 / tensor2.norm(dim=1, keepdim=True)

    N, M = tensor1_norm.size(0), tensor2_norm.size(0)
    result = torch.empty(N, M, device=tensor1.device)

    for i in range(0, N, batch_size):
        tensor1_batch = tensor1_norm[i : i + batch_size]
        result[i : i + batch_size] = torch.matmul(tensor1_batch, tensor2_norm.T)

    return result


if __name__ == "__main__":
    max_cosine_vals_per_layer = []
    num_layers = len(atoms_dict[BASE_MODEL])  # typically 24

    # Collect max cosine similarity arrays for each layer
    for layer in range(num_layers):
        atoms_base = atoms_dict[BASE_MODEL][layer].to(device)
        atoms_instruct = atoms_dict[INSTRUCT_MODEL][layer].to(device)

        cosine_similarity = batched_pairwise_cosine_similarity(
            atoms_base, atoms_instruct
        )
        max_cosine_similarity, _ = torch.max(cosine_similarity, dim=1)
        max_cosine_vals_per_layer.append(max_cosine_similarity.cpu().numpy())

    # Plot violin plots for all layers side by side
    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(
        dataset=max_cosine_vals_per_layer,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )

    # Color the violins
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    if "cmeans" in parts:
        parts["cmeans"].set_edgecolor("darkblue")

    ax.set_xticks(range(1, num_layers + 1))
    ax.set_xticklabels([str(i) for i in range(num_layers)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Cosine Similarity")
    ax.set_title("Distribution of Max Cosine Similarities per Layer (Violin Plot)")

    plt.tight_layout()
    plt.show()

    torch.cuda.empty_cache()

# %%


def find_atom_occurrences_vectorized(
    model_name: str,
    dataset_name: str,
    hook_name: str,
    atoms: torch.Tensor,
    batch_size: int = 32,
    seq_len: int = 128,
    device: str = "cuda",
    chunk_size: int = 64,
):
    """
    Searches for the *first* occurrence of each activation pattern (atom)
    in the dataset at a specific hook point, using a chunked vectorized approach.

    Args:
        model_name: Name of the model to load via HookedTransformer.
        dataset_name: Hugging Face dataset name.
        hook_name: The name of the hook to attach for capturing activations.
        atoms: A tensor of shape [num_atoms, d_model] containing all “atoms”.
        batch_size: How many sequences to process per forward pass.
        seq_len: Sequence length used in tokenization.
        device: PyTorch device ("cuda" or "cpu").
        rtol, atol: Tolerances for approximate matching (unused in exact example below).
        chunk_size: How many atoms to broadcast at once in the vectorized comparison.

    Returns:
        A dict of {atom_index: (sequence_idx, token_idx) or None},
        indicating where each atom was first found in the dataset.
    """

    # 1) Load model
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # 2) Prepare to capture activations via hook
    global_activations = None

    def capture_activations(acts, hook=None):
        nonlocal global_activations
        global_activations = acts.detach()

    # Make sure no hooks are left over, then add our capture hook
    model.remove_all_hook_fns()
    model.add_hook(hook_name, capture_activations)

    # 3) Tokenize dataset
    ds = load_dataset(dataset_name, split="train", streaming=False)
    token_ds = tokenize_and_concatenate(
        dataset=ds,
        tokenizer=model.tokenizer,
        streaming=False,
        max_length=seq_len,
        add_bos_token=True,
    )
    tokens = token_ds["tokens"]  # shape [num_sequences, seq_len]
    num_sequences = tokens.shape[0]

    # 4) Bookkeeping for matched vs unmatched atoms
    matched_positions = {atom_idx: None for atom_idx in range(len(atoms))}
    unmatched_atom_indices = list(range(len(atoms)))  # dynamic list

    # 5) Loop over dataset in batches
    pbar = tqdm(range(0, num_sequences, batch_size), desc="Searching for atoms")
    for batch_start in pbar:
        if not unmatched_atom_indices:
            # All atoms matched already; we can stop
            break

        batch_end = min(batch_start + batch_size, num_sequences)
        batch_tokens = tokens[batch_start:batch_end].to(device)

        # Forward pass triggers our capture hook -> sets global_activations
        _ = model(batch_tokens)  # shape of output can be ignored

        # global_activations shape is [B, S, d_model]
        batch_acts = global_activations
        bsz, slen, d_model = batch_acts.shape

        # Flatten: [B, S, d_model] -> [B*S, d_model]
        flat_acts = batch_acts.view(-1, d_model)

        # 6) Chunked broadcast for unmatched atoms
        # We'll create a new list for any atoms that remain unmatched after this batch
        still_unmatched = []
        start_idx = 0

        while start_idx < len(unmatched_atom_indices):
            end_idx = min(start_idx + chunk_size, len(unmatched_atom_indices))
            chunk_atom_indices = unmatched_atom_indices[start_idx:end_idx]

            # Gather a chunk of atoms
            chunk_atoms = atoms[chunk_atom_indices].to(device)  # [chunk_size, d_model]

            # For exact equality matching:
            # eq_matrix = shape [B*S, chunk_size]
            eq_matrix = (flat_acts.unsqueeze(1) == chunk_atoms.unsqueeze(0)).all(dim=-1)

            # If you wanted approximate matching, you could do something like:
            #
            # diffs = (flat_acts.unsqueeze(1) - chunk_atoms.unsqueeze(0)).abs()
            # eq_matrix = diffs < (atol + rtol * chunk_atoms.unsqueeze(0).abs())
            # eq_matrix = eq_matrix.all(dim=-1)  # shape [B*S, chunk_size]

            # Find all matches: 2D indices [row, col_in_chunk]
            matched_indices = eq_matrix.nonzero(
                as_tuple=False
            )  # shape [num_matches, 2]

            # Record the earliest match for each atom in this chunk
            # Note: multiple rows might match the same atom; we only want the earliest
            earliest_row_per_atom = {}
            for row_col in matched_indices:
                row, col = row_col.tolist()  # e.g. row=17, col=3
                global_atom_idx = chunk_atom_indices[col]
                # If we haven't recorded a match for this atom yet, do so
                if global_atom_idx not in earliest_row_per_atom:
                    earliest_row_per_atom[global_atom_idx] = row

            # For each matched atom in this chunk, fill matched_positions if empty
            for atom_idx_in_chunk, row_val in earliest_row_per_atom.items():
                if matched_positions[atom_idx_in_chunk] is None:
                    # Convert flat row back to (sequence_in_batch, token_in_sequence)
                    i = row_val // slen
                    j = row_val % slen
                    seq_idx = batch_start + i
                    matched_positions[atom_idx_in_chunk] = (seq_idx, j)

            # Mark which chunk atoms got matched
            matched_atom_indices_chunk = set(earliest_row_per_atom.keys())

            # For the chunk atoms that never got matched, we still keep them unmatched
            for global_atom_idx in chunk_atom_indices:
                if global_atom_idx not in matched_atom_indices_chunk:
                    still_unmatched.append(global_atom_idx)

            start_idx = end_idx

        # After processing the batch, update unmatched_atom_indices
        unmatched_atom_indices = [
            idx for idx in still_unmatched if matched_positions[idx] is None
        ]

        # Update tqdm description with the number of unmatched atoms
        pbar.set_postfix(unmatched=len(unmatched_atom_indices))

        # Early exit if everything matched
        if not unmatched_atom_indices:
            break

    # 7) Cleanup
    model.remove_all_hook_fns()
    del model

    return matched_positions


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = 0

    atoms_base = atoms_dict[BASE_MODEL][layer].to(device)
    atoms_instruct = atoms_dict[INSTRUCT_MODEL][layer].to(device)

    cosine_similarity = batched_pairwise_cosine_similarity(atoms_base, atoms_instruct)
    max_cosine_similarity, _ = torch.max(cosine_similarity, dim=1)
    worst_k = max_cosine_similarity.topk(10, largest=False)

    layer_metadata = metadata[sorted_model_dict[INSTRUCT_MODEL][layer]]
    # This normally stops early don't worry about the tqdm runtime.
    atom_indices = find_atom_occurrences_vectorized(
        model_name=layer_metadata["model"],
        dataset_name=layer_metadata["dataset"],
        hook_name=layer_metadata["hook_name"],
        atoms=atoms_instruct[worst_k.indices],
        batch_size=32,
        seq_len=layer_metadata["seq_len"],
        device=device,
    )

# %%

if __name__ == "__main__":
    print(atom_indices)
