import argparse
import os
import math
import zarr
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import itertools
import shutil

###############################################################################
#                           Helper: parse_layers
###############################################################################
def parse_layers(layers_str: str, max_layer: int = None):
    """
    Parse a string that may contain comma-separated integers or dash-separated ranges
    into a sorted list of unique layer indices.

    Examples:
        "2,4,5" -> [2, 4, 5]
        "1-3" -> [1, 2, 3]
        "1-3,5,7-8" -> [1, 2, 3, 5, 7, 8]

    If max_layer is provided, any layer index exceeding max_layer is ignored.
    Note that in Transformer Lens, layers are 0-based. If a user provides 1-3,
    that means blocks 1,2,3. But ensure it doesn't exceed `model.cfg.n_layers - 1`.
    """
    if not layers_str:
        return []

    parts = layers_str.split(",")
    layer_set = set()
    for p in parts:
        p = p.strip()
        if "-" in p:
            start_str, end_str = p.split("-")
            start_val, end_val = int(start_str), int(end_str)
            if start_val > end_val:
                start_val, end_val = end_val, start_val
            for val in range(start_val, end_val + 1):
                layer_set.add(val)
        else:
            layer_set.add(int(p))

    if max_layer is not None:
        layer_set = {l for l in layer_set if 0 <= l <= max_layer}

    return sorted(layer_set)


###############################################################################
#                         get_activations_tl (main function)
###############################################################################
def get_activations_tl(
    model_name: str,
    dataset_name: str,
    activations_path: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    num_examples: int,
    layers_str: str = None,
    tokenizer_name: str = None,
):
    """
    Collects 'resid_post' activations from a Transformer Lens HookedTransformer
    for up to `num_examples` text items from `dataset_name`,
    and stores them in a Zarr directory structure.

    By default, we gather all layers [0..n_layers-1] unless `layers_str` is specified
    (e.g. "0,2,4-6"). The final Zarr group has sub-datasets called "layer_{layer_idx}"
    with shape (N, seq_len, hidden_dim).
    """

    # -------------------------------------------------------------------------
    # 1. Load the model and tokenizer
    # -------------------------------------------------------------------------
    print(f"Loading model {model_name} via Transformer Lens...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    n_layers = model.cfg.n_layers
    hidden_dim = model.cfg.d_model

    # If no custom tokenizer provided, use the same name as the model
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # 2. Figure out which layers to collect
    # -------------------------------------------------------------------------
    if layers_str:
        collect_layers = parse_layers(layers_str, max_layer=n_layers - 1)
        if not collect_layers:
            print("No valid layers after parsing; aborting.")
            return
    else:
        # By default, collect all layers
        collect_layers = list(range(n_layers))

    print(f"Model has {n_layers} layers; hidden_dim={hidden_dim}")
    print(f"Will collect resid_post for layers: {collect_layers}")

    # -------------------------------------------------------------------------
    # 3. Prepare dataset
    # -------------------------------------------------------------------------
    print(f"Loading dataset {dataset_name} for {num_examples} examples...")
    ds_stream = load_dataset(dataset_name, split="train", streaming=True)

    # Collect first `num_examples` from the streaming dataset
    limited_samples = list(itertools.islice(ds_stream, num_examples))
    if len(limited_samples) == 0:
        print(f"No data available in dataset {dataset_name}!")
        return

    # Convert to an in-memory Dataset
    ds_local = Dataset.from_list(limited_samples)

    # Tokenize
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )

    # Remove original text column
    ds_local = ds_local.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_local.set_format(type="torch", columns=["input_ids"])

    if len(ds_local) == 0:
        print(f"No data left after tokenization in dataset {dataset_name}!")
        return

    # -------------------------------------------------------------------------
    # 4. Prepare Zarr directory
    # -------------------------------------------------------------------------
    save_dir = os.path.join(activations_path, model_name, dataset_name)
    if os.path.exists(save_dir):
        print(f"Activations already exist at {save_dir}, skipping.")
        return

    print(f"Creating directory {save_dir} to store Zarr arrays...")
    created_dir = False
    try:
        os.makedirs(save_dir, exist_ok=False)
        created_dir = True

        store = zarr.DirectoryStore(save_dir)
        zf = zarr.open_group(store=store, mode="w")

        # Create one dataset per layer
        dsets = {}
        for layer_idx in collect_layers:
            dset = zf.create_dataset(
                f"layer_{layer_idx}",
                shape=(0, seq_len, hidden_dim),
                maxshape=(None, seq_len, hidden_dim),
                chunks=(batch_size, seq_len, hidden_dim),
                dtype="float32",
            )
            dsets[layer_idx] = dset

        # ---------------------------------------------------------------------
        # 5. Collect activations
        # ---------------------------------------------------------------------
        # We'll create a list of hook names to pass to run_with_cache
        hook_names = [f"blocks.{ly}.hook_resid_post" for ly in collect_layers]
        max_needed_layer = max(collect_layers)

        print("Collecting activations (resid_post) via run_with_cache...")
        num_processed = 0
        total_batches = math.ceil(len(ds_local) / batch_size)

        with torch.no_grad():
            for start_idx in tqdm(range(0, len(ds_local), batch_size), total=total_batches):
                end_idx = min(start_idx + batch_size, len(ds_local))
                batch_tokens = ds_local[start_idx:end_idx]["input_ids"].to(device)

                # forward pass
                _, cache = model.run_with_cache(
                    batch_tokens,
                    stop_at_layer=max_needed_layer + 1,  # ensure we go up to max layer
                    names_filter=hook_names,
                )

                # for each layer, retrieve its 'resid_post'
                for ly in collect_layers:
                    acts = cache[f"blocks.{ly}.hook_resid_post"]  # shape [B, S, d_model]
                    # append to the Zarr dataset
                    old_size = dsets[ly].shape[0]
                    new_size = old_size + acts.size(0)
                    dsets[ly].resize((new_size, seq_len, hidden_dim))
                    dsets[ly][old_size:new_size, :, :] = acts.cpu().numpy()

                num_processed += batch_tokens.size(0)

        print(f"Done! Saved activations for {num_processed} sequences to {save_dir}.")

    except Exception as e:
        # If something fails and we created the directory, remove it
        if created_dir:
            print(f"An error occurred, removing directory {save_dir} ...")
            shutil.rmtree(save_dir, ignore_errors=True)
        raise e


###############################################################################
#                               Main / CLI
###############################################################################
if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="Model name or local path recognized by Transformer Lens.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NeelNanda/pile-10k",
        help="HuggingFace dataset name (split=train) for tokenization/activation collection.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for forward passes.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default="artifacts/data",
        help="Where to store final Zarr outputs. The final path is <activations_path>/<model>/<dataset>.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10_000,
        help="How many total examples to read from the dataset (and load into memory).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Comma-separated list or dash-separated ranges of 0-based layer indices "
            "to collect. E.g. '0,2,4-6'. If not set, all layers are collected."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Optional: override the default tokenizer. If None, use same name as --model.",
    )

    args = parser.parse_args()

    get_activations_tl(
        model_name=args.model,
        dataset_name=args.dataset,
        activations_path=args.activations_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        num_examples=args.num_examples,
        layers_str=args.layers,
        tokenizer_name=args.tokenizer_name,
    )