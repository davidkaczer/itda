import argparse
import os
import math
import zarr
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
import shutil


def parse_layers(layers_str: str, max_layer: int = None):
    """
    Parse a string that may contain comma-separated integers or dash-separated ranges
    into a sorted list of unique layer indices.

    Examples:
        "2,4,5" -> [2, 4, 5]
        "1-3" -> [1, 2, 3]
        "1-3,5,7-8" -> [1, 2, 3, 5, 7, 8]

    If max_layer is provided, any layer index exceeding max_layer is ignored.
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
                start_val, end_val = end_val, start_val  # swap if reversed
            for val in range(start_val, end_val + 1):
                layer_set.add(val)
        else:
            layer_set.add(int(p))

    # If max_layer is specified, filter out anything beyond that
    if max_layer is not None:
        layer_set = {l for l in layer_set if 1 <= l <= max_layer}

    return sorted(layer_set)


def get_activations_hf(
    model_name: str,
    revision: str,
    dataset_name: str,
    activations_path: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    num_examples: int,
    stop_forward_pass_layer: int = None,
    offload_folder: str = None,
    layers_str: str = None,
):
    """
    Collect hidden states (residual-stream-like) from a Hugging Face model
    for up to `num_examples` items, and store them in a Zarr directory.

    We'll produce one dataset per layer, named 'layer_{i}', where i starts
    at 1 for the hidden states after the first layer, up to `num_layers`.

    If `stop_forward_pass_layer` is specified (only valid for LLaMA models),
    we slice the model to only retain layers up to `stop_forward_pass_layer`.

    If `offload_folder` is specified, we use Accelerate-based offloading,
    storing the model's parameters on disk to reduce RAM usage.

    If `layers_str` is provided, only collect/store activations for those
    specific layers (e.g. '2,4,6' or '2-5,7'). If not provided, all layers
    (up to `stop_forward_pass_layer`, if set) are collected.
    """

    # Make a friendly, filesystem-safe folder name that includes model and revision
    output_model_nae = model_name
    if revision:
        output_model_nae += f"__{revision}"

    # Prepare final directory where Zarr data will be stored
    save_dir = os.path.join(activations_path, output_model_nae, dataset_name)

    # If this directory already exists, assume we've done it before and skip
    if os.path.exists(save_dir):
        print(f"Activations already exist at {save_dir}, skipping.")
        return

    # -------------------------------------------------------------------------
    # 1. Load the model
    # -------------------------------------------------------------------------
    if offload_folder:
        print(f"Offloading parameters to disk at {offload_folder} ...")

    if revision:
        print(f"Loading model {model_name} at revision={revision} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            device_map="auto" if offload_folder else None,
            offload_folder=offload_folder,
        )
    else:
        print(f"Loading model {model_name} with default branch ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if offload_folder else None,
            offload_folder=offload_folder,
        )

    if not offload_folder:
        model = model.to(device)

    # If stop_forward_pass_layer is specified, check model type and slice layers
    if stop_forward_pass_layer is not None:
        # Only allow stop_forward_pass_layer for LLaMA-based models
        if "llama" not in getattr(model.config, "model_type", "").lower():
            raise ValueError(
                "The 'stop_forward_pass_layer' argument can only be used with LLaMA-based models. "
                f"Current model_type: {model.config.model_type}"
            )
        # Slice the model layers up to stop_forward_pass_layer
        model.model.layers = model.model.layers[:stop_forward_pass_layer]

    model.eval()

    # -------------------------------------------------------------------------
    # 2. Load / prepare dataset
    # -------------------------------------------------------------------------
    print(f"Loading dataset {dataset_name} for {num_examples} examples ...")
    ds_stream = load_dataset(dataset_name, split="train", streaming=True)

    # Collect first `num_examples` from the streaming dataset
    limited_samples = list(itertools.islice(ds_stream, num_examples))
    if len(limited_samples) == 0:
        print(f"No data available in dataset {dataset_name}!")
        return

    # Convert to an in-memory Dataset
    ds_local = Dataset.from_list(limited_samples)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        )

    # Remove the original text column to save space
    ds_local = ds_local.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_local.set_format(type="torch", columns=["input_ids"])

    if len(ds_local) == 0:
        print(f"No data left after tokenization in dataset {dataset_name}!")
        return

    # -------------------------------------------------------------------------
    # 3. Probe pass to figure out dimension info
    # -------------------------------------------------------------------------
    probe_input = ds_local[0]["input_ids"].unsqueeze(0).to(device)
    with torch.no_grad():
        probe_out = model(probe_input, output_hidden_states=True)

    probe_hs = probe_out.hidden_states
    # The first element is the embedding output,
    # so hidden_states[1:] correspond to post-layer states for layers 1..num_layers.
    num_layers = len(probe_hs) - 1
    hidden_dim = probe_hs[-1].size(-1)
    print(f"Model has {num_layers} layers. Hidden size = {hidden_dim}")

    # Figure out which layers to collect
    max_possible_layer = stop_forward_pass_layer if stop_forward_pass_layer else num_layers

    if layers_str:
        collect_layers = parse_layers(layers_str, max_layer=max_possible_layer)
        if not collect_layers:
            print("No valid layers to collect after parsing, aborting.")
            return
    else:
        # If no layers_str provided, collect all layers
        collect_layers = list(range(1, max_possible_layer + 1))

    print(f"Will collect activations for layer(s): {collect_layers}")

    # -------------------------------------------------------------------------
    # 4. Create folder only after successful data/model prep
    # -------------------------------------------------------------------------
    created_dir = False
    try:
        os.makedirs(save_dir, exist_ok=False)
        created_dir = True
        print(f"Created directory {save_dir} to store activations.")

        # Prepare Zarr store
        store = zarr.DirectoryStore(save_dir)
        zf = zarr.open_group(store=store, mode="w")

        print("Creating Zarr datasets ...")
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
        # 5. Collect hidden states
        # ---------------------------------------------------------------------
        print("Collecting hidden states ...")
        num_processed = 0
        total_batches = math.ceil(len(ds_local) / batch_size)

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(ds_local), batch_size), desc="Batches", total=total_batches
            ):
                end_idx = min(start_idx + batch_size, len(ds_local))
                batch = ds_local[start_idx:end_idx]["input_ids"].to(device)
                outputs = model(batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # hidden_states[1] = after layer 1
                # hidden_states[2] = after layer 2
                # ...
                # hidden_states[num_layers] = after layer num_layers
                for layer_idx in collect_layers:
                    layer_acts = hidden_states[layer_idx]  # 1-based indexing
                    old_size = dsets[layer_idx].shape[0]
                    new_size = old_size + layer_acts.size(0)
                    dsets[layer_idx].resize((new_size, seq_len, hidden_dim))
                    dsets[layer_idx][old_size:new_size, :, :] = layer_acts.cpu().numpy()

                num_processed += batch.size(0)

        print(f"Done! Saved hidden states for {num_processed} sequences to {save_dir}.")

    except Exception as e:
        # If we created the directory during this run, clean up
        if created_dir:
            print(f"An error occurred, removing directory {save_dir} ...")
            shutil.rmtree(save_dir, ignore_errors=True)
        raise e  # Re-raise so you see the actual error


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
        help="Hugging Face model repository name or local path, e.g. 'EleutherAI/pythia-70m-deduped'.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Which HF revision/branch/tag to use from the model repository. E.g. 'step1000'.",
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
        help="Where to store final Zarr outputs. The final path is <activations_path>/<model_and_revision>/<dataset>.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10_000,
        help="How many total examples to read from the dataset (and load into memory).",
    )
    parser.add_argument(
        "--stop_forward_pass_layer",
        type=int,
        default=None,
        help="If set (only valid for LLaMA models), truncate the model to include only the first `stop_forward_pass_layer` layers.",
    )
    parser.add_argument(
        "--offload_folder",
        type=str,
        default=None,
        help="If set, offload model parameters to disk at this path (requires Accelerate).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Comma-separated list or dash-separated ranges of 1-based layer indices "
            "to collect. E.g. '1,3,5' or '2-4,6,10-12'. If not set, "
            "all layers (up to --stop_forward_pass_layer if given) are collected."
        ),
    )

    args = parser.parse_args()

    get_activations_hf(
        model_name=args.model,
        revision=args.revision,
        dataset_name=args.dataset,
        activations_path=args.activations_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        num_examples=args.num_examples,
        stop_forward_pass_layer=args.stop_forward_pass_layer,
        offload_folder=args.offload_folder,
        layers_str=args.layers,
    )
