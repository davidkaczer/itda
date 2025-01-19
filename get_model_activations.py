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


def get_activations_hf(
    model_name: str,
    dataset_name: str,
    activations_path: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    num_examples: int,
):
    """
    Collect hidden states (residual-stream-like) from a Hugging Face model
    for up to `num_examples` items, and store them in a Zarr directory.

    We'll produce one dataset per layer, named 'layer_{i}', where i starts
    at 1 for the hidden states after the first layer, up to `num_layers`.

    Args:
        model_name (str): Hugging Face model name or path.
        dataset_name (str): Hugging Face dataset to load (split=train).
        activations_path (str): Path to directory in which to store Zarr outputs.
        seq_len (int): Sequence length for tokenization.
        batch_size (int): Batch size for forward passes.
        device (torch.device): Device to run on (cpu or cuda).
        num_examples (int): How many examples to process in total.
    """

    # Prepare final directory where Zarr data will be stored
    save_dir = os.path.join(activations_path, model_name, dataset_name)

    # If this directory already exists, assume we've done it before and skip
    if os.path.exists(save_dir):
        print(f"Activations already exist at {save_dir}, skipping.")
        return

    # -------------------------------------------------------------------------
    # 1. Load/prepare data first (so we don't create the folder until successful)
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
    # 2. Load model + run a probe pass
    # -------------------------------------------------------------------------
    print(f"Loading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Probe pass to figure out dimensions
    probe_input = ds_local[0]["input_ids"].unsqueeze(0).to(device)
    with torch.no_grad():
        probe_out = model(probe_input, output_hidden_states=True)
    probe_hs = probe_out.hidden_states
    num_layers = len(probe_hs) - 1  # ignoring hidden_states[0] if you want post-layer
    hidden_dim = probe_hs[-1].size(-1)
    print(f"Model has {num_layers} layers. Hidden size = {hidden_dim}")

    # -------------------------------------------------------------------------
    # 3. Create folder only after successful data/model prep; if something
    #    fails *after* creation, we clean up.
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
        dsets = []
        for layer_idx in range(1, num_layers + 1):
            layer_dset = zf.create_dataset(
                f"layer_{layer_idx}",
                shape=(0, seq_len, hidden_dim),
                maxshape=(None, seq_len, hidden_dim),
                chunks=(batch_size, seq_len, hidden_dim),
                dtype="float32",
            )
            dsets.append(layer_dset)

        # ---------------------------------------------------------------------
        # 4. Collect hidden states
        # ---------------------------------------------------------------------
        print("Collecting hidden states ...")
        num_processed = 0
        total_batches = math.ceil(len(ds_local) / batch_size)

        with torch.no_grad():
            for start_idx in tqdm(range(0, len(ds_local), batch_size), desc="Batches", total=total_batches):
                end_idx = min(start_idx + batch_size, len(ds_local))
                batch = ds_local[start_idx:end_idx]["input_ids"].to(device)
                outputs = model(batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # hidden_states[1:] are the post-layer states for layers 1..num_layers
                for layer_idx, layer_acts in enumerate(hidden_states[1:], start=1):
                    old_size = dsets[layer_idx - 1].shape[0]
                    new_size = old_size + layer_acts.size(0)
                    dsets[layer_idx - 1].resize((new_size, seq_len, hidden_dim))
                    dsets[layer_idx - 1][old_size:new_size, :, :] = layer_acts.cpu().numpy()

                num_processed += batch.size(0)

        print(f"Done! Saved hidden states for {num_processed} sequences to {save_dir}.")

    except Exception as e:
        # If we created the directory during this run, clean up
        if created_dir:
            print(f"An error occurred, removing directory {save_dir} ...")
            shutil.rmtree(save_dir, ignore_errors=True)
        raise e  # re-raise so you see the actual error


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model name or path, e.g. 'gpt2', 'EleutherAI/pythia-1.4b', etc.",
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
    args = parser.parse_args()

    get_activations_hf(
        model_name=args.model,
        dataset_name=args.dataset,
        activations_path=args.activations_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        num_examples=args.num_examples,
    )
