import argparse
import fnmatch
import os
from functools import partial

import torch
import zarr
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from typing import List

from train import DATA_DIR

def get_activations(
    model_name: str,
    dataset_name: str,
    hook_name: List[str],
    activations_path: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
):
    activations_path = os.path.join(activations_path, model_name, dataset_name)
    if os.path.exists(activations_path):
        print(f"Activations already exist at {activations_path}. Skipping.")
        return

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    hook_names = model.hook_dict.keys()
    if hook_name in hook_names:
        hook_names = [hook_name]
    else:
        # assume there's a wild-card
        hook_names = fnmatch.filter(hook_names, hook_name)
    if not hook_names:
        raise ValueError(f"No hook found matching {hook_name}.")

    print(f"Collecting activations from {hook_names} in {model_name}.")

    ds = load_dataset(path=dataset_name, split="train", streaming=False)
    token_ds = tokenize_and_concatenate(
        dataset=ds,
        tokenizer=model.tokenizer,
        streaming=False,
        max_length=seq_len,
        add_bos_token=True,
    )
    tokens = token_ds["tokens"].to(device)

    store = zarr.DirectoryStore(activations_path)
    zf = zarr.open_group(store=store, mode="w")
    dsets = {hook_name: None for hook_name in hook_names}

    def cache_activations(hook_name, dsets, acts, hook=None):
        acts_cpu = acts.detach().cpu().numpy()
        if dsets[hook_name] is None:
            shape = (0,) + acts_cpu.shape[1:]
            max_shape = (None,) + acts_cpu.shape[1:]
            chunk_shape = (batch_size,) + acts_cpu.shape[1:]
            dsets[hook_name] = zf.create_dataset(
                hook_name,
                shape=shape,
                maxshape=max_shape,
                chunks=chunk_shape,
                dtype=acts_cpu.dtype,
            )
        old_size = dsets[hook_name].shape[0]
        new_size = old_size + acts_cpu.shape[0]
        dsets[hook_name].resize((new_size,) + dsets[hook_name].shape[1:])
        dsets[hook_name][old_size:new_size, ...] = acts_cpu

    model.remove_all_hook_fns()
    for hook_name in hook_names:
        model.add_hook(hook_name, partial(cache_activations, hook_name, dsets))

    for chunk in tqdm(torch.split(tokens, batch_size), desc="Collecting Activations"):
        model(chunk)

    model.remove_all_hook_fns()
    del model, ds, token_ds, tokens
    return activations_path


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
        help="Model name as recognized by transformer_lens, e.g. 'gpt2', 'EleutherAI/pythia-1.4b', etc.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NeelNanda/pile-10k",
        help="HuggingFace dataset name for tokenization and activation collection, e.g. 'NeelNanda/pile-10k'.",
    )
    parser.add_argument(
        "--hook_name",
        type=str,
        default="blocks.8.hook_resid_pre",
        help="List of hook points at which to collect activations from the transformer. Can also be a regex.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    get_activations(
        model_name=args.model,
        dataset_name=args.dataset,
        hook_name=args.hook_name,
        activations_path=DATA_DIR,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
