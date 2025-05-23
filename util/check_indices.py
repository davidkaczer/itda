"""
Checks whether the indices of the atoms and the activations of the atoms match.

At some point during development, I made a code error that resulted in the
indices of the atoms and the activations of the atoms not matching. That error
was a bit of a pain to figure out, so I'm leaving this script here in case
something similar happens again.
"""
import torch
from datasets import load_dataset
from layer_similarity import get_atoms_from_wandb_run
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import wandb

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    api = wandb.Api()
    entity = "patrickaaleask"
    project = "itda"
    runs = api.runs(f"{entity}/{project}", order="+created_at")

    for run in runs:
        if run.id != "lyi5ufYS":
            continue
        try:
            atoms, atom_indices = get_atoms_from_wandb_run(run, project)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping run {run.id}: {e}")
            continue

        model = HookedTransformer.from_pretrained(run.config["lm_name"], device=device)
        tokenizer = AutoTokenizer.from_pretrained(run.config["lm_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(run.config["dataset"], split="train", streaming=True)
        data_stream = (item["text"] for item in dataset)

        batch_size = 32
        progress = tqdm(
            total=atom_indices[:, 0].max(), desc="Collecting activations", unit="step"
        )
        activations = []
        for step in range(0, atom_indices[:, 0].max(), batch_size):
            batch = []
            for _ in range(batch_size):
                try:
                    text = next(data_stream)
                except StopIteration:
                    break
                batch.append(text)
            if not batch:
                print("Data stream exhausted.")
                break

            tokens = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=run.config["seq_len"],
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                # Forward pass with caching up to the max layer we need
                _, cache = model.run_with_cache(
                    tokens["input_ids"],
                    stop_at_layer=run.config["seq_len"] + 1,
                    names_filter=f"blocks.{run.config['layer']}.hook_resid_post",
                )
                activations.append(
                    cache[f"blocks.{run.config['layer']}.hook_resid_post"]
                )

            progress.update(len(batch))

        activations = torch.cat(activations, dim=0)

        reconstructed_atoms = activations[atom_indices[:, 0], atom_indices[:, 1]]

        # get the cosine similarities of the reconstructed atoms with the original atoms
        # and store them in a tensor
        similarities = torch.nn.functional.cosine_similarity(
            torch.from_numpy(atoms), reconstructed_atoms.cpu(), dim=1
        )
        print(similarities.mean())
