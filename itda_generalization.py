import torch
from datasets import load_dataset
from layer_similarity import get_atoms_from_wandb_run
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from dl_train import ITDA
from layer_similarity import get_similarity_measure

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
    runs = [run for run in runs if run.id in ["ty2u9Rr9", "Ym5Gi2ze"]]
    atoms_and_indices = [get_atoms_from_wandb_run(run, project) for run in runs]

    source_model = 1
    target_model = 0

    print(
        "Source model:",
        runs[source_model].config["lm_name"],
        "Layer:",
        runs[source_model].config["layer"],
    )
    print(
        "Target model:",
        runs[target_model].config["lm_name"],
        "Layer:",
        runs[target_model].config["layer"],
    )

    def get_similarity_measure(ai1, ai2):
        ai1s = set([tuple(r) for r in ai1.tolist()])
        ai2s = set([tuple(r) for r in ai2.tolist()])
        return len(ai1s.intersection(ai2s)) / len(ai1s)

    print(
        "Atom similarity:",
        get_similarity_measure(
            atoms_and_indices[source_model][1], atoms_and_indices[target_model][1]
        ),
    )

    # get the atoms for model 1 using model 0's indices
    model = HookedTransformer.from_pretrained(
        runs[target_model].config["lm_name"], device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(runs[target_model].config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        runs[target_model].config["dataset"], split="train", streaming=True
    )
    data_stream = (item["text"] for item in dataset)

    source_atom_indices = atoms_and_indices[source_model][1]

    batch_size = 32
    progress = tqdm(
        total=source_atom_indices[:, 0].max(),
        desc="Collecting activations",
        unit="step",
    )
    activations = []
    for step in range(0, source_atom_indices[:, 0].max(), batch_size):
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
            max_length=runs[target_model].config["seq_len"],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # Forward pass with caching up to the max layer we need
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                stop_at_layer=runs[target_model].config["seq_len"] + 1,
                names_filter=f"blocks.{runs[target_model].config['layer']}.hook_resid_post",
            )
            activations.append(
                cache[f"blocks.{runs[target_model].config['layer']}.hook_resid_post"]
            )

        progress.update(len(batch))
    activations = torch.cat(activations, dim=0)
    reconstructed_atoms = activations[
        source_atom_indices[:, 0], source_atom_indices[:, 1]
    ]
    original_atoms = atoms_and_indices[target_model][0]

    reconstructed_itda = ITDA(
        reconstructed_atoms.to(device),
        torch.from_numpy(source_atom_indices).to(device),
        runs[source_model].config["k"],
        runs[source_model].config,
    )
    reconstructed_itda.normalize_decoder()
    original_itda = ITDA(
        torch.from_numpy(original_atoms).to(device),
        torch.from_numpy(atoms_and_indices[target_model][1]).to(device),
        runs[source_model].config["k"],
        runs[source_model].config,
    )
    reconstructed_itda.normalize_decoder()

    def get_error(original, reconstruction):
        eps = 1e-9
        norm_x = original.norm(dim=1, keepdim=True).clamp_min(eps)
        norm_recon = reconstruction.norm(dim=1, keepdim=True).clamp_min(eps)
        normalized_x = original / norm_x
        normalized_recon = reconstruction / norm_recon
        errors = (normalized_x - normalized_recon).pow(2).mean(dim=1)
        return errors

    recon_recon_errors = []
    orig_recon_errors = []
    for batch in tqdm(torch.split(activations, 128)):
        flatten_x = batch.flatten(end_dim=1).to(device)
        recon_recons = reconstructed_itda(flatten_x)
        original_recons = original_itda(flatten_x)

        recon_recon_error = get_error(flatten_x, recon_recons)
        recon_recon_errors.append(recon_recon_error)

        orig_recon_error = get_error(flatten_x, original_recons)
        orig_recon_errors.append(orig_recon_error)

    print("Original atoms error:", torch.cat(orig_recon_errors).mean().item())
    print("Reconstructed atoms error:", torch.cat(recon_recon_errors).mean().item())
