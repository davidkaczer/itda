import argparse
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch
from layer_similarity import (
    get_atom_indices_from_wandb_run,
    get_layered_runs_for_models,
    get_similarity_measure,
)
from transformer_lens import HookedTransformer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser(
        description="Get the ITDA IoU similarity between a set of models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        required=True,
        help="List of model names to compare.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="patrickaaleask",
        help="Entity (user or team) under which the W&B project lives.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="itda",
        help="Name of the W&B project for logging or fetching runs.",
    )
    args = parser.parse_args()

    atom_unions = {}
    for model_name in args.models:
        model = HookedTransformer.from_pretrained(model_name, device=device)
        n_layers = model.cfg.n_layers
        existing_itdas = get_layered_runs_for_models(
            [model_name],
            list(range(1, n_layers)),
            entity=args.wandb_entity,
            project=args.wandb_project,
        )[model_name]

        all_atom_indices = []
        for run in existing_itdas.values():
            atom_indices = get_atom_indices_from_wandb_run(run, args.wandb_project)
            all_atom_indices.append(atom_indices)
        all_atom_indices = np.concatenate(all_atom_indices, axis=0)

        atom_unions[model_name] = all_atom_indices

    similarities = np.array(
        [
            get_similarity_measure(atom_unions[model1], atom_unions[model2])
            for model1, model2 in product(args.models, args.models)
        ]
    )

    similarities = similarities.reshape(len(args.models), len(args.models))

    fig, ax = plt.subplots()
    cax = ax.matshow(similarities, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(args.models)))
    ax.set_yticks(np.arange(len(args.models)))
    ax.set_xticklabels(args.models, rotation=90)
    ax.set_yticklabels(args.models)

    # TODO: The models names are long so this is not lovely
    for i in range(len(args.models)):
        for j in range(len(args.models)):
            ax.text(j, i, f"{similarities[i, j]:.2f}", ha="center", va="center", color="white")

    plt.tight_layout()
    # TODO: this needs to be configurable
    plt.savefig(f"artifacts/similarities/model_sim.png")