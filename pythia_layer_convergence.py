
# %%
import re
import wandb
from collections import defaultdict

MODEL_NAME = "EleutherAI/pythia-410m-deduped"
NUM_LAYERS = {
    "EleutherAI/pythia-70m-deduped": 6,
    "EleutherAI/pythia-160m-deduped": 12,
    "EleutherAI/pythia-410m-deduped": 24,
}[MODEL_NAME]

def get_layered_runs(entity="your-entity", project="example_saes"):
    """
    Query W&B for runs that have the tag 'layer_convergence' in the specified
    entity/project. Return a dictionary mapping layer -> list of runs (only those
    in the 'finished' state), sorted by step.
    """
    api = wandb.Api()

    # Fetch runs in this project that have the tag 'layer_convergence'
    runs = api.runs(f"{entity}/{project}", filters={"tags": "layer_convergence"})

    runs_by_layer = defaultdict(list)

    for run in runs:
        # Skip runs that are not finished successfully
        if run.state != "finished":
            continue

        layer = run.config.get("layer")
        if layer is None:
            # Skip runs that don't have a layer in config
            continue

        model_name = run.config.get("model", "")
        if not model_name.startswith(MODEL_NAME):
            # Skip runs that don't match the model name
            continue

        match = re.search(r"step(\d+)", model_name)
        if not match:
            # If we can't find a step, skip
            continue

        step = int(match.group(1))
        runs_by_layer[layer].append((step, run))

    # Sort each layer's runs by step
    for layer in runs_by_layer:
        runs_by_layer[layer].sort(key=lambda x: x[0])

    # Convert each list of (step, run) into just the run objects if desired
    layered_runs = {
        layer: [r for step, r in runs_by_layer[layer]] for layer in runs_by_layer
    }

    return layered_runs

# %%

import argparse
import torch
import wandb
import re

from train import train_ito_saes
from get_model_activations import get_activations_hf
from collections import defaultdict

if __name__ == "__main__":
    layered_runs = get_layered_runs(entity="patrickaaleask", project="example_saes")

    # Build a dict of { layer_number: set_of_completed_steps }
    completed_steps_by_layer = defaultdict(set)
    for layer, runs in layered_runs.items():
        for run in runs:
            model_name = run.config.get("model", "")
            match = re.search(r"step(\d+)", model_name)
            if match:
                step = int(match.group(1))
                completed_steps_by_layer[layer].add(step)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available. Running on CPU may be slow.")

    # For each step, check if *all* layers are already completed. 
    # If so, skip. Otherwise, get activations, train only the missing layers, then remove.
    for step in range(10_000, 150_000, 10_000):
        # Check if all layers are already done for this step
        all_layers_done = all(
            step in completed_steps_by_layer[layer] 
            for layer in range(1, NUM_LAYERS)
        )
        if all_layers_done:
            print(f"All layers complete for step={step}. Skipping activation download and training.")
            continue

        # Otherwise, we need the activations for the missing layers
        print(f"Getting activations for step={step}...")
        get_activations_hf(
            model_name=MODEL_NAME,
            revision=f"step{step}",
            dataset_name="NeelNanda/pile-10k",
            activations_path=f"artifacts/data",
            seq_len=128,
            batch_size=64,
            device=device,
            num_examples=10_000,
        )

        # Train ITO for any layer that's not done yet
        for layer in range(1, NUM_LAYERS):
            if step in completed_steps_by_layer[layer]:
                print(f"Skipping layer={layer}, step={step} (already finished).")
                continue

            print("Training layer:", layer, "Step:", step)
            args_dict = {
                "method": "ito",
                "model": f"{MODEL_NAME}__step{step}",
                "dataset": "NeelNanda/pile-10k",
                "layer": layer,
                "batch_size": 16,
                "target_loss": 0.30,
                "max_sequences": None,
                "load_run_id": None,
                "l0": 40,
                "seq_len": 128,
                "skip_validation": True,
                "max_atoms": 200_000,
            }
            args = argparse.Namespace(**args_dict)

            wandb.init(
                project="example_saes", config=vars(args), tags=["layer_convergence"]
            )
            last_id = wandb.run.id

            train_ito_saes(args, device)

            wandb.finish()

        # Remove the activations once training for this step is done
        import shutil
        print(f"Removing activations for step={step}...")
        shutil.rmtree(f"artifacts/data/{MODEL_NAME}__step{step}")


# %%

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    layered_runs = get_layered_runs(entity="patrickaaleask", project="example_saes")

    # Build a dictionary of data for plotting
    plot_data = {}
    for layer, runs in layered_runs.items():
        steps = []
        dict_sizes = []
        for run in runs:
            # Extract the step from the model name
            model_name = run.config.get("model", "")
            match = re.search(r"step(\d+)", model_name)
            if not match:
                continue
            step = int(match.group(1))

            # Replace "dictionary_size" below with the actual key name
            # you use in your run's summary or metrics for dictionary size.
            dict_size = run.summary.get("dict_size", None)
            if dict_size is not None:
                steps.append(step)
                dict_sizes.append(dict_size)

        # Sort by step so the line plot doesn't jump around
        sorted_data = sorted(zip(steps, dict_sizes), key=lambda x: x[0])
        if sorted_data:  # Only populate if there's at least one data point
            steps, dict_sizes = zip(*sorted_data)
            dict_sizes = np.array(dict_sizes).astype(float)
            dict_sizes /= dict_sizes.max()
            plot_data[layer] = [steps, dict_sizes]

    # Plot each layer's dictionary size vs. step
    plt.figure(figsize=(6, 4))
    for layer in range(1, NUM_LAYERS):
        steps, dict_sizes = plot_data[layer]
        plt.plot(steps, dict_sizes, marker="o", label=f"Layer {layer}")

    plt.xlabel("Step")
    plt.ylabel("Dictionary Size")
    plt.title("Dictionary Size vs. Training Step for Each Layer")
    plt.legend()
    plt.grid(True)
    plt.show()

# %%


import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == "__main__":
    # Dictionary to store the set of row indices for each (layer, step)
    atom_indices_data = defaultdict(dict)

    for layer, runs in layered_runs.items():
        for run in runs:
            # Extract the step number from the model name
            model_name = run.config.get("model", "")
            match = re.search(r"step(\d+)", model_name)
            if not match:
                continue
            step = int(match.group(1))

            # Download the atom_indices.pt from this run
            # -------------------------------------------------------
            # Option 1: If atom_indices.pt is logged as a file in the run,
            #           you can download it directly via:
            #
            #   atom_file = run.file("atom_indices.pt")
            #   local_path = atom_file.download(replace=True)
            #
            # Option 2: If it is an Artifact, you can do something like:
            #
            #   artifact = run.use_artifact("your_artifact_name:latest")
            #   artifact_dir = artifact.download()
            #   local_path = os.path.join(artifact_dir, "atom_indices.pt")
            #
            # Option 3: If you already have a local path structure for
            #           your runs, you might do something like:
            #
            #   local_path = f"artifacts/runs/{run.id}/atom_indices.pt"
            #
            # Adjust as needed for your workflow:
            # -------------------------------------------------------

            # Example below: if it's just stored locally in "artifacts/runs/{run_id}"
            local_path = f"artifacts/runs/{run.id}/atom_indices.pt"
            if not os.path.exists(local_path):
                # If it doesn't exist locally, try to download
                try:
                    atom_file = run.file("atom_indices.pt")
                    local_path = atom_file.download(replace=True)
                except:
                    # If there's no file or artifact, skip
                    print(f"Warning: No atom_indices.pt found for run {run.id}")
                    continue

            # Load the row indices as a set for easier intersection/union operations
            row_indices = torch.load(local_path, weights_only=True)
            # If row_indices is a Tensor, convert to set of integers
            row_indices_set = set([tuple(r) for r in row_indices.tolist()])

            # Store it in our dictionary
            atom_indices_data[layer][step] = row_indices_set

    # Now compute IoU between consecutive steps within each layer
    iou_by_layer = {}
    for layer in atom_indices_data:
        # Sort steps so we go in ascending order
        sorted_steps = sorted(atom_indices_data[layer].keys())
        if len(sorted_steps) < 2:
            continue  # Need at least 2 steps for IoU

        iou_values = []
        iou_steps = []
        for i in range(1, len(sorted_steps)):
            current_step = sorted_steps[i]
            prev_step = sorted_steps[- 1]
            set_prev = atom_indices_data[layer][prev_step]
            set_curr = atom_indices_data[layer][current_step]

            intersection = set_prev.intersection(set_curr)
            union = set_prev.union(set_curr)

            if len(union) == 0:
                # Guard against divide-by-zero if union is somehow empty
                iou = 0.0
            else:
                iou = len(intersection) / len(union)

            iou_values.append(iou)
            iou_steps.append(current_step)

        # Store steps and IoU values for later plotting
        iou_by_layer[layer] = (iou_steps, iou_values)

    # Plot the IoU vs. step for each layer
    plt.figure(figsize=(10, 6))
    for layer in range(1, NUM_LAYERS):
        steps, iou_vals = iou_by_layer[layer]
        plt.plot(steps, iou_vals, marker="o", label=f"Layer {layer}")

    plt.xlabel("Step")
    plt.ylabel("Dictionary Simiarity")
    plt.title("Sequential Dictionary Similarity for Each Layer")
    plt.grid(True)
    plt.legend()
    plt.show()
