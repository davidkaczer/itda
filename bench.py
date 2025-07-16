"""
Script for running SAE Bench evaluations on ITDAs from W&B. Uploads results to
W&B.
"""
import argparse
import json
import os
from typing import Any, Optional

import torch
import yaml
import wandb
from tqdm import tqdm

import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils

from train import ITDA
from dataclasses import dataclass

RANDOM_SEED = 42

MODEL_CONFIGS = {
    "EleutherAI/pythia-70m-deduped": {
        "batch_size": 32,
        "dtype": "float32",
        "layers": [3, 4],
        "d_model": 512,
    },
    "EleutherAI/pythia-160m-deduped": {
        "batch_size": 32,
        "dtype": "float32",
        "layers": [8],
        "d_model": 512,
    },
    "google/gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [5, 12, 19],
        "d_model": 2304,
    },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
}

@dataclass
class GenericSAEConfig:
    model_name: str
    dtype: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    prepend_bos: bool
    normalize_activations: str
    dataset_trust_remote_code: bool
    seqpos_slice: tuple
    device: str

def run_evals(
    model_name: str,
    selected_saes: list[tuple[str, Any]],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    api_key: Optional[str] = None,
    force_rerun: bool = False,
    save_activations: bool = False,
):
    """Run selected evaluations for the given model and SAEs."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                output_folders["absorption"],
                force_rerun,
            )
        ),
        "autointerp": (
            lambda: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,
                output_folders["autointerp"],
                force_rerun,
            )
        ),
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=4,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=output_folders["core"],
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # subfolder scr
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # subfolder tpp
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                output_folders["sparse_probing"],
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "unlearning": (
            lambda: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name="gemma-2-2b-it",
                    random_seed=RANDOM_SEED,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size // 8,
                ),
                selected_saes,
                device,
                output_folders["unlearning"],
                force_rerun,
            )
        ),
    }

    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            print("Skipping (example) unlearning for now... remove or handle if needed.")
            continue

        print(f"\nRunning {eval_type} evaluation...\n")
        runner = eval_runners.get(eval_type)
        if runner is None:
            print(f"No runner for eval_type {eval_type}!")
            continue

        os.makedirs(output_folders[eval_type], exist_ok=True)
        runner()


def flatten_dict(d, parent_key="", sep="/"):
    """
    Recursively flatten a nested dictionary.
    For example, {'a': {'b': 1, 'c': 2}} -> {'a/b': 1, 'a/c': 2}.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="W&B run ID (or artifact prefix).")
    parser.add_argument(
        "--eval_types", nargs="+", default=["core"], help="List of evaluation types."
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force re-running all evals."
    )
    args = parser.parse_args()

    # Set up environment
    device = general_utils.setup_environment()

    eval_types = args.eval_types
    # If autointerp is requested, try loading an API key
    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            api_key = None
    else:
        api_key = None

    api = wandb.Api()
    run = api.run(f"patrickaaleask/itda/{args.run_id}")
    artifact_name = [a.name for a in run.logged_artifacts() if "ITDA" in a.name][0]

    wandb.init(
        project="itda",
        id=args.run_id,
        resume="allow",
    )
    print(f"Downloading artifact '{artifact_name}' from W&B...")
    artifact = wandb.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")

    meta_path = os.path.join(artifact_dir, "metadata.yaml")
    atoms_path = os.path.join(artifact_dir, "atoms.pt")
    atom_indices_path = os.path.join(artifact_dir, "atom_indices.pt")

    if not (os.path.exists(meta_path) and os.path.exists(atoms_path) and os.path.exists(atom_indices_path)):
        raise FileNotFoundError(
            "Could not find one of [metadata.yaml, atoms.pt, atom_indices.pt] in the artifact."
        )

    with open(meta_path, "r") as f:
        sae_metadata = yaml.safe_load(f)

    if sae_metadata["lm_name"] not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {sae_metadata['lm_name']}")

    model_config = MODEL_CONFIGS[sae_metadata["lm_name"]]
    d_model = model_config["d_model"]
    llm_batch_size = model_config["batch_size"]
    llm_dtype = model_config["dtype"]

    # This is the actual layer used in training—must match what your training code produced
    hook_layer = sae_metadata["layer"]

    # Load the actual ITDA dictionary
    print("Loading atoms and atom_indices from artifact directory...")
    atoms = torch.load(atoms_path, map_location=device)
    atom_indices = torch.load(atom_indices_path, map_location=device)

    # Construct the ITDA (or ITO_SAE) object
    # If you are actually using ITO_SAE, be sure to instantiate that class the same way:
    itda = ITDA(
        atoms=atoms,
        atom_indices=atom_indices,
        k=sae_metadata["k"],  # or whatever is stored
        cfg=GenericSAEConfig(
            model_name=sae_metadata["lm_name"],
            dtype=llm_dtype,
            d_in=d_model,
            d_sae=atoms.size(0),
            hook_layer=hook_layer,
            hook_name=f"blocks.{hook_layer}.hook_resid_post",
            prepend_bos=True,
            normalize_activations="none",
            dataset_trust_remote_code=True,
            seqpos_slice=(None,),
            device=device,
        ),
    )
    itda.normalize_decoder()

    # Put it into a list for SAE-Bench style calls
    selected_saes = [(args.run_id, itda)]

    ############################################################################
    # 2. Run your evaluations
    ############################################################################

    run_evals(
        sae_metadata["lm_name"],
        selected_saes,
        llm_batch_size // 32,  # or use exactly llm_batch_size if desired
        llm_dtype,
        device,
        eval_types=args.eval_types,
        api_key=api_key,
        force_rerun=args.force_rerun,
        save_activations=True,
    )

    ############################################################################
    # 3. Upload the evaluation results back to W&B
    ############################################################################

    for eval_type in args.eval_types:
        result_path = os.path.join(
            "eval_results", eval_type, f"{args.run_id}_custom_sae_eval_results.json"
        )
        if not os.path.isfile(result_path):
            print(f"No results found for {eval_type} at {result_path}")
            continue

        # Optionally log results as artifacts
        artifact_name = f"eval_{eval_type}"
        artifact = wandb.Artifact(name=artifact_name, type="evaluation")
        artifact.add_file(result_path)
        wandb.log_artifact(artifact)
        print(f"Uploaded {result_path} to W&B as artifact '{artifact_name}'.")

        # If you have metrics to log directly:
        with open(result_path, "r") as f:
            results = json.load(f)
        eval_result_metrics = results.get("eval_result_metrics", {})
        if eval_result_metrics:
            flattened_metrics = flatten_dict(eval_result_metrics, parent_key=eval_type)
            wandb.log(flattened_metrics)

    wandb.finish()
