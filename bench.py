import argparse
import pickle
import json
import os
from typing import Any, Optional

import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils
from dictionary_learning.dictionary import AutoEncoder
import torch
import yaml
from ito_sae import ITO_SAE, ITO_SAEConfig
from tqdm import tqdm

import wandb

RANDOM_SEED = 42

MODEL_CONFIGS = {
    "EleutherAI/pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3, 4],
        "d_model": 512,
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

    # Mapping of eval types to their functions and output paths
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
                "eval_results/absorption",
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
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        # TODO: Do a better job of setting num_batches and batch size
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
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
                "eval_results",  # We add scr or tpp depending on perform_scr
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
                "eval_results",  # We add scr or tpp depending on perform_scr
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
                "eval_results/sparse_probing",
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
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            if model_name != "gemma-2-2b":
                print("Skipping unlearning evaluation for non-GEMMA model")
                continue
            print("Skipping, need to clean up unlearning interface")
            continue  # TODO:
            if not os.path.exists(
                "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            ):
                print(
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        if eval_type in eval_runners:
            os.makedirs(output_folders[eval_type], exist_ok=True)
            eval_runners[eval_type]()


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
    parser.add_argument("--run_id", type=str, help="Run ID for ITO_SAE")
    parser.add_argument(
        "--eval_types", nargs="+", default=["core"], help="List of eval types"
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force re-running all evals"
    )
    args = parser.parse_args()

    device = general_utils.setup_environment()

    eval_types = args.eval_types
    # If autointerp is requested, load API key if present
    if "autointerp" in eval_types:
        raise NotImplementedError("Autointerp evaluation is not yet supported")
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            api_key = None
    else:
        api_key = None

    # Load model configs

    selected_saes = []

    run_dir = os.path.join("artifacts", "runs", args.run_id)
    meta_path = os.path.join(run_dir, "metadata.yaml")
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"No run found at {run_dir}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata.yaml found in {run_dir}")

    with open(meta_path, "r") as f:
        sae_metadata = yaml.safe_load(f)

    model_config = MODEL_CONFIGS[sae_metadata["model"]]
    d_model = model_config["d_model"]
    llm_batch_size = model_config["batch_size"]
    llm_dtype = model_config["dtype"]
    hook_layer = model_config["layers"][0]

    if ("method" not in sae_metadata) or (sae_metadata["method"] != "dictlearn"):
        atoms_path = os.path.join(run_dir, "atoms.pt")
        if not os.path.exists(atoms_path):
            raise FileNotFoundError("No atoms.pt found for the specified ITO run")

        atoms = torch.load(atoms_path).to(device)
        ito_sae = ITO_SAE(
            atoms,
            l0=sae_metadata["l0"],
            cfg=ITO_SAEConfig(
                model_name=sae_metadata["model"],
                dtype=llm_dtype,
                d_in=d_model,
                d_sae=atoms.size(0),
                hook_layer=sae_metadata["layer"],
                hook_name=f"blocks.{hook_layer}.hook_resid_post",  # Assuming same hook name as GPT-2 SAEs
                prepend_bos=True,
                normalize_activations="none",
                dataset_trust_remote_code=True,
                seqpos_slice=(None,),
                device=device,
            ),
        )
        ito_sae.normalize_decoder()
        selected_saes.append((args.run_id, ito_sae))
    else:
        # load pickle of ae from run_dir, "ae.pt"
        with open(os.path.join(run_dir, "ae.pt"), "rb") as f:
            ae = pickle.load(f)
        ae.cfg = ITO_SAEConfig(
            model_name=sae_metadata["model"],
            dtype=llm_dtype,
            d_in=d_model,
            d_sae=ae.W_dec.size(0),
            hook_layer=sae_metadata["layer"],
            hook_name=f"blocks.{hook_layer}.hook_resid_post",  # Assuming same hook name as GPT-2 SAEs
            prepend_bos=True,
            normalize_activations="none",
            dataset_trust_remote_code=True,
            seqpos_slice=(None,),
            device=device,
        )
        selected_saes.append((args.run_id, ae))

    if not selected_saes:
        raise ValueError(
            "No SAEs selected. Provide either an ITO run ID or use --pretrained_saes."
        )

    run_evals(
        sae_metadata["model"],
        selected_saes,
        llm_batch_size,
        llm_dtype,
        device,
        eval_types=eval_types,
        api_key=api_key,
        force_rerun=args.force_rerun,
    )

    # Upload the evaluation results to the wandb run of the ITO_SAE
    wandb.init(
        project="example_saes",
        id=args.run_id,
        resume="allow",
    )

    for eval_type in eval_types:
        result_path = os.path.join(
            "eval_results", eval_type, f"{args.run_id}_custom_sae_eval_results.json"
        )
        if not os.path.isfile(result_path):
            print(f"Could not find results for {eval_type} at {result_path}.")
            continue

        artifact_name = f"eval_{eval_type}"
        artifact = wandb.Artifact(name=artifact_name, type="evaluation")
        artifact.add_file(result_path)
        wandb.log_artifact(artifact)
        print(f"Uploaded {result_path} to wandb as artifact '{artifact_name}'.")

        if eval_type != "core":
            continue

        with open(result_path, "r") as f:
            results = json.load(f)

        eval_result_metrics = results.get("eval_result_metrics", {})
        if eval_result_metrics:
            flattened_metrics = flatten_dict(eval_result_metrics, parent_key=eval_type)
            wandb.log(flattened_metrics)

    wandb.finish()
