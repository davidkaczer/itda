import os
import argparse
import torch
from typing import Any, Optional
from tqdm import tqdm
import yaml

import evals.absorption.main as absorption
import evals.autointerp.main as autointerp
import evals.core.main as core
import evals.scr_and_tpp.main as scr_and_tpp
import evals.sparse_probing.main as sparse_probing
import evals.unlearning.main as unlearning
import sae_bench_utils.general_utils as general_utils

from example_saes.ito_sae import ITO_SAE, ITO_SAEConfig
from example_saes.train import load_model

RANDOM_SEED = 42

MODEL_CONFIGS = {
    "gpt2": {"batch_size": 512, "dtype": "float32", "layers": [8], "d_model": 768},
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
                filtered_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=False,
                compute_featurewise_weight_based_metrics=False,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=output_folders["core"],
                verbose=True,
                dtype=llm_dtype,
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
                output_folders["scr"],
                force_rerun,
                clean_up_activations=True,
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
                output_folders["tpp"],
                force_rerun,
                clean_up_activations=True,
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
            )
        ),
        # "unlearning" evals are model-specific, omitted as requested
    }

    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        if eval_type in eval_runners:
            os.makedirs(output_folders[eval_type], exist_ok=True)
            eval_runners[eval_type]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model to evaluate (currently only gpt2)")
    parser.add_argument("--ito_run_id", type=str, default=None, help="ID of the run directory for ITO_SAE")
    parser.add_argument("--pretrained_saes", action="store_true", help="Run evals on pretrained GPT-2 SAEs as well")
    parser.add_argument("--eval_types", nargs="+", default=["core", "scr", "tpp", "sparse_probing"], help="List of eval types")
    parser.add_argument("--force_rerun", action="store_true", help="Force re-running all evals")
    args = parser.parse_args()

    device = general_utils.setup_environment()

    model_name = args.model
    if model_name not in MODEL_CONFIGS:
        raise ValueError("Unsupported model. Currently only gpt2 is supported.")

    eval_types = args.eval_types

    # If autointerp is requested, load API key if present
    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            api_key = None
    else:
        api_key = None

    # Load model configs
    model_config = MODEL_CONFIGS[model_name]
    d_model = model_config["d_model"]
    llm_batch_size = model_config["batch_size"]
    llm_dtype = model_config["dtype"]
    hook_layer = model_config["layers"][0]

    selected_saes = []

    # If specified, load the ITO_SAE from the given run ID
    if args.ito_run_id is not None:
        run_dir = os.path.join("runs", args.ito_run_id)
        meta_path = os.path.join(run_dir, "metadata.yaml")
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"No run found at {run_dir}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No metadata.yaml found in {run_dir}")

        with open(meta_path, "r") as f:
            metadata = yaml.safe_load(f)

        atoms_path = os.path.join(run_dir, "atoms.pt")
        if not os.path.exists(atoms_path):
            raise FileNotFoundError("No atoms.pt found for the specified ITO run")

        atoms = torch.load(atoms_path).to(device)
        ito_sae = ITO_SAE(atoms, l0=metadata["l0"], cfg=ITO_SAEConfig(
            model_name=model_name,
            dtype=llm_dtype,
            d_in=d_model,
            d_sae=atoms.size(0),
            hook_layer=metadata["layer"],
            hook_name="blocks.8.hook_resid_pre",  # Assuming same hook name as GPT-2 SAEs
            prepend_bos=True,
            normalize_activations="none",
            dataset_trust_remote_code=True,
            seqpos_slice=(None,),
            device=device,
        ))
        selected_saes.append((f"{model_name}_layer_{metadata['layer']}_ito_sae_{atoms.size(0)}", ito_sae))

    # If requested, load pretrained GPT-2 SAEs
    if args.pretrained_saes:
        _, saes, _ = load_model(
            model_name,
            device=device,
            gpt2_saes=[2, 4, 6],
        )
        # Adjust batch size to avoid OOM for pretrained SAEs if needed
        llm_batch_size = 128
        for sae in saes:
            sae.cfg.model_name = model_name
            sae.cfg.model_from_pretrained_kwargs = {}
            selected_saes.append((f"{model_name}_layer_{hook_layer}_sae_{sae.W_dec.size(0)}", sae))

    if not selected_saes:
        raise ValueError("No SAEs selected. Provide either an ITO run ID or use --pretrained_saes.")

    run_evals(
        model_name,
        selected_saes,
        llm_batch_size,
        llm_dtype,
        device,
        eval_types=eval_types,
        api_key=api_key,
        force_rerun=args.force_rerun,
    )
