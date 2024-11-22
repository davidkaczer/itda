import argparse
import time
from copy import copy
import json

from dotenv import load_dotenv
from evals.core.main import (
    CoreEvalConfig,
)
from evals.core.main import run_evals as run_core
from evals.sparse_probing.main import SparseProbingEvalConfig
from evals.sparse_probing.main import run_eval_single_sae as run_sparse_probing
from sae_lens.training.activations_store import ActivationsStore
import torch
from ito import get_model_name

from ito import GPT2, ITO_SAE, OMP, OMP_L0, SEQ_LEN, load_model

# TODO: Commonise this code with interp.py
if __name__ == "__main__":
    load_dotenv(override=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=GPT2)
    parser.add_argument("--layer", type=str, default=8)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--ito_fn", type=str, default=OMP)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.parse_args()
    args = parser.parse_args()

    model_name = get_model_name(args.model, args.layer, args.l0)

    out_dir = f"data/{model_name}"
    if args.timestamp is None:
        timestamp = int(time.time())

    model_activations = torch.load(f"data/{model_name}/model_activations.pt")

    model, saes, token_dataset = load_model(
        # args.model, device=device, gpt2_saes=range(8, 9)
        args.model,
        device=device,
        gpt2_saes=range(1, 2),
    )
    sae = saes[0]

    atoms = torch.load(f"data/{model_name}/atoms.pt")
    ito_cfg = copy(sae.cfg)
    ito_cfg.device = "cpu"
    ito_sae = ITO_SAE(atoms, l0=40, cfg=ito_cfg)

    print(
        f"Evaluating an ITO SAE with {ito_sae.W_dec.size(0)} atoms against an SAE with {sae.W_dec.size(0)} atoms"
    )

    context_size = 4

    print("---- CORE -----")
    activation_store = ActivationsStore.from_sae(
        model, ito_sae, context_size=context_size
    )
    eval_config = CoreEvalConfig(
        model_name=args.model,
        llm_dtype="float32",
        compute_kl=True,
        compute_ce_loss=True,
        batch_size_prompts=16,
        context_size=128,
    )
    ito_sae_core = run_core(ito_sae, activation_store, model, eval_config=eval_config)
    activation_store = ActivationsStore.from_sae(model, sae, context_size=context_size)
    sae_core = run_core(sae, activation_store, model, eval_config=eval_config)

    print("---- SPARSE PROBING -----")
    eval_config = SparseProbingEvalConfig(
        sae_batch_size=512,
        probe_train_set_size=10,
        probe_test_set_size=10,
        model_name=args.model,
        llm_dtype="float32",
        dataset_names=[
            "LabHC/bias_in_bios_class_set1",
        ],
        k_values=[1],
    )
    activation_store = ActivationsStore.from_sae(model, sae, context_size=context_size)
    ito_sae_sparse_probe = run_sparse_probing(
        eval_config, ito_sae, model, activation_store, "cuda", "ito_sae_prob"
    )
    activation_store = ActivationsStore.from_sae(model, sae, context_size=context_size)
    sae_sparse_probe = run_sparse_probing(
        eval_config, sae, model, activation_store, "cuda", "sae_probe"
    )

    # save results to model folder
    results = {
        "core": {"ito_sae": ito_sae_core, "sae": sae_core},
        "sparse_probe": {"ito_sae": ito_sae_sparse_probe, "sae": sae_sparse_probe},
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f)
