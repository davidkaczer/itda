# %%
import argparse
import os
from queue import Empty
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from datasets import load_dataset
from dl_train import ITDA, activation_dims
from layer_similarity import get_layered_runs_for_models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from sae_lens import SAE

import wandb

LM_NAME = "EleutherAI/pythia-70m-deduped"
BATCH_SIZE = 32
PROBING_DATASET = "LabHC/bias_in_bios"
LAYERS = list(range(1, 6))
SEQ_LEN = 128
TOTAL_SAMPLES = 10000

# %%

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 1) Load the ITDA runs for each layer
    runs = get_layered_runs_for_models(
        [LM_NAME], LAYERS, entity="patrickaaleask", project="itda", tag="probing"
    )[LM_NAME]
    itdas = {
        layer: ITDA.from_pretrained(f"artifacts/runs/{run.id}")
        for layer, run in runs.items()
    }

    # 2) Load dataset (streaming)
    dataset = load_dataset(PROBING_DATASET, split="train", streaming=True)
    data_stream = ((item["hard_text"], item["profession"]) for item in dataset)

    if LM_NAME not in activation_dims:
        raise ValueError(
            f"Unknown activation_dim for model {LM_NAME}. Please update `activation_dims` dict."
        )
    activation_dim = activation_dims[LM_NAME]

    # 3) Load SAE for each layer
    saes = {}
    for layer_idx in LAYERS:
        saes[layer_idx] = SAE.from_pretrained(
            release="pythia-70m-deduped-res-sm",
            sae_id=f"blocks.{layer_idx}.hook_resid_post",
            device=device
        )[0]

    # 4) Load the TransformerLens model and tokenizer
    model = HookedTransformer.from_pretrained(LM_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(LM_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5) Collect activations for ITDA, SAE, and pure
    activations_itda = {layer: [] for layer in LAYERS}
    activations_sae = {layer: [] for layer in LAYERS}
    activations_pure = {layer: [] for layer in LAYERS}

    targets = []
    samples_seen = 0

    progress = tqdm(
        data_stream,
        total=min([dataset.info.splits["train"].num_examples, TOTAL_SAMPLES]),
        desc="Collecting activations"
    )

    with torch.no_grad():
        while samples_seen < TOTAL_SAMPLES:
            batch_texts = []
            for _ in range(BATCH_SIZE):
                try:
                    text, label = next(data_stream)
                except StopIteration:
                    print("Data stream exhausted.")
                    break
                batch_texts.append(text)
                targets.append(label)

            if not batch_texts:
                break

            # Tokenize
            tokens = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=SEQ_LEN,
                return_tensors="pt",
            ).to(device)

            # Forward pass with cache
            _, cache = model.run_with_cache(
                tokens["input_ids"],
                stop_at_layer=max(LAYERS) + 1,
                names_filter=[
                    f"blocks.{layer_idx}.hook_resid_post" for layer_idx in LAYERS
                ],
            )

            # Store activations
            for layer_idx in LAYERS:
                resid = cache[f"blocks.{layer_idx}.hook_resid_post"]

                # ITDA
                acts_itda = itdas[layer_idx].encode(resid).sum(dim=1).cpu()
                activations_itda[layer_idx].append(acts_itda)

                # SAE
                acts_sae = saes[layer_idx].encode(resid).sum(dim=1).cpu()
                activations_sae[layer_idx].append(acts_sae)

                # Pure
                acts_pure = resid.sum(dim=1).cpu()
                activations_pure[layer_idx].append(acts_pure)

            progress.update(len(batch_texts))
            samples_seen += len(batch_texts)

    # Concatenate list of mini-batches into final activation Tensors
    activations_itda = {
        layer: torch.cat(acts) for layer, acts in activations_itda.items()
    }
    activations_sae = {
        layer: torch.cat(acts) for layer, acts in activations_sae.items()
    }
    activations_pure = {
        layer: torch.cat(acts) for layer, acts in activations_pure.items()
    }

    # normalise the activations
    # for layer in LAYERS:
    #     activations_itda[layer] /= activations_itda[layer].norm(dim=1, keepdim=True)
    #     activations_sae[layer] /= activations_sae[layer].norm(dim=1, keepdim=True)
    #     activations_pure[layer] /= activations_pure[layer].norm(dim=1, keepdim=True)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l1_regularized_loss(model, criterion, X, y, alpha=1.0):
    """
    Compute cross-entropy loss + alpha * L1 penalty on model parameters.
    """
    logits = model(X)
    loss = criterion(logits, y)
    l1_penalty = 0.0
    for name, param in model.named_parameters():
        if "weight" in name:  # Skip biases in L1
            l1_penalty += torch.sum(torch.abs(param))
    loss = loss + alpha * l1_penalty
    return loss, logits


def train_sparse_probe(X_layer, y, alpha=1., max_iter=1000, lr=1e-5):
    """
    Train a simple linear classifier (logistic regression in PyTorch) with L1 regularization.
    Returns the final accuracy on a hold-out set.
    """
    # Convert to CPU NumPy for split
    X_cpu = X_layer.cpu().detach().float().numpy()
    y_cpu = y  # already a NumPy array after label encoding

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cpu, y_cpu, test_size=0.2, random_state=42
    )

    # Convert back to torch Tensors on GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    input_dim = X_train_t.shape[1]
    num_classes = len(set(y_cpu))  # for multi-class

    # Define linear model
    model = nn.Linear(input_dim, num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_iter):
        model.train()
        optimizer.zero_grad()
        loss, _ = l1_regularized_loss(model, criterion, X_train_t, y_train_t, alpha=alpha)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_test_t).float().mean().item()

    return accuracy, model

if __name__ == "__main__":
    # Encode the text labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(targets)

    # Print header for the results table
    print(f"{'Layer':<6} | {'ITDA':<10} | {'SAE':<10} | {'Model':<10}")
    print("-"*45)

    for layer_idx in LAYERS:
        acc_itda = train_sparse_probe(activations_itda[layer_idx], y)[0]
        acc_sae = train_sparse_probe(activations_sae[layer_idx], y)[0]
        acc_pure = train_sparse_probe(activations_pure[layer_idx], y)[0]

        print(
            f"{layer_idx:<6} | "
            f"{acc_itda:<10.4f} | "
            f"{acc_sae:<10.4f} | "
            f"{acc_pure:<10.4f}"
        )


# %%


X_itda_concat = torch.cat([activations_itda[layer] for layer in LAYERS], dim=1)
X_itda_concat[X_itda_concat < 0.] = 0.
X_sae_concat = torch.cat([activations_sae[layer] for layer in LAYERS], dim=1)
X_pure_concat = torch.cat([activations_pure[layer] for layer in LAYERS], dim=1)

acc_itda_all, itda_probe = train_sparse_probe(X_itda_concat, y, alpha=0.1)
acc_sae_all, sae_probe = train_sparse_probe(X_sae_concat, y, alpha=0.1)
acc_pure_all, pure_probe = train_sparse_probe(X_pure_concat, y, alpha=0.1)

print("Accuracy on concatenated activations (all layers together):")
print(f"ITDA : {acc_itda_all:.4f}")
print(f"SAE  : {acc_sae_all:.4f}")
print(f"Pure : {acc_pure_all:.4f}")
# %%
