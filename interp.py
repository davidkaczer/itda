"""
Methods for running interpretation studies on the SAEs, both with human trials
and with automated methods.
"""

import argparse
import os
import random
import re

import numpy as np
import openai
import torch
from dotenv import load_dotenv
from ito import GPT2, OMP, OMP_L0, SEQ_LEN, load_model, ITO_SAE
from tqdm import tqdm


def highlight_string(tokenizer, tokens, target_idx, window=16, highlight=True):
    str = ""
    for i, token in enumerate(tokens):
        # Skip tokens that aren't within `window` places of the target token
        if abs(i - target_idx) > window:
            continue

        token_str = tokenizer.decode(token).replace("\n", "")
        if highlight and (i == target_idx):
            str += f"<<<{token_str}>>>"
        else:
            str += f"{token_str}"
    return str


def get_strings_activations(acts, tokens, n=20):
    non_zero_indices = np.argwhere(acts != 0)
    zero_indices = np.argwhere(acts == 0)

    num_non_zero = int(n * 0.8)
    num_zero = n - num_non_zero

    try:
        non_zero_idxs = non_zero_indices[
            np.random.choice(
                non_zero_indices.shape[0], size=num_non_zero, replace=False
            )
        ]
        zero_idxs = zero_indices[
            np.random.choice(zero_indices.shape[0], size=num_zero, replace=False)
        ]

        idxs = np.concatenate((non_zero_idxs, zero_idxs))
        np.random.shuffle(idxs)
    except ValueError as e:
        raise ValueError(
            f"Not enough activations found to sample {n} values. Found {non_zero_indices.shape[0]} activations.",
            e,
        )

    sample_strs, sample_acts = [], []
    for inp, tok in idxs[:-1]:
        act = acts[inp, tok]
        str = highlight_string(tokens[inp], tok)
        sample_strs.append(str)
        sample_acts.append(act)

    sample_strs, sample_acts = zip(
        *sorted(zip(sample_strs, sample_acts), key=lambda x: x[1], reverse=True)
    )

    return sample_strs, sample_acts


def construct_comparison_question(strings, activations, answer_idxs):
    """
    Given a a list of token-tagged inputs and their activations, constructs a
    comparison question. I.e. given two strings, which has the higher
    activation?
    """
    prompt = ""
    for i, (a, s) in enumerate(zip(activations, strings)):
        if i in answer_idxs:
            continue
        prompt += f"{a:.2f} {s}\n"
    prompt += "\n\n"
    prompt += f"1) {strings[answer_idxs[0]]}\n"
    prompt += f"2) {strings[answer_idxs[1]]}\n"

    answer = 1 if activations[answer_idxs[0]] > activations[answer_idxs[1]] else 2

    return prompt, answer


def predict_comparison_answers(prompts, answers):
    """
    Runs auto-interp on comparison questions and returns the accuracy.
    """
    predictions = []
    for prompt in tqdm(prompts, "Running auto-interp on comparison questions..."):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are doing autointerp on SAE features. First you will be provided with a list of activations and strings. Then you will be asked to compare two strings based on their activations. Answer only with the number of the string you think has the higher activation.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        predictions.append(int(answer))
    correct = sum([1 for p, a in zip(predictions, answers) if p == a]) / len(answers)
    return correct


def get_nonzero_activations(sparse_activations, count=20, get_zero=False):
    if get_zero:
        # Get sequences where the feature is not active
        idxs = np.nonzero((sparse_activations > 0).sum(axis=1) == 0)
        idxs = np.column_stack(
            [idxs, np.random.choice(sparse_activations.shape[1], idxs.shape[0])]
        )
    else:
        idxs = np.nonzero(sparse_activations > 0)

    try:
        sample = np.random.choice(idxs.shape[0], count, replace=False)
    except:
        raise ValueError(
            f"Not enough {'zero' if get_zero else 'non-zero'} activations found to sample {count} values. Found {idxs.shape[0]} activations."
        )
    idxs = idxs[sample]
    acts = sparse_activations[idxs[:, 0], idxs[:, 1]]
    if get_zero:
        acts = np.zeros_like(acts)
    return idxs, acts


def generate_latent_description(
    sparse_activations,
    tokenizer,
    token_dataset,
    latent,
    active_example_count=20,
    inactive_example_count=10,
):
    nonzero_idxs, nonzero_activations = get_nonzero_activations(
        sparse_activations[:, :, latent], active_example_count
    )
    zero_idxs, zero_activations = get_nonzero_activations(
        sparse_activations[:, :, latent],
        inactive_example_count,
        get_zero=True,
    )

    # Sort the indices by activations
    all_indices = np.concatenate((nonzero_idxs, zero_idxs), axis=0)
    all_activations = np.concatenate((nonzero_activations, zero_activations))
    all_indices, all_activations = zip(
        *sorted(zip(all_indices, all_activations), key=lambda x: x[1], reverse=True)
    )
    prompt = ""
    for (seq_idx, token_idx), act in zip(all_indices, all_activations):
        s = highlight_string(tokenizer, token_dataset[seq_idx], token_idx)
        prompt += f"{float(act):.2f} {s}\n"

    with open("prompts/latent_description_system_prompt.txt", "r") as f:
        system_prompt = f.read()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"Description:\s*(.*)", text)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            f"No description found in response.\n\nResponse: {text}\n\nPrompt: {prompt}"
        )


def detection_eval(sae, model_activations, tokenizer, token_dataset, n_latents=100):
    explanation_active_example_count, explanation_inactive_example_count = 20, 10
    # Do this in two steps because the activations are too large to fit in
    # memory
    # 1) Filter out the latents that are active in too few or too many
    # examples, and randomly sample
    # 2) Get the activations for the sampled latents. We want latents that
    # have more than 20 activations and more than 10 samples where they are
    # inactive.

    sae_latent_frequencies = torch.zeros(sae.W_dec.size(0), device=sae.W_dec.device)
    sae_sample_frequencies = []
    for batch in tqdm(
        torch.split(model_activations, 32),
        desc="Getting SAE activations...",
    ):
        acts = sae.encode(batch.to(sae.W_dec.device))
        sae_latent_frequencies += (acts > 0).sum(dim=(0, 1))
        sae_sample_frequencies.append((acts == 0).all(dim=1))
    sae_sample_frequencies = torch.cat(sae_sample_frequencies, dim=0).sum(dim=0)

    filtered_latents = (
        sae_latent_frequencies > explanation_active_example_count
    ) & (sae_sample_frequencies > explanation_inactive_example_count)
    sampled_latents = np.random.choice(
        np.nonzero(filtered_latents.cpu().numpy())[0], n_latents, replace=False
    )

    activations = []
    for batch in tqdm(
        torch.split(model_activations, 32), desc="Getting SAE activations..."
    ):
        acts = sae.encode(batch.to(device))[:, :, sampled_latents]
        activations.append(acts.cpu())
    sparse_activations = torch.cat(activations, dim=0)

    correct = 0
    failed = 0
    for latent in tqdm(range(n_latents), desc="Running detection eval..."):
        try:
            latent_description = generate_latent_description(
                sparse_activations,
                tokenizer,
                token_dataset,
                latent,
                active_example_count=explanation_active_example_count,
                inactive_example_count=explanation_inactive_example_count,
            )
        except ValueError as e:
            print(f"Failed to generate description for latent {latent}.", e)
            failed += 1
            continue

        nonzero_count = 5
        nonzero_idxs, nonzero_acts = get_nonzero_activations(
            sparse_activations[:, :, latent], count=nonzero_count
        )

        zero_count = 20
        zero_idxs, _ = get_nonzero_activations(
            sparse_activations[:, :, latent], count=zero_count, get_zero=True
        )

        answer = random.randint(0, 4)
        try:
            options_sample = np.random.choice(zero_idxs.shape[0], size=5)
            options = zero_idxs[options_sample]
        except TypeError as e:
            print("Failed to sample zero activations:", e, zero_idxs)
            failed += 1
            continue
        options[answer] = random.choice(nonzero_idxs)

        prompt = f"""Description: {latent_description}

Which of the following inputs activates on this feature?
"""
        for i, idx in enumerate(options):
            s = highlight_string(
                tokenizer, token_dataset[idx[0]], idx[1], highlight=False
            )
            prompt += f"{i}) {s}\n"

        with open("prompts/detection_eval_system.txt", "r") as f:
            system_prompt = f.read()

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        response = response.choices[0].message.content.strip()
        try:
            response = int(response)
            if int(response) == answer:
                correct += 1
        except ValueError:
            print("Invalid response:", response)
            failed += 1

    return correct / (n_latents - failed)


if __name__ == "__main__":
    load_dotenv(override=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=GPT2)
    parser.add_argument("--l0", type=int, default=OMP_L0)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--ito_fn", type=str, default=OMP)
    parser.parse_args()
    args = parser.parse_args()

    # TODO: Just reconstruct the activations each time rather than bothering
    # saving them?
    model_activations = torch.load(f"data/{args.model}/model_activations.pt")

    model, saes, token_dataset = load_model(args.model, device=device)
    sae = saes[0]

    atoms = torch.load(f"data/{args.model}/atoms.pt")
    ito_sae = ITO_SAE(atoms, device=device)

    # assume the test set is at the end of the dataset
    test_size = int(model_activations.shape[0] * 0.3)
    train_size = len(token_dataset["tokens"]) - test_size
    token_dataset = token_dataset["tokens"][train_size:, : args.seq_len]

    feature_count = 100
    sample_string_count = 20

    print("Running ITO detection eval...")
    ito_detection_score = detection_eval(
        ito_sae,
        model_activations[train_size:, : args.seq_len],
        model.tokenizer,
        token_dataset,
        n_latents=feature_count,
    )

    print("Running SAE detection eval...")
    sae_detection_score = detection_eval(
        sae,
        model_activations[train_size:, : args.seq_len],
        model.tokenizer,
        token_dataset,
        n_latents=feature_count,
    )
    print(f"ITO detection score: {ito_detection_score}")
    print(f"SAE detection score: {sae_detection_score}")
