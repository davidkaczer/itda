"""
Methods for running interpretation studies on the SAEs, both with human trials
and with automated methods.
"""

import argparse
import asyncio
import json
import os
import random
import re
import time

import numpy as np
import openai
import torch
from dotenv import load_dotenv
from ito import GPT2, ITO_SAE, OMP, OMP_L0, SEQ_LEN, load_model
from tqdm import tqdm

GPT4o_MINI = "gpt-4o-mini"
GPT4o = "gpt-4o"


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


async def generate_latent_description(
    client,
    run_id: str,
    sparse_activations,
    tokenizer,
    token_dataset,
    active_example_count=20,
    inactive_example_count=10,
    interp_model=GPT4o_MINI,
    out_dir="data",
):
    print("Generating latent descriptions...")
    requests = []
    for latent in range(sparse_activations.shape[-1]):
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

        with open("prompts/latent_description_system.txt", "r") as f:
            system_prompt = f.read()

        request = {
            "custom_id": f"{run_id}-{latent}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": interp_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
        }
        requests.append(request)

    responses = await batch_request(client, requests, out_dir, run_id)

    descriptions = []
    for response in responses:
        match = re.search(r"Description:\s*(.*)", response)
        if match:
            descriptions.append(match.group(1))
        else:
            print('No description found in response:', response)
            descriptions.append(None)
    return descriptions


# TODO: Handle failed batches
async def batch_request(client, requests, out_dir, run_id):
    """
    Creates an openai batch request, and waits for the response.
    """
    response_filename = f"{out_dir}/{run_id}-response.json"
    
    # Check if response file already exists
    if os.path.exists(response_filename):
        print(f"Response file {response_filename} already exists. Loading from file...")
        results = []
        with open(response_filename, "r") as file:
            for line in file:
                result = json.loads(line.strip())
                try:
                    text = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                    results.append(text)
                except KeyError as e:
                    print("Failed to get description:", e)
                    print(result)
                    results.append(None)
                    continue
        return results

    if len(requests) <= 10:
        # assume that we're testing stuff and don't want to wait for a batch
        results = []
        for request in tqdm(
            requests, desc="Running requests synchronously as small number"
        ):
            result = client.chat.completions.create(
                model=request["body"]["model"], messages=request["body"]["messages"]
            )
            try:
                response = result.choices[0].message.content.strip()
                results.append(response)
            except KeyError as e:
                print("Failed to get description:", e)
                print(result)
                results.append(None)
        return results

    request_filename = f"{out_dir}/{run_id}-request.json"
    with open(request_filename, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    batch_file = client.files.create(
        file=open(request_filename, "rb"),
        purpose="batch",
    )
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print("Batch job created. Waiting for completion...")
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        if batch_job.status == "completed":
            print(f"Batch job {run_id} completed.")
            break
        await asyncio.sleep(5)

    response = client.files.content(batch_job.output_file_id).content
    with open(response_filename, "wb") as f:
        f.write(response)

    print(f"Response written to {response_filename}")

    results = []
    with open(response_filename, "r") as file:
        for line in file:
            result = json.loads(line.strip())
            try:
                text = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                results.append(text)
            except KeyError as e:
                print("Failed to get description:", e)
                print(result)
                results.append(None)
                continue

    return results


async def fuzzing_eval(
    client,
    run_id: str,
    sae,
    model_activations,
    tokenizer,
    token_dataset,
    n_latents=100,
    interp_model=GPT4o_MINI,
    out_dir="data",
):
    """
    In the fuzzing eval, the auto-interp model is given a description of a
    feature and five pairs of activations and inputs, where the token is
    highlighted. One of these activations is incorrect, and the model is tasked
    with identifying which activation is incorrect.
    """
    explanation_active_example_count, explanation_inactive_example_count = 20, 10
    sampled_latents = get_candidate_latents(
        sae,
        model_activations,
        explanation_active_example_count=explanation_active_example_count,
        explanation_inactive_example_count=explanation_inactive_example_count,
        n_latents=n_latents,
    )

    sparse_activations = get_latent_activations(sae, model_activations, sampled_latents)

    latent_descriptions = await generate_latent_description(
        client,
        f"{run_id}-latent-description",
        sparse_activations,
        tokenizer,
        token_dataset,
        active_example_count=explanation_active_example_count,
        inactive_example_count=explanation_inactive_example_count,
        interp_model=interp_model,
        out_dir=out_dir,
    )

    requests = []
    answers = []
    for latent in tqdm(range(n_latents), desc="Running fuzzing eval..."):
        latent_description = latent_descriptions[latent]
        if latent_description is None:
            continue

        # always include at least one non-zero value
        nonzero_count = random.randint(1, 4)
        nonzero_idxs, nonzero_acts = get_nonzero_activations(
            sparse_activations[:, :, latent], count=nonzero_count
        )
        zero_count = 5 - nonzero_count
        zero_idxs, zero_acts = get_nonzero_activations(
            sparse_activations[:, :, latent], count=zero_count, get_zero=True
        )

        idxs = np.concatenate((nonzero_idxs, zero_idxs))
        acts = np.concatenate((nonzero_acts, zero_acts))
        idxs, acts = zip(*sorted(zip(idxs, acts), key=lambda x: x[1], reverse=True))
        idxs, acts = np.array(idxs), np.array(acts)

        answer = random.randint(0, 4)
        answers.append(answer)
        unique_activations = sparse_activations[:, :, latent].unique()
        unique_activations = unique_activations[unique_activations >= 0]
        # select a random unique activation that is different from the original
        # activation
        new_activation = random.choice(
            unique_activations[unique_activations != acts[answer]]
        )
        acts[answer] = new_activation

        prompt = f"""Description: {latent_description}

Which of the following activations is incorrect?
"""
        for i, (idx, act) in enumerate(zip(idxs, acts)):
            s = highlight_string(tokenizer, token_dataset[idx[0]], idx[1])
            prompt += f"{i}) {act:.2f} {s}\n"

        with open("prompts/fuzzing_eval_system.txt", "r") as f:
            system_prompt = f.read()

        request = {
            "custom_id": f"{run_id}-{latent}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": interp_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
        }
        requests.append(request)

    results = await batch_request(client, requests, out_dir, run_id)
    correct, failed = 0, 0
    for result, answer in zip(results, answers):
        try:
            result = int(result)
            if int(result) == answer:
                correct += 1
        except ValueError:
            print("Invalid result:", result)
            failed += 1

    return correct / (len(results) - failed)


def get_candidate_latents(
    sae,
    model_activations,
    explanation_active_example_count=20,
    explanation_inactive_example_count=10,
    n_latents=100,
):
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
        desc="Finding candidate latents...",
    ):
        acts = sae.encode(batch.to(sae.W_dec.device))
        sae_latent_frequencies += (acts > 0).sum(dim=(0, 1))
        sae_sample_frequencies.append((acts == 0).all(dim=1))
    sae_sample_frequencies = torch.cat(sae_sample_frequencies, dim=0).sum(dim=0)

    filtered_latents = (sae_latent_frequencies > explanation_active_example_count) & (
        sae_sample_frequencies > explanation_inactive_example_count
    )
    sampled_latents = np.random.choice(
        np.nonzero(filtered_latents.cpu().numpy())[0], n_latents, replace=False
    )
    return sampled_latents


def get_latent_activations(sae, model_activations, latents):
    activations = []
    for batch in tqdm(
        torch.split(model_activations, 32), desc="Getting latent activations..."
    ):
        acts = sae.encode(batch.to(device))[:, :, latents]
        activations.append(acts.cpu())
    sparse_activations = torch.cat(activations, dim=0)
    return sparse_activations


async def detection_eval(
    client,
    run_id: str,
    sae,
    model_activations,
    tokenizer,
    token_dataset,
    n_latents=100,
    interp_model=GPT4o_MINI,
    out_dir="data",
):
    """
    In the detection eval, the auto-interp model is given the description of a
    latent and five inputs, and asked which of those five inputs activates.

    We always use 4 non-activating inputs and 1 activating input. Eleuther use a
    random mix, but it's unclear why this would be better, perhaps just a bit
    harder.
    """
    explanation_active_example_count, explanation_inactive_example_count = 20, 10
    sampled_latents = get_candidate_latents(
        sae,
        model_activations,
        explanation_active_example_count=explanation_active_example_count,
        explanation_inactive_example_count=explanation_inactive_example_count,
        n_latents=n_latents,
    )

    sparse_activations = get_latent_activations(sae, model_activations, sampled_latents)

    latent_descriptions = await generate_latent_description(
        client,
        f"{run_id}-latent-description",
        sparse_activations,
        tokenizer,
        token_dataset,
        active_example_count=explanation_active_example_count,
        inactive_example_count=explanation_inactive_example_count,
        interp_model=interp_model,
        out_dir=out_dir,
    )

    requests = []
    answers = []
    for latent in tqdm(range(n_latents), desc="Running detection eval..."):
        latent_description = latent_descriptions[latent]
        if latent_description is None:
            print("Skipping due to missing description")
            continue

        nonzero_count = 5
        nonzero_idxs, _ = get_nonzero_activations(
            sparse_activations[:, :, latent], count=nonzero_count
        )

        zero_count = 20
        # TODO: I'm not sure why sometimes happens, seems quite rare. Seems like
        # an issue with the code that checks that there are sufficient zero
        # activations. Might be better to switch to EAFP and just catch the
        # exception.
        try:
            zero_idxs, _ = get_nonzero_activations(
                sparse_activations[:, :, latent], count=zero_count, get_zero=True
            )
        except ValueError as e:
            print("Skipping due to error:", e)

        answer = random.randint(0, 4)
        answers.append(answer)
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

        request = {
            "custom_id": f"{run_id}-{latent}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": interp_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
        }
        requests.append(request)

    results = await batch_request(client, requests, out_dir, run_id)
    correct, failed = 0, 0
    for result, answer in zip(results, answers):
        try:
            result = int(result)
            if int(result) == answer:
                correct += 1
        except ValueError:
            print("Invalid result:", result)
            failed += 1

    return correct / (len(results) - failed)


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
    parser.add_argument("--interp_model", type=str, default=GPT4o_MINI)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.parse_args()
    args = parser.parse_args()

    out_dir = f"data/{args.model}"
    if args.timestamp is None:
        timestamp = int(time.time())

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

    openai_client = openai.OpenAI()

    async def run_parallel_evals():
        print("Running all evaluations in parallel...")
        ito_detection_task = detection_eval(
            openai_client,
            f"{timestamp}-detection-ito",
            ito_sae,
            model_activations[train_size:, : args.seq_len],
            model.tokenizer,
            token_dataset,
            n_latents=feature_count,
            interp_model=args.interp_model,
            out_dir=out_dir,
        )
        sae_detection_task = detection_eval(
            openai_client,
            f"{timestamp}-detection-sae",
            sae,
            model_activations[train_size:, : args.seq_len],
            model.tokenizer,
            token_dataset,
            n_latents=feature_count,
            interp_model=args.interp_model,
            out_dir=out_dir,
        )
        ito_fuzzing_task = fuzzing_eval(
            openai_client,
            f"{timestamp}-fuzzing-ito",
            ito_sae,
            model_activations[train_size:, : args.seq_len],
            model.tokenizer,
            token_dataset,
            n_latents=feature_count,
            interp_model=args.interp_model,
            out_dir=out_dir,
        )
        sae_fuzzing_task = fuzzing_eval(
            openai_client,
            f"{timestamp}-fuzzing-sae",
            sae,
            model_activations[train_size:, : args.seq_len],
            model.tokenizer,
            token_dataset,
            n_latents=feature_count,
            interp_model=args.interp_model,
            out_dir=out_dir,
        )

        ito_detection_score, sae_detection_score, ito_fuzzing_score, sae_fuzzing_score = await asyncio.gather(
            ito_detection_task, sae_detection_task, ito_fuzzing_task, sae_fuzzing_task
        )

        print(f"ITO detection score: {ito_detection_score}")
        print(f"SAE detection score: {sae_detection_score}")
        print(f"ITO fuzzing score: {ito_fuzzing_score}")
        print(f"SAE fuzzing score: {sae_fuzzing_score}")

    # Run all evaluations in parallel
    asyncio.run(run_parallel_evals())