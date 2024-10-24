# Save this code as app.py and run it using `streamlit run app.py`

import html
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
from main import OMP_L0, SEQ_LEN, TRAIN_SIZE, GPT2, GEMMA2
from transformers import GPT2Tokenizer, AutoTokenizer
from transformer_lens import HookedTransformer

from meta_saes.sae import load_feature_splitting_saes, load_gemma_sae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(layout="wide")

MODEL_NAME = GEMMA2
SEQ_LEN = 128

def match_activation(search_activation, all_activations):
    if all_activations.shape[-1] != search_activation.shape[0]:
        raise ValueError(
            "Target tensor dimensions do not match the last dimension of the larger tensor."
        )

    reshaped_tensor = all_activations.view(-1, all_activations.shape[-1])
    matches = torch.all(reshaped_tensor == search_activation, dim=1)
    matching_indices_flat = torch.nonzero(matches, as_tuple=False).squeeze()

    if matching_indices_flat.numel() > 0:
        try:
            first_match_index = matching_indices_flat[0]
        except IndexError:
            first_match_index = matching_indices_flat
        batch_index = torch.div(
            first_match_index, all_activations.shape[1], rounding_mode="floor"
        )
        sequence_index = first_match_index % all_activations.shape[1]
        return batch_index.item(), sequence_index.item()
    else:
        raise ValueError("No matching vector found in the larger tensor.")


def get_activation_color(value):
    min_value = -1
    max_value = 1
    normalized_value = (
        (value - min_value) / (max_value - min_value) if max_value > min_value else 0
    )
    red = int(255 * (1 - normalized_value))
    green = int(255 * normalized_value)
    return f"rgb({red},{green},0)"


def highlight_string(tokens, idx, tokenizer):
    str = ""
    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token]).replace("\n", "")
        if i == idx:
            str += f"<<{token_str}>>"
        else:
            str += f"{token_str}"
    return str


def get_strings_activations(acts, tokens, n=20):
    non_zero_indices = np.argwhere(acts != 0)
    zero_indices = np.argwhere(acts == 0)

    num_non_zero = int(n * 0.8)
    num_zero = n - num_non_zero

    if non_zero_indices.shape[0] == 0:
        raise ValueError("No non-zero activations found.")

    if zero_indices.shape[0] == 0:
        raise ValueError("No zero activations found.")

    non_zero_idxs = non_zero_indices[
        np.random.choice(
            non_zero_indices.shape[0],
            size=min(num_non_zero, non_zero_indices.shape[0]),
            replace=False,
        )
    ]
    zero_idxs = zero_indices[
        np.random.choice(
            zero_indices.shape[0],
            size=min(num_zero, zero_indices.shape[0]),
            replace=False,
        )
    ]

    idxs = np.concatenate((non_zero_idxs, zero_idxs))
    np.random.shuffle(idxs)

    sample_strs, sample_acts = [], []
    for inp, tok in idxs:
        act = acts[inp, tok]

        str = highlight_string(tokens[TRAIN_SIZE + inp], tok, tokenizer)

        sample_strs.append(str)
        sample_acts.append(act)

    sample_strs, sample_acts = zip(
        *sorted(zip(sample_strs, sample_acts), key=lambda x: x[1], reverse=True)
    )

    return sample_strs, sample_acts, [i[0] for i in idxs], [i[1] for i in idxs]


def generate_test_samples(activations, feature_sample_count, tokens):
    tests = []
    for i in range(feature_sample_count):
        try:
            sample_strs, sample_acts, sample_idxs, token_idxs = get_strings_activations(
                activations[:, :, i], tokens, n=20
            )
            tests.append((sample_strs, sample_acts, sample_idxs, token_idxs))
        except ValueError as e:
            continue
    return tests


# Load data
@st.cache_resource
def load_data():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_NAME == GPT2:
        model, saes, token_dataset = load_feature_splitting_saes(
            device=device,
            saes_idxs=list(range(1, 9)),
        )
    elif MODEL_NAME == GEMMA2:
        model, saes, token_dataset = load_gemma_sae(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_12/width_16k/average_l0_41",
            device=device,
            dataset="NeelNanda/pile-10k",
        )
    else:
        raise ValueError("Invalid model")

    atoms = torch.load(f"data/{MODEL_NAME}/atoms.pt")
    omp_activations = torch.load(f"data/{MODEL_NAME}/omp_activations.pt").reshape(
        -1, SEQ_LEN, OMP_L0
    )
    omp_indices = torch.load(f"data/{MODEL_NAME}/omp_indices.pt").reshape(-1, SEQ_LEN, OMP_L0)
    model_activations = torch.load(f"data/{MODEL_NAME}/model_activations.pt")
    train_activations = model_activations[:TRAIN_SIZE]
    normed_activations = train_activations / train_activations.norm(dim=2).unsqueeze(2)
    tokens = torch.stack([s["tokens"] for s in token_dataset])[:, :SEQ_LEN].to(device)
    return model, saes, atoms, omp_activations, omp_indices, normed_activations, tokens


model, saes, atoms, omp_activations, omp_indices, normed_activations, tokens = (
    load_data()
)
tokenizer = model.tokenizer

TRAIN_SIZE = 10000


# Read query parameters
params = st.query_params
page = params.get("page", "activation_interface")

if page == "activation_interface":
    # Activation Interface Page
    st.title("Activation Interface")

    # Retrieve query parameters
    query_params = st.query_params
    input_idx = int(query_params.get("input_idx", 0))
    token_idx = int(query_params.get("token_idx", 0))

    # Provide UI controls
    input_idx = st.number_input(
        "Input index",
        min_value=0,
        max_value=omp_indices.shape[0] - 1,
        value=input_idx,
        step=1,
    )
    token_idx = st.number_input(
        "Token index",
        min_value=0,
        max_value=omp_indices.shape[1] - 1,
        value=token_idx,
        step=1,
    )

    # Update the query parameters based on user inputs
    st.query_params.input_idx = input_idx
    st.query_params.token_idx = token_idx

    # Get indices and activations
    indices = omp_indices[input_idx][token_idx]
    activations = omp_activations[input_idx][token_idx]

    sorted_activations_indices = sorted(
        zip(activations, indices), key=lambda x: x[0], reverse=True
    )

    title = highlight_string(tokens[TRAIN_SIZE + input_idx], token_idx, tokenizer)
    st.subheader(title)
    rows = []

    for a, i in sorted_activations_indices:
        try:
            atom_input_idx, atom_token_idx = match_activation(
                atoms[i], normed_activations
            )
            highlighted_string = highlight_string(
                tokens[atom_input_idx], atom_token_idx, tokenizer
            )
            color = get_activation_color(a)
            activation_text = f"{a:.2f}"
            rows.append(
                {
                    "Atom": i,
                    "Activation": activation_text,
                    "String": highlighted_string,
                    "Color": color,
                }
            )
        except ValueError as e:
            st.write(f"Error: {e}")
            continue

    header_col1, header_col2, header_col3 = st.columns([1, 1, 10])

    with header_col1:
        st.markdown("**Atom**")
    with header_col2:
        st.markdown("**Activation**")
    with header_col3:
        st.markdown("**String**")

    # Create rows for each entry in the data
    for row in rows:
        atom_index = row["Atom"]
        activation = row["Activation"]
        string = row["String"]
        color = row["Color"]

        # Create a row of columns for each data field
        col1, col2, col3 = st.columns([1, 1, 10])

        # Render the values in their respective columns
        with col1:
            link = (
                f'<a href="?page=test_samples&atom_index={atom_index}" target="_self">{atom_index}</a>'
            )
            st.markdown(link, unsafe_allow_html=True)
        with col2:
            st.markdown(
                f"<div style='background-color:{color};padding:5px'>{activation}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.text(string)

elif page == "test_samples":
    # Test Samples Page
    atom_index = int(params.get("atom_index", [0]))
    st.title(f"Test Samples for Atom {atom_index}")

    atom_index = st.number_input(
        "Atom index",
        min_value=0,
        max_value=omp_indices.max(),
        value=atom_index,
        step=1,
    )

    st.query_params.atom_index = atom_index

    mask = omp_indices == atom_index
    feature_activations = omp_activations * mask
    dense_omp_activations = np.sum(feature_activations, axis=-1)
    dense_omp_activations = np.expand_dims(
        dense_omp_activations, axis=-1
    )
    if np.max(dense_omp_activations) != 0:
        dense_omp_activations = dense_omp_activations / dense_omp_activations.max()
    dense_omp_activations[dense_omp_activations < 0.0] = 0.0

    tests = generate_test_samples(
        dense_omp_activations,
        feature_sample_count=1,
        tokens=tokens,
    )

    if tests:
        sample_strs, sample_acts, sample_idxs, token_idxs = tests[0]
        st.subheader("Sample Strings and Activations:")

        data = {
            "Sample Index": zip(sample_idxs, token_idxs),
            "Activation": [f"{act:.2f}" for act in sample_acts],
            "Sample String": [html.escape(s) for s in sample_strs],
        }

        df = pd.DataFrame(data)
        df["Sample Index"] = df["Sample Index"].apply(
            lambda idxs: f'<a href="http://localhost:8501/?input_idx={idxs[0]}&token_idx={idxs[1]}" target="_self">{idxs[0]}:{idxs[1]}</a>'
        )

        def highlight_activations(val):
            color = get_activation_color(float(val))
            return f"background-color: {color}"

        styled_df = df.style.applymap(highlight_activations, subset=["Activation"])
        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.write("No test samples found for this atom.")

    # Provide a link back to Activation Interface
    st.markdown(
        '<a href="?page=activation_interface" target="_self">Back to Activation Interface</a>',
        unsafe_allow_html=True,
    )

else:
    st.error("Invalid page")
