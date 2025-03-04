# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
import os

if __name__ == "__main__":
    # hf_repo_id = "adamkarvonen/sae_bench_results"
    local_dir = "./sae_bench_results"
    # os.makedirs(local_dir, exist_ok=true)

    # snapshot_download(
    #     repo_id=hf_repo_id,
    #     local_dir=local_dir,
    #     repo_type="dataset",
    #     ignore_patterns=[
    #         "*autointerp_with_generations*",
    #         "*core_with_feature_statistics*",
    #     ],
    # )

    print(os.listdir(local_dir))

# %%

    sae_bench_df = pd.DataFrame(sae_bench_data)
    result_df = sae_bench_df[sae_bench_df["modelId"] != "gemma-2-9b"]
    result_df["saeType"] = "SAE"

    # ------------------------------------------------
    # Existing metric: CE loss score
    # ------------------------------------------------
    result_df["ce_loss_score"] = result_df["core||model_performance_preservation||ce_loss_score"]

    # ------------------------------------------------
    # New metrics from JSON
    # ------------------------------------------------
    result_df["autointerp_score"] = result_df["autointerp||autointerp||autointerp_score"]
    result_df["sae_test_accuracy"] = result_df["sparse_probing||llm||llm_test_accuracy"]
    result_df["sae_top_1_test_accuracy"] = result_df["sparse_probing||llm||llm_top_1_test_accuracy"]
    result_df["sae_top_5_test_accuracy"] = result_df["sparse_probing||llm||llm_top_5_test_accuracy"]

    # ------------------------------------------------
    # Other columns used for filtering/plotting
    # ------------------------------------------------
    result_df["l0"] = result_df["core||sparsity||l0"]
    result_df = result_df[
        [
            "modelId",
            "layer",
            "dSae",
            "ce_loss_score",
            "autointerp_score",
            "sae_test_accuracy",
            "sae_top_1_test_accuracy",
            "sae_top_5_test_accuracy",
            "saeType",
            "l0",
            "saeClass",
        ]
    ]

    ################################################################
    # 2) Filter to targets for l0
    ################################################################
    targets = np.array([8, 16, 40, 80])

    mask = np.any(
        (result_df["l0"].values[:, None] >= targets * 0.9)
        & (result_df["l0"].values[:, None] <= targets * 1.1),
        axis=1,
    )
    df_filtered = result_df[mask].copy()

    # Round each l0 to the closest target
    df_filtered["l0"] = df_filtered["l0"].apply(lambda x: targets[np.argmin(np.abs(targets - x))])

    ################################################################
    # 3) Load ITDA data from W&B
    ################################################################
    api = wandb.Api()
    project_name = "itda"
    runs = api.runs(project_name)
    data = []

    for run in runs:
        k = run.config.get("k")
        if k == 45:
            k = 40

        data.append(
            {
                "modelId": run.config.get("lm_name").split("/")[-1],
                "layer": run.config.get("layer"),
                "dSae": run.summary.get("dict_size"),
                # Existing metric
                "ce_loss_score": run.summary.get("core/model_performance_preservation/ce_loss_score"),
                # New metrics
                "autointerp_score": run.summary.get("autointerp/autointerp/autointerp_score"),
                "sae_test_accuracy": run.summary.get("sparse_probing/sae/sae_test_accuracy"),
                "sae_top_1_test_accuracy": run.summary.get("sparse_probing/sae/sae_top_1_test_accuracy"),
                "sae_top_5_test_accuracy": run.summary.get("sparse_probing/sae/sae_top_5_test_accuracy"),
                "l0": k,
            }
        )

    itda_df = pd.DataFrame(data)
    itda_df["saeType"] = "ITDA"
    itda_df["saeClass"] = "ITDA"

    ################################################################
    # 4) Combine filtered SAE data + ITDA data
    ################################################################
    combined_df = pd.concat([df_filtered, itda_df], ignore_index=True)

    # Filter for specific (modelId, layer) combos
    combined_df = combined_df[
        ((combined_df["modelId"] == "pythia-70m-deduped") & (combined_df["layer"] == 3))
        | ((combined_df["modelId"] == "pythia-160m-deduped") & (combined_df["layer"] == 8))
        | ((combined_df["modelId"] == "gemma-2-2b") & (combined_df["layer"] == 12))
    ]

    ################################################################
    # 5) Define the aggregator for "worst" metrics
    ################################################################
    # For metrics where bigger = worse, we use 'max'
    # For metrics where bigger = better, we use 'min'
    # Adjust if you have different scoring conventions
    WORST_AGG = {
        "ce_loss_score": "max",            # higher CE loss => worse
        "autointerp_score": "min",         # assume higher autointerp score => better, so min is worst
        "sae_test_accuracy": "min",        # higher accuracy => better, so min is worst
        "sae_top_1_test_accuracy": "min",  # same reasoning
        "sae_top_5_test_accuracy": "min",  # same reasoning
    }

    ################################################################
    # 6) Plotting function
    ################################################################
    def plot_metric(df, metric_name, y_label=None):
        """
        Plots the given metric (metric_name) for each (modelId, layer) pair
        in the same style as the CE loss plots, but uses the 'worst' 4k SAE
        instead of the best 4k.
        """
        # 1) Get all unique (modelId, layer) pairs
        unique_combos = df[["modelId", "layer"]].drop_duplicates().iloc[::-1]
        n_combos = len(unique_combos)

        # 2) Prepare subplots (one row per combination)
        fig, axes = plt.subplots(n_combos, 1, figsize=(6, 3 * n_combos), sharey=True)

        if n_combos == 1:
            axes = [axes]  # Ensure axes is iterable

        # 3) Loop over each (modelId, layer) pair and plot
        for ax, (model, layer) in zip(axes, unique_combos.values):
            subdf = df[(df["modelId"] == model) & (df["layer"] == layer)]

            # Separate out ITDA vs. SAE
            itda_part = subdf[subdf["saeType"] == "ITDA"].copy()
            sae_part = subdf[subdf["saeType"] == "SAE"].copy()

            # Group by l0 for ITDA => min/max
            itda_grouped = (
                itda_part.groupby("l0")[metric_name].agg(["min", "max"]).reset_index()
            )

            # Group by l0 for all SAE => best overall (unchanged logic: use max)
            #   (This part is still "best" because the user didn't ask to change
            #    the best 16k/64k logic. 
            #    If you want "worst" for that, you can also switch to .min() or .max() 
            #    as appropriate.)
            sae_max_all = sae_part.groupby("l0")[metric_name].max().reset_index()

            # Group by l0 for standard 4k ReLU => use the aggregator for "worst"
            # (Was previously .max() for best; now we do .agg(WORST_AGG[metric_name]))
            sae_std_4096 = (
                sae_part[(sae_part["saeClass"] == "standard") & (sae_part["dSae"] == 4096)]
                .groupby("l0")[metric_name]
                .agg(WORST_AGG[metric_name])
                .reset_index()
            )

            # --- Plot ITDA min/max as lines ---
            if not itda_grouped.empty:
                ax.plot(
                    itda_grouped["l0"],
                    itda_grouped["min"],
                    marker="o",
                    color="blue",
                    label="4k ITDA (min)",
                )
                bigger_dict_label = "64k ITDA" if model in ["pythia-160m-deduped", "gemma-2-2b"] else "16k ITDA"
                ax.plot(
                    itda_grouped["l0"],
                    itda_grouped["max"],
                    marker="o",
                    color="red",
                    label=bigger_dict_label + " (max)",
                )

            # --- Plot the "worst" 4k ReLU SAE
            if not sae_std_4096.empty:
                ax.scatter(
                    sae_std_4096["l0"],
                    sae_std_4096[metric_name],
                    marker="x",
                    s=80,
                    color="purple",
                    label="Worst 4k SAE",
                )

            # --- Plot the best 16k or 64k SAE
            if not sae_max_all.empty:
                bigger_sae_label = "64k SAE" if model in ["pythia-160m-deduped", "gemma-2-2b"] else "16k SAE"
                ax.scatter(
                    sae_max_all["l0"],
                    sae_max_all[metric_name],
                    marker="^",
                    s=80,
                    color="green",
                    label=f"Best {bigger_sae_label}",
                )

            # Labels and title
            ax.set_xlabel("l0")
            ax.set_ylabel(y_label if y_label else metric_name)
            ax.set_title(f"{model}, layer={layer}")
            ax.legend(loc="best")

        plt.tight_layout()
        plt.show()

    ################################################################
    # 7) Generate a figure for each metric
    ################################################################
    metrics_to_plot = [
        ("ce_loss_score", "CE Loss Score"),
        ("autointerp_score", "Autointerp Score"),
        ("sae_test_accuracy", "SAE Test Accuracy"),
        ("sae_top_1_test_accuracy", "SAE Top-1 Accuracy"),
        ("sae_top_5_test_accuracy", "SAE Top-5 Accuracy"),
    ]

    for metric_col, label in metrics_to_plot:
        plot_metric(combined_df, metric_col, y_label=label)

# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    with open("sae_bench_data.json") as f:
        sae_bench_data = json.load(f)

    sae_bench_df = pd.DataFrame(sae_bench_data)
    result_df = sae_bench_df[sae_bench_df["modelId"] != "gemma-2-9b"]
    result_df["saeType"] = "SAE"
    result_df["ce_loss_score"] = result_df[
        "core||model_performance_preservation||ce_loss_score"
    ]
    result_df["l0"] = result_df["core||sparsity||l0"]
    result_df = result_df[
        ["modelId", "layer", "dSae", "ce_loss_score", "saeType", "l0", "saeClass"]
    ]

    targets = np.array([8, 16, 40, 80])

    # Find rows where l0 is within 10% of any target
    mask = np.any(
        (result_df["l0"].values[:, None] >= targets * 0.9)
        & (result_df["l0"].values[:, None] <= targets * 1.1),
        axis=1,
    )

    # Filter DataFrame
    df_filtered = result_df[mask].copy()

    # Round each l0 to the closest target
    df_filtered["l0"] = df_filtered["l0"].apply(
        lambda x: targets[np.argmin(np.abs(targets - x))]
    )

    api = wandb.Api()
    project_name = "itda"
    runs = api.runs(project_name)
    data = []
    for run in runs:
        k = run.config.get("k")
        if k == 45:
            k = 40
        data.append(
            {
                "modelId": run.config.get("lm_name").split("/")[-1],
                "layer": run.config.get("layer"),
                "dSae": run.summary.get("dict_size"),
                "ce_loss_score": run.summary.get(
                    "core/model_performance_preservation/ce_loss_score"
                ),
                "l0": k,
            }
        )

    itda_df = pd.DataFrame(data)
    itda_df["saeType"] = "ITDA"
    itda_df["saeClass"] = "ITDA"

    combined_df = pd.concat([df_filtered, itda_df])

    # filter for modelId = "EleutherAI/pythia-70m-deduped"
    combined_df = combined_df[
        (combined_df["modelId"] == "pythia-70m-deduped") & (combined_df["layer"] == 3)
        | (combined_df["modelId"] == "pythia-160m-deduped")
        & (combined_df["layer"] == 8)
        | (combined_df["modelId"] == "gemma-2-2b") & (combined_df["layer"] == 12)
    ]

    # -------------------------------------------
    # Assuming combined_df is already created
    # -------------------------------------------
    # combined_df columns:
    #   ['modelId', 'layer', 'dSae', 'ce_loss_score', 'saeType', 'l0', 'saeClass']

    # 1) Get all unique (modelId, layer) pairs
    unique_combos = combined_df[["modelId", "layer"]].drop_duplicates().iloc[::-1]

    # 2) Prepare subplots (one per combination)
    n_combos = len(unique_combos)
    fig, axes = plt.subplots(n_combos, 1, figsize=(6, 3 * n_combos), sharey=True)

    if n_combos == 1:
        # If there's only one combination, axes is a single Axes object
        axes = [axes]

    # 3) Loop over each (modelId, layer) pair and plot
    for ax, (model, layer) in zip(axes, unique_combos.values):
        # Subset the data for this model/layer
        subdf = combined_df[
            (combined_df["modelId"] == model) & (combined_df["layer"] == layer)
        ]

        # For convenience, separate out ITDA vs. SAE
        itda_df = subdf[subdf["saeType"] == "ITDA"].copy()
        sae_df = subdf[subdf["saeType"] == "SAE"].copy()

        # Group by l0 for ITDA, get min/max
        itda_grouped = (
            itda_df.groupby("l0")["ce_loss_score"].agg(["min", "max"]).reset_index()
        )

        # Group by l0 for all SAE: get the max overall
        sae_max_all = sae_df.groupby("l0")["ce_loss_score"].max().reset_index()

        # Group by l0 for standard SAE with dSae=4096: get the max
        sae_std_4096 = (
            sae_df[(sae_df["saeClass"] == "standard") & (sae_df["dSae"] == 4096)]
            .groupby("l0")["ce_loss_score"]
            .max()
            .reset_index()
        )

        # --- Plot ITDA min/max as lines ---
        # If there's any ITDA data at all:
        if not itda_grouped.empty:
            ax.plot(
                itda_grouped["l0"],
                itda_grouped["min"],
                marker="o",
                color="blue",
                label="4k ITDA",
            )
            ax.plot(
                itda_grouped["l0"],
                itda_grouped["max"],
                marker="o",
                color="red",
                label=f"{'64k' if model in ['pythia-160m-deduped', 'gemma-2-2b'] else '16k'} ITDA",
            )

        if not sae_std_4096.empty:
            ax.scatter(
                sae_std_4096["l0"],
                sae_std_4096["ce_loss_score"],
                marker="x",
                s=80,
                color="purple",
                label="Best 4k ReLU SAE",
            )

        if not sae_max_all.empty:
            ax.scatter(
                sae_max_all["l0"],
                sae_max_all["ce_loss_score"],
                marker="^",
                s=80,
                color="green",
                label=f"Best {'64k' if model in ['pythia-160m-deduped', 'gemma-2-2b'] else '16k'} SAE",
            )

        # Labels and title
        ax.set_xlabel("l0")
        ax.set_ylabel("ce_loss_score")
        ax.set_title(f"{model}, layer={layer}")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


# %%

# Initialize wandb API client
api = wandb.Api()

# Fetch runs from the project 'example_saes'
project_name = "itda"
runs = api.runs(project_name, filters={"tags": []})

# Extract relevant data into a DataFrame
data = []
for run in runs:
    print(run)
    continue
    # Extract run data
    method = run.config.get("method", "unknown")
    target_loss = run.config.get("target_loss", "unknown")
    max_sequences = run.config.get("max_sequences")
    ce_loss_score = run.summary.get("core/model_performance_preservation/ce_loss_score")
    dict_size = run.summary.get("dict_size")

    if max_sequences is None:
        max_sequences = 10_000

    if (method == "ito") and (target_loss != 0.07):
        continue

    data.append(
        {
            "method": method,
            "target_loss": target_loss,
            "total_training_tokens": max_sequences * 128,
            "ce_loss_score": ce_loss_score,
            "dict_size": dict_size,
        }
    )

# Convert data to DataFrame
df = pd.DataFrame(data)

# get the max ce_loss_score for each pair of method and total_training_tokens
df = df.groupby(["method", "total_training_tokens"]).max().reset_index()

# Group data by method and target_loss
grouped_data = df.groupby(["method", "target_loss"])

# Create a combined figure with two vertically stacked plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

# Plot CE Loss Score
for (method, target_loss), group in grouped_data:
    group = group.sort_values("total_training_tokens")  # Sort by x-axis value

    if method == "ito":
        label = f"ITDA"
    else:
        label = "Top-K SAE"
    ax1.plot(group["total_training_tokens"], group["ce_loss_score"], label=label)

# Customize CE Loss Score plot
ax1.set_ylabel("CE Loss Score")
ax1.set_xscale("log")
ax1.legend(title="Method")
ax1.grid(True)

# Plot Dictionary Size
for (method, target_loss), group in grouped_data:
    group = group.sort_values("total_training_tokens")  # Sort by x-axis value

    if method == "ito":
        label = "ITDA"
    else:
        label = "Top-K SAE"
    ax2.plot(group["total_training_tokens"], group["dict_size"], label=label)

# Customize Dictionary Size plot
ax2.set_xlabel("Total Training Tokens")
ax2.set_xscale("log")
ax2.set_ylabel("Dictionary Size")
ax2.grid(True)

# Adjust layout and show the combined figure
plt.tight_layout()
plt.show()

# %%

l0s = [8, 20, 40, 80]
best_sae = [0.87, 0.96, 0.98, 0.98]
worst_sae = [0.71, 0.84, 0.92, 0.95]

runs = api.runs(project_name, filters={"tags": "performance"})

# Extract relevant data into a DataFrame
data = []
for run in runs:
    # Extract run data
    method = run.config.get("method", "unknown")
    target_loss = run.config.get("target_loss", "unknown")
    max_sequences = run.config.get("max_sequences")
    ce_loss_score = run.summary.get("core/model_performance_preservation/ce_loss_score")
    l0 = run.config.get("l0")
    dict_size = run.summary.get("dict_size")

    data.append(
        {
            "ce_loss_score": ce_loss_score,
            "l0": l0,
            "dict_size": dict_size,
        }
    )

df = pd.DataFrame(data)
df = df.dropna(subset=["ce_loss_score"])
best_worst_scores = df.groupby("l0")["ce_loss_score"].agg(["min", "max"]).reset_index()
best_worst_scores.rename(
    columns={"min": "worst_ce_loss_score", "max": "best_ce_loss_score"}, inplace=True
)
l0s = best_worst_scores["l0"].tolist()
best_ito = best_worst_scores["best_ce_loss_score"].tolist()
worst_ito = best_worst_scores["worst_ce_loss_score"].tolist()

best_ito_df = df[df["ce_loss_score"].isin(best_ito)].sort_values("l0")
worst_ito_df = df[df["ce_loss_score"].isin(worst_ito)].sort_values("l0")

# Create subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

# First subplot: Performance by L0
ax1.fill_between(l0s, best_sae, worst_sae, color="blue", alpha=0.1, label="SAE Bounds")
ax1.fill_between(l0s, best_ito, worst_ito, color="green", alpha=0.1, label="ITO Bounds")
ax1.plot(l0s, best_sae, marker="o", label="Overall Best SAE", color="green")
ax1.plot(l0s, worst_sae, marker="o", label="Best ReLU SAE", color="red")
ax1.plot(l0s, best_ito, marker="o", label="Best ITDA", linestyle="--", color="green")
ax1.plot(l0s, worst_ito, marker="o", linestyle="--", label="Worst ITDA", color="red")
ax1.set_title("Performance by L0", fontsize=14)
ax1.set_ylabel("CE Loss Score", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True)

# Second subplot: Dictionary Sizes of Best and Worst ITDAs
ax2.plot(
    best_ito_df["l0"],
    best_ito_df["dict_size"],
    marker="o",
    linestyle="-",
    color="green",
    label="Best ITDA Dictionary Sizes",
)
ax2.plot(
    worst_ito_df["l0"],
    worst_ito_df["dict_size"],
    marker="o",
    linestyle="--",
    color="red",
    label="Worst ITDA Dictionary Sizes",
)
ax2.set_yscale("log")
ax2.set_title("Dictionary Sizes of Best and Worst ITDAs", fontsize=14)
ax2.set_xlabel("l0", fontsize=12)
ax2.set_ylabel("Dictionary Size", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()
