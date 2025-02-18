# %%
import json
import pandas as pd

if __name__ == "__main__":
    with open("sae_bench_data.json") as f:
        sae_bench_data = json.load(f)

    sae_bench_df = pd.DataFrame(sae_bench_data)
    filtered_df = sae_bench_df[(sae_bench_df["core||sparsity||l0"] > 35) & (sae_bench_df["core||sparsity||l0"] < 45)]
    filtered_df = filtered_df[filtered_df["modelId"] != "gemma-2-9b"]
    result_df = (
        filtered_df
        .groupby(["modelId", "layer", "dSae"])["core||model_performance_preservation||ce_loss_score"]
        .max()
        .reset_index()
    )
    result_df = result_df.sort_values(by=["modelId", "dSae"], ascending=[True, False])
    result_df["saeType"] = "SAE"
    


# %%
import wandb
import matplotlib.pyplot as plt
import pandas as pd

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
    label="Best ITDA Dictionary Sizes"
)
ax2.plot(
    worst_ito_df["l0"],
    worst_ito_df["dict_size"],
    marker="o",
    linestyle="--",
    color="red",
    label="Worst ITDA Dictionary Sizes"
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

