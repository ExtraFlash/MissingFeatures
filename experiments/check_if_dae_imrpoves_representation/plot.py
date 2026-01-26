import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from my_models import ModelFactory
from plots_utils import utils


MODELS_NAMES = [
    ModelFactory.Random_Forest_NAME,
    ModelFactory.Ada_Boost_NAME,
    ModelFactory.XGB_NAME
]


def get_max_features_amount(dataset_names: list):
    max_features_amount = 0
    for dataset_name in dataset_names:
        data = pd.read_csv(f"{dataset_name}/results_with_dae.csv")
        features_amounts = data["features_amounts"].to_list()
        features_amount = max(features_amounts)
        if features_amount > max_features_amount:
            max_features_amount = features_amount

    return max_features_amount


def make_subplot_for_model(model_name: str, dataset_names: list,
                           axs,
                           plot_col_index: int):
    """
    For results with dae:
    - load results from each dataset
    - calculate mean and std for this model
    - plot the results
    """
    utils.make_style(plt)
    ax = axs[plot_col_index]

    max_features_amount = get_max_features_amount(dataset_names)

    results_with_dae = {features_amount: [] for features_amount in range(1, max_features_amount + 1)}
    for features_amount in range(1, max_features_amount + 1):
        for dataset_name in dataset_names:
            data = pd.read_csv(f"{dataset_name}/results_with_dae.csv")
            scores = data.loc[data["features_amounts"] == features_amount, model_name].to_list()

            if len(scores) >= 2:
                raise Exception(f"More than one score for features amount {features_amount} in dataset {dataset_name}")
            if scores:
                results_with_dae[features_amount].append(scores[0])

    results_without_dae = {features_amount: [] for features_amount in range(1, max_features_amount + 1)}
    for features_amount in range(1, max_features_amount + 1):
        for dataset_name in dataset_names:
            data = pd.read_csv(f"{dataset_name}/results_without_dae.csv")
            scores = data.loc[data["features_amounts"] == features_amount, model_name].to_list()

            if len(scores) >= 2:
                raise Exception(f"More than one score for features amount {features_amount} in dataset {dataset_name}")
            if scores:
                results_without_dae[features_amount].append(scores[0])

    results_only_dae = {features_amount: [] for features_amount in range(1, max_features_amount + 1)}
    for features_amount in range(1, max_features_amount + 1):
        for dataset_name in dataset_names:
            data = pd.read_csv(f"{dataset_name}/results_only_dae.csv")
            scores = data.loc[data["features_amounts"] == features_amount, model_name].to_list()

            if len(scores) >= 2:
                raise Exception(f"More than one score for features amount {features_amount} in dataset {dataset_name}")
            if scores:
                results_only_dae[features_amount].append(scores[0])


    # make plot
    features_amounts = list(range(1, max_features_amount + 1))
    x_ticks = features_amounts[::(len(features_amounts) // 5) + 1]
    if 1 not in x_ticks:
        x_ticks.append(1)

    means_with_dae = np.array([np.mean(results_with_dae[feature_amount]) for feature_amount in features_amounts])
    stds_with_dae = np.array([np.std(results_with_dae[feature_amount], ddof=1) / np.sqrt(np.size(results_with_dae[feature_amount])) for feature_amount in features_amounts])

    ax.plot(features_amounts, means_with_dae, label="With DAE")
    ax.fill_between(features_amounts, means_with_dae+stds_with_dae, means_with_dae-stds_with_dae, alpha=0.2)

    means_without_dae = np.array([np.mean(results_without_dae[feature_amount]) for feature_amount in features_amounts])
    stds_without_dae = np.array([np.std(results_without_dae[feature_amount], ddof=1) / np.sqrt(np.size(results_without_dae[feature_amount])) for feature_amount in features_amounts])

    ax.plot(features_amounts, means_without_dae, label="Without DAE")
    ax.fill_between(features_amounts, means_without_dae+stds_without_dae, means_without_dae-stds_without_dae, alpha=0.2)

    means_only_dae = np.array([np.mean(results_only_dae[feature_amount]) for feature_amount in features_amounts])
    stds_only_dae = np.array([np.std(results_only_dae[feature_amount], ddof=1) / np.sqrt(np.size(results_only_dae[feature_amount])) for feature_amount in features_amounts])

    ax.plot(features_amounts, means_only_dae, label="Only DAE")
    ax.fill_between(features_amounts, means_only_dae+stds_only_dae, means_only_dae-stds_only_dae, alpha=0.2)

    ax.set_xlabel("Num features")
    ax.invert_xaxis()
    ax.grid()
    ax.legend()
    ax.set_title(model_name)






def main():
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']

    datasets_names = [dataset['name'] for dataset in datasets]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(13, 7), layout="constrained")

    col = 0
    for model_name in MODELS_NAMES:
        make_subplot_for_model(model_name, datasets_names, axs, plot_col_index=col)
        col += 1

    plt.savefig("comparison_with_and_without_dae.png")


if __name__ == "__main__":
    main()








# Load the datasets
# dataset_with_dae = pd.read_csv("results_with_dae.csv")
# dataset_without_dae = pd.read_csv("results_without_dae.csv")
#
# # Calculate the average AUC scores and standard errors for each model in each dataset
# avg_auc_with_dae = dataset_with_dae.mean()
# avg_auc_without_dae = dataset_without_dae.mean()
# stderr_auc_with_dae = dataset_with_dae.sem()  # Standard error of the mean
# stderr_auc_without_dae = dataset_without_dae.sem()
#
# # Combine the averages and standard errors into a single DataFrame for plotting
# comparison_df = pd.DataFrame({
#     "With DAE": avg_auc_with_dae,
#     "Without DAE": avg_auc_without_dae
# })
# errors_df = pd.DataFrame({
#     "With DAE": stderr_auc_with_dae,
#     "Without DAE": stderr_auc_without_dae
# })
#
# # Create the plot
# fig, ax = plt.subplots(figsize=(12, 7))
#
# # Define colors for the bars
# colors = ['#92C5F9', '#87BB62']  # Blue-green palette
#
# # Plot bars with error bars
# comparison_df.plot(kind='bar', yerr=errors_df, ax=ax, color=colors, capsize=4, edgecolor='black')
#
# # Add plot enhancements
# plt.title("Comparison of Average AUC Scores with Standard Errors", fontsize=16)
# plt.ylabel("Average AUC Score", fontsize=14)
# plt.xlabel("Models", fontsize=14)
# plt.xticks(rotation=0, fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(title="Datasets", fontsize=12, title_fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# # Improve layout
# plt.tight_layout()
#
# # Show the plot
# plt.savefig("comparison_with_and_without_dae.png")
