import pandas as pd
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import numpy as np
from my_models import ModelFactory


def main():
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    datasets_names = [dataset['name'] for dataset in datasets]

    """
    pipeline:
    For each model, make a curve in the plot where the x axis are the number of features
    and the y axis is the AUC.
    For each number of features, take the avg and std of the AUC of all datasets for this model
    """
    models_names = [ModelFactory.DAE_NAME, ModelFactory.DAE_Dynamic_TYPE_LIGHTNING_NAME]

    max_num_of_features = 0
    for dataset in tqdm(datasets):
        dataset_name = dataset['name']
        data = pd.read_csv(f"{dataset_name}/results.csv")
        num_of_features = data['features_amounts'].max()
        if num_of_features > max_num_of_features:
            max_num_of_features = num_of_features

    results_with_dae = {features_amount: [] for features_amount in range(1, max_num_of_features + 1)}

    for features_amount in range(1, max_num_of_features + 1):
        for dataset_name in datasets_names:
            data = pd.read_csv(f"{dataset_name}/results.csv")
            scores = data.loc[data["features_amounts"] == features_amount, ModelFactory.DAE_NAME].to_list()

            if len(scores) >= 2:
                raise Exception(f"More than one score for features amount {features_amount} in dataset {dataset_name}")
            if scores:
                results_with_dae[features_amount].append(scores[0])

    results_with_dae_typing = {features_amount: [] for features_amount in range(1, max_num_of_features + 1)}

    for features_amount in range(1, max_num_of_features + 1):
        for dataset_name in datasets_names:
            data = pd.read_csv(f"{dataset_name}/results.csv")
            scores = data.loc[data["features_amounts"] == features_amount, ModelFactory.DAE_Dynamic_TYPE_LIGHTNING_NAME].to_list()

            if len(scores) >= 2:
                raise Exception(f"More than one score for features amount {features_amount} in dataset {dataset_name}")
            if scores:
                results_with_dae_typing[features_amount].append(scores[0])

    # make plot
    features_amounts = list(range(1, max_num_of_features + 1))
    x_ticks = features_amounts[::(len(features_amounts) // 5) + 1]

    # Create a figure and axes
    fig, ax = plt.subplots()

    if 1 not in x_ticks:
        x_ticks.append(1)

    means_with_dae = np.array([np.mean(results_with_dae[feature_amount]) for feature_amount in features_amounts])
    stds_with_dae = np.array(
        [np.std(results_with_dae[feature_amount], ddof=1) / np.sqrt(np.size(results_with_dae[feature_amount])) for
         feature_amount in features_amounts])
    ax.plot(features_amounts, means_with_dae, label="DAE")
    ax.fill_between(features_amounts, means_with_dae + stds_with_dae, means_with_dae - stds_with_dae, alpha=0.2)

    means_with_dae_typing = np.array([np.mean(results_with_dae_typing[feature_amount]) for feature_amount in features_amounts])
    stds_with_dae_typing = np.array(
        [np.std(results_with_dae_typing[feature_amount], ddof=1) / np.sqrt(np.size(results_with_dae_typing[feature_amount])) for
         feature_amount in features_amounts])
    ax.plot(features_amounts, means_with_dae_typing, label="DAE Typing")
    ax.fill_between(features_amounts, means_with_dae_typing + stds_with_dae_typing, means_with_dae_typing - stds_with_dae_typing, alpha=0.2)

    ax.set_xlabel("Num features")
    ax.set_ylabel("AUC")
    ax.invert_xaxis()
    ax.grid()
    ax.legend()
    ax.set_title("AUC vs Num features")
    plt.savefig("auc_vs_num_features.png")


if __name__ == '__main__':
    main()
