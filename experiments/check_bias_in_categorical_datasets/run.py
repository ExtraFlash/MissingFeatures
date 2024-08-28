import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm


def save_results_for_dataset(dataset_name: str, is_multy_class: bool):
    # Load data
    data_path = "../../data"
    # train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")
    X_test = test.iloc[:, :-1]

    non_categorical_columns = [col for col in X_test.columns[:-1] if X_test[col].nunique() > 2]
    categorical_columns = [col for col in X_test.columns[:-1] if col not in non_categorical_columns]

    ratios = []

    for categorical_column in categorical_columns:
        # print(categorical_column)
        ratio_of_ones = X_test[categorical_column].mean()
        ratios.append(ratio_of_ones)

    plt.clf()
    plt.hist(ratios, bins=20)
    plt.xlabel('Ratio of ones')
    plt.xlim(0, 1)
    plt.ylabel('Amount of features')
    plt.title(f'{dataset_name}, categorical features: {len(categorical_columns)} / {(len(non_categorical_columns) + len(categorical_columns))}')
    # plt.show()
    plt.savefig(f"results_{dataset_name}.png")

    with open(f"results_{dataset_name}.json", 'w') as f:
        json.dump(ratios, f)


if __name__ == "__main__":
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # run experiment for each dataset
    for dataset in tqdm(datasets):
        dataset_name_ = dataset['name']
        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']

        save_results_for_dataset(dataset_name_, is_multy_class_)