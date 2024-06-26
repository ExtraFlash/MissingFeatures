########## PROCESS DATA AND SAVE TRAIN VALIDATION AND TEST SETS ##########

import os
import pathlib
import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_data(dataset_name: str, relative_path: str, label_position: int, has_header: bool, has_id: str,
                 positive: str, negative: str, sep):
    # Load data
    datasets_path = "../datasets"

    # Define a custom delimiter regex
    delimiter_pattern = '\s+' if sep is not None else ','

    path = datasets_path + '/' + relative_path
    if not has_header:
        data = pd.read_csv(path, header=None, sep=delimiter_pattern)
    else:
        data = pd.read_csv(path, sep=delimiter_pattern)

    # Remove id column
    if has_id:
        data.drop(data.columns[0], axis=1, inplace=True)

    # shuffle
    data = shuffle(data, random_state=42)
    data.reset_index(drop=True, inplace=True)

    # Rearrange columns
    if label_position == "start":
        cols = data.columns.tolist()
        cols = cols[1:] + [cols[0]]
        data = data[cols]

    # Replace categories with numbers for the label
    if positive:  # if there is a positive and negative strings, replace them with 1 and 0
        # categories = data.iloc[:, -1].unique()
        data.iloc[:, -1].replace([positive, negative], [1, 0], inplace=True)

    # Split data into train, validation and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    # train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Standardize data
    # standard_scaler = StandardScaler()
    # data.iloc[:, :-1] = standard_scaler.fit_transform(data.iloc[:, :-1])

    # Create directories
    data_path = "../data"
    if not os.path.exists(f"{data_path}/{dataset_name}"):
        os.makedirs(f"{data_path}/{dataset_name}")
    if not os.path.exists(f"{data_path}/{dataset_name}/train"):
        os.makedirs(f"{data_path}/{dataset_name}/train")
    # if not os.path.exists(f"{data_path}/{dataset_name}/val"):
    #     os.makedirs(f"{data_path}/{dataset_name}/val")
    if not os.path.exists(f"{data_path}/{dataset_name}/test"):
        os.makedirs(f"{data_path}/{dataset_name}/test")

    # Save data
    train.to_csv(f"{data_path}/{dataset_name}/train/data.csv", index=False)
    # val.to_csv(f"{data_path}/{dataset_name}/val/data.csv", index=False)
    test.to_csv(f"{data_path}/{dataset_name}/test/data.csv", index=False)


def load_datasets_from_config(config_path):
    # Load config file
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # process each dataset
    for dataset in datasets:
        dataset_name = dataset['name']
        relative_path = dataset['relative_path']
        label_position = dataset['label_position']
        has_header = dataset['has_header']
        has_id = dataset['has_id']
        positive = dataset.get('positive', None)
        negative = dataset.get('negative', None)
        sep = dataset.get('sep', None)
        process_data(dataset_name, relative_path, label_position, has_header, has_id, positive, negative, sep)


if __name__ == "__main__":
    load_datasets_from_config("../datasets/config.json")
