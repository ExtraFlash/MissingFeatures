import torch
import pandas as pd
import numpy as np
import json
import os


def create_data(n_samples, n_features, n_rank, n_categorical, ratios, n_categories=2, n_output=1):
    """
    Create a dataset with n_samples, n_features and n_numerical numerical features
    :param n_samples: number of samples
    :param n_features: number of features
    :param n_numerical: number of numerical features
    :return: pandas DataFrame
    """
    n_numerical = n_features - n_categorical
    # data = {}
    # for i in range(n_rank):
    #     if i < n_rank:
    #         data[f'num_{i}'] = torch.randn(n_samples)
    #     else:
    #         data[f'fac_{i}'] = np.random.choice(range(n_categories), n_samples)
    # df = pd.DataFrame(data)

    # init data in low rank space
    init_data = torch.randn(n_samples, n_rank)

    # df to torch
    # df_tensor = torch.tensor(df.values, dtype=torch.float32)

    # manifold matrix to project the data to high rank space
    manifold_matrix = torch.randn(n_rank, n_features)
    # input data in high rank space
    input_data = torch.mm(init_data, manifold_matrix)

    check_ratios = []

    # change numerical data to binary
    for i in range(len(ratios)):
        # get the threshold from the ratios list
        sorted_data = torch.sort(input_data[:, i])[0]
        threshold = sorted_data[int(n_samples * (1 - ratios[i]))]
        # change to binary feature according to threshold
        input_data[:, i] = (input_data[:, i] > threshold).int()

        check_ratios.append((input_data[:, i] == 1).sum().item() / n_samples)

    n_hidden = 100

    # layers
    hidden1_matrix = torch.randn(n_features, n_hidden)
    output_matrix = torch.randn(n_hidden, n_output)

    # forward
    X = torch.mm(input_data, hidden1_matrix)
    X = torch.relu(X)
    output = torch.mm(X, output_matrix)
    # apply sigmoid
    probs = torch.sigmoid(output)

    data = {}
    for i in range(n_features):
        data[f'num_{i}'] = input_data[:, i]
    data['label'] = (probs > 0.5).int().view(-1)

    df = pd.DataFrame(data)

    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/data.csv", index=False)

    # print([x - y for x, y in zip(ratios, check_ratios)])


# create_data(1000, 10, 5, 5)
with open("generated_data_config.json") as f:
    config = json.load(f)
n_samples = config['n_samples']
n_features = config['n_features']
n_rank = config['n_rank']
n_categorical = config['n_categorical']

dataset = "Statlog (German Credit Data)"

with open(f"../../check_bias_in_categorical_datasets/results_{dataset}.json") as f:
    ratios = json.load(f)

create_data(n_samples, n_features, n_rank, n_categorical, ratios)
