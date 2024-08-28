import torch
import pandas as pd
import numpy as np
import json
import os


def create_data(n_samples, n_features, n_rank, n_output=1):
    """
    Create a dataset with n_samples, n_features and n_numerical numerical features
    :param n_samples: number of samples
    :param n_features: number of features
    :param n_numerical: number of numerical features
    :return: pandas DataFrame
    """

    # init data in low rank space
    init_data = torch.randn(n_samples, n_rank)

    # manifold matrix to project the data to high rank space
    manifold_matrix = torch.randn(n_rank, n_features)

    # input data in high rank space
    input_data = torch.mm(init_data, manifold_matrix)

    # perform sigmoid
    input_data = torch.sigmoid(input_data)

    df = pd.DataFrame(input_data)

    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/data.csv", index=False)


create_data(1000, 3, 2)
