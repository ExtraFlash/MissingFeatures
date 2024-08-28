import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from tqdm import tqdm
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding


if __name__ == '__main__':
    # load data
    config_path = "../../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # run experiment for each dataset
    for dataset in tqdm(datasets):
        dataset_name_ = dataset['name']
        # if dataset_name_ != "Statlog (German Credit Data)":
        #     continue
        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']

        data_path = "../../../data"
        df = pd.read_csv(f"{data_path}/{dataset_name_}/train/data.csv")

        # Apply a non-linear function like ReLU or Sigmoid
        # sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # relu = lambda x: np.maximum(0, x)

        # Uncomment one of the following based on your needs
        # data = sigmoid(df)  # Apply sigmoid
        # data = relu(df)  # Apply ReLU

        # Initialize lists to store results
        dimensions = range(1, df.shape[1] + 1)
        errors = []

        # Loop over different target dimensions
        for dim in dimensions:
            isomap = Isomap(n_components=dim)
            data_isomap = isomap.fit_transform(df)
            error = isomap.reconstruction_error()
            errors.append(error)

            print(f"Dimension: {dim}, Reconstruction Error: {error}")

        # Plot the errors to find the "elbow" where the error stops decreasing significantly
        plt.figure()
        plt.plot(dimensions, errors, marker='o')
        plt.title('Reconstruction Error vs. Dimensions')
        plt.xlabel('Dimensions')
        plt.ylabel('Reconstruction Error')
        plt.grid(True)
        plt.savefig(f'{dataset_name_}_isomap_reconstruction_error.png')
