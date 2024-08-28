import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from tqdm import tqdm
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA


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
        if dataset_name_ != "Statlog (German Credit Data)":
            continue
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
    residual_variances = []

    # Loop over different target dimensions
    for dim in dimensions:
        lle = LocallyLinearEmbedding(n_components=dim, method='standard', n_neighbors=10)
        data_lle = lle.fit_transform(df)

        # Calculate residual variance
        original_distances = np.linalg.norm(df - df.mean(axis=0), axis=1)
        embedded_distances = np.linalg.norm(data_lle - data_lle.mean(axis=0), axis=1)
        residual_variance = 1 - (np.var(embedded_distances) / np.var(original_distances))
        residual_variances.append(residual_variance)

        print(f"Dimension: {dim}, Residual Variance: {residual_variance}")

    # Plot the residual variances to find the "elbow" where the variance stops decreasing significantly
    plt.figure()
    plt.plot(dimensions, residual_variances, marker='o')
    plt.title('Residual Variance vs. Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Residual Variance')
    plt.grid(True)
    plt.savefig("lle_residual_variance.png")
