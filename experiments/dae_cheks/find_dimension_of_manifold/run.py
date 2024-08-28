import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import Isomap

if __name__ == '__main__':
    # load data
    df = pd.read_csv("data/data.csv")

    # Apply a non-linear function like ReLU or Sigmoid
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    relu = lambda x: np.maximum(0, x)

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
    plt.savefig("isomap_reconstruction_error.png")
