import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go


if __name__ == '__main__':
    # load data
    df = pd.read_csv("data/data.csv")

    # Apply sigmoid to the original data
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))
    # relu = lambda x: np.maximum(0, x)
    # df = sigmoid(df)

    # standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # apply PCA
    pca = PCA(n_components=2)
    pca.fit(df_scaled)
    df_pca = pca.transform(df_scaled)

    pca_2d_to_3d = pca.inverse_transform(df_pca)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original 3D data
    ax.scatter(df_scaled[:, 0], df_scaled[:, 1], df_scaled[:, 2], color='blue',
               label='Original 3D data')

    # Plot the 2D points transformed back to 3D space
    ax.scatter(pca_2d_to_3d[:, 0], pca_2d_to_3d[:, 1], pca_2d_to_3d[:, 2], color='red',
               label='PCA 2D points in 3D space')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("pca_on_linear_data.png")

    # Create a 3D scatter plot using plotly
    fig = go.Figure()

    # Add the original 3D data points
    fig.add_trace(go.Scatter3d(
        x=df_scaled[:, 0],
        y=df_scaled[:, 1],
        z=df_scaled[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Original 3D Data'
    ))

    # Add the 2D PCA points transformed back to 3D space
    fig.add_trace(go.Scatter3d(
        x=pca_2d_to_3d[:, 0],
        y=pca_2d_to_3d[:, 1],
        z=pca_2d_to_3d[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='PCA 2D Points in 3D Space'
    ))

    # Set plot labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Plot of Original Data and PCA-Reduced Data',
    )

    # Show the plot
    fig.show()



