import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments.calculate_datasets_scores import plot_style


if __name__ == '__main__':

    plot_style.set_plot_style()

    df = pd.read_csv('homophily_scores.csv')
    homophily_scores  = df['scores']
    datasets = df['datasets']

    # Sort datasets by homophily score for better visualization
    sorted_indices = np.argsort(homophily_scores)
    datasets = [datasets[i] for i in sorted_indices]
    homophily_scores = [homophily_scores[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.barh(datasets, homophily_scores, color="skyblue", edgecolor="black")
    plt.xlabel("Homophily Score")
    plt.ylabel("Datasets")
    plt.title("Homophily Scores Across Datasets")
    plt.xlim(0.4, 1)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show values on bars
    for index, value in enumerate(homophily_scores):
        plt.text(value + 0.02, index, f"{value:.2f}", va="center", fontsize=10)

    plt.savefig("homophily_scores.png", bbox_inches="tight")
