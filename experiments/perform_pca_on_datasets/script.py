import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from my_models import ModelFactory

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

from utils import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_cumulative_variance(dataset_name: str, data_path: str = "../../data"):
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    X_train, y_train, X_val, y_val = utils.preprocess_split(train_set)
    X = pd.concat([X_train, X_val])
    X_std = StandardScaler().fit_transform(X)

    pca = PCA()
    pca.fit(X_std)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    return cumulative_var

def perform_all_pca_plots(datasets):
    n_datasets = len(datasets)
    n_cols = 4
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, dataset in enumerate(tqdm(datasets)):
        dataset_name = dataset["name"]
        cumulative_var = get_cumulative_variance(dataset_name)

        axes[i].plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o')
        axes[i].axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
        axes[i].set_title(dataset_name)
        axes[i].set_xlabel("Principal Components")
        axes[i].set_ylabel("Cumulative Variance")
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True)

    # Remove unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig("pca_plot.png", bbox_inches='tight')


if __name__ == "__main__":
    config_path = "../../datasets/config.json"
    with open(config_path) as f:
        config = json.load(f)
    datasets: list = config['datasets']
    perform_all_pca_plots(datasets)
