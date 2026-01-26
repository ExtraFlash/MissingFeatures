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


def check_homophily(dataset_name: str) -> None:
    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    X_train, y_train, X_val, y_val = utils.preprocess_split(train_set)

    X = pd.concat([X_train, X_val])
    y = pd.concat([y_train, y_val])

    # Check homophily
    # Build KNN graph
    k = 5  # Number of neighbors
    knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)

    # Convert sparse adjacency matrix to a list of neighbors
    neighbors = [np.where(row.toarray().flatten())[0] for row in knn_graph]

    # Compute homophily score
    y_np = y.to_numpy()
    homophily_ratios = [(y_np[i] == y_np[neighbors[i]]).mean() for i in range(len(y))]
    avg_homophily = np.mean(homophily_ratios)
    print(f"Average homophily score for {dataset_name}: {avg_homophily}")

    results['datasets'].append(dataset_name)
    results['scores'].append(avg_homophily)




if __name__ == "__main__":
    config_path = "../../datasets/config.json"
    results = {
        'datasets': [],
        'scores': []
    }
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']

    for dataset in tqdm(datasets):
        dataset_name_ = dataset['name']
        check_homophily(dataset_name_)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("homophily_scores.csv", index=False)
