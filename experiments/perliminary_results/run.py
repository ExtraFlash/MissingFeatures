import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from models import ModelFactory

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import learning_curve


def plot_learning_curve(dataset_name, model_name, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(f"results_{dataset_name}_{model_name}.png")
    return plt


def run_experiment(dataset_name: str, model_name: str, relative_path: str, label_position: int, has_header: bool, has_id: str):
    # Load data
    data_path = "../../data"
    train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    val = pd.read_csv(f"{data_path}/{dataset_name}/val/data.csv")
    test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")

    # Split features and labels
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    # Get model
    model = ModelFactory.get_model(model_name)

    plot_learning_curve(dataset_name, model_name, model, f"Learning curve for {model_name}", X_train, y_train, cv=5)


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

        models_names = ModelFactory.MODELS

        for model_name in models_names:
            run_experiment(dataset_name, model_name, relative_path, label_position, has_header, has_id)


if __name__ == "__main__":
    load_datasets_from_config("../../datasets/config.json")



