import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import optuna
from functools import partial
import pickle

from models import ModelFactory

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def objective(trial, X_train, y_train):
    # Determine the columns for preprocessing
    non_categorical_columns = [col for col in X_train.columns if X_train[col].nunique() > 2]
    categorical_columns = [col for col in X_train.columns if col not in non_categorical_columns]

    # Create a ColumnTransformer for preprocessing
    scaler_pipeline = ColumnTransformer([
        ('scaler', StandardScaler(), non_categorical_columns)
    ], remainder='passthrough')

    # Hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 32)

    # Create the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    # Create a pipeline that includes the scaler and the model
    pipeline = Pipeline([
        ('preprocessing', scaler_pipeline),
        ('classifier', model)
    ])

    # Evaluate the pipeline using cross-validation
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score


def optimize_model_for_dataset(dataset_name: str):
    # Load data
    data_path = "../data"
    train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")

    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

    # Create a new function with X_train and y_train pre-filled
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train)
    #
    # # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=1000)

    _best_params = study.best_params

    # Create directories
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    if not os.path.exists(f"{dataset_name}/{model_name}"):
        os.makedirs(f"{dataset_name}/{model_name}")

    # Save the best hyperparameters
    with open(f"{dataset_name}/{model_name}/best_params.json", 'w') as f:
        json.dump(_best_params, f)

    # Save the study
    fig_slice = optuna.visualization.plot_slice(study,
                                    params=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'])
    fig_slice.write_image(f"{dataset_name}/{model_name}/slice_plot.png")

    fig_param_importances = optuna.visualization.plot_param_importances(study)
    fig_param_importances.write_image(f"{dataset_name}/{model_name}/param_importances_plot.png")

    # Run the model with the best hyperparameters
    model = RandomForestClassifier(**_best_params)

    non_categorical_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2]
    categorical_columns = [col for col in train.columns[:-1] if col not in non_categorical_columns]
    target_column = [train.columns[-1]]
    original_columns = train.columns

    scaler_pipeline = ColumnTransformer([
        ('scaler', StandardScaler(), non_categorical_columns)
    ], remainder='passthrough')

    train = pd.DataFrame(scaler_pipeline.fit_transform(train),
                         columns=non_categorical_columns + categorical_columns + target_column)
    # keep the order of columns
    train = train[original_columns]

    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

    model.fit(X_train, y_train)

    # Save the model
    with open(f"{dataset_name}/{model_name}/model.pkl", 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    config_path = "../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)

    model_name = ModelFactory.Random_Forest_NAME

    # get datasets from config
    datasets: list = config['datasets']
    # run experiment for each dataset
    for dataset in tqdm(datasets):
        dataset_name_ = dataset['name']
        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']

        optimize_model_for_dataset(dataset_name_)
