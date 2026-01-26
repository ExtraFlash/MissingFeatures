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
import torch.nn as nn

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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from my_models import ActivationFactory

from utils import utils

BATCH_SIZE = 64

dims = {
    'Tokyo': 15,
    'Connectionist Bench': 15,
    'Ionosphere': 35,
    'Pima Indians Diabetes Database': 20,
    'Heart Disease': 22,
    'Statlog (German Credit Data)': 30,
}


def objective(trial, X_train, y_train, X_val, y_val, dataset_name):

    # Hyperparameters to optimize
    latent_dim = trial.suggest_int('latent_dim', 10, 50)
    # latent_dim = dims[dataset_name]
    encoder_layers = trial.suggest_int('encoder_layers', 1, 3)

    encoder_dims = []
    previous_size = 256
    for i in range(encoder_layers):
        current_size = trial.suggest_int(f'encoder_units_{i}', 64, previous_size)
        encoder_dims.append(current_size)
        previous_size = current_size

    # Decoder units are the reverse of encoder units
    # decoder_units = encoder_units[::-1]

    # Convert to tuple
    # encoder_units = tuple(encoder_units)
    # decoder_units = tuple(decoder_units)

    # activation_name = trial.suggest_categorical('activation_name',
    #                                             [ActivationFactory.relu_NAME, ActivationFactory.leaky_relu_NAME,
    #                                              ActivationFactory.tanh_NAME])
    activation_name = ActivationFactory.leaky_relu_NAME
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = BATCH_SIZE
    # n_epochs = trial.suggest_int('n_epochs', 200, 1500)

    dae, _ = ModelFactory.get_model(
        model_name=model_name,
        input_size=X_train.shape[1],
        latent_dim=latent_dim,
        encoder_dims=encoder_dims,
        activation_name=activation_name,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    dae.fit(X_train, y_train, X_val, y_val, show_progress=False)

    # y_pred = dae.predict(X_val)
    # accuracy = (y_pred.flatten() == y_val).mean()

    y_val_probs = dae.predict_proba(X_val)
    auc = roc_auc_score(y_val, y_val_probs[:, 1])

    return auc


def optimize_model_for_dataset(dataset_name: str):
    # Load data
    data_path = "../data"
    train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")

    X_train, y_train, X_val, y_val = utils.preprocess_split(train)

    # Create a new function with X_train and y_train pre-filled
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, dataset_name=dataset_name)

    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=400)

    _best_params = study.best_params

    # drop the encoder_layers key
    encoder_layers = _best_params.pop('encoder_layers')
    # Instead of having encoder_units_{i} as keys, we will have encoder_dims as a list
    encoder_dims = [_best_params[f'encoder_units_{i}'] for i in range(encoder_layers)]
    _best_params['encoder_dims'] = encoder_dims
    # drop the encoder_units_{i} keys according to encoder_layers amount
    for i in range(encoder_layers):
        _best_params.pop(f'encoder_units_{i}')

    # Create directories
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    if not os.path.exists(f"{dataset_name}/{model_name}"):
        os.makedirs(f"{dataset_name}/{model_name}")

    # Save the best hyperparameters
    with open(f"{dataset_name}/{model_name}/best_params.json", 'w') as f:
        json.dump(_best_params, f)

    # Save the study

    # fig_slice = optuna.visualization.plot_slice(study,
    #                                             params=['latent_dim',
    #                                                     'encoder_layers'] +
    #                                                    [f'encoder_units_{i}' for i in
    #                                                     range(encoder_layers)] +
    #                                                    ['dropout_rate', 'learning_rate'])
    # fig_slice.write_image(f"{dataset_name}/{model_name}/slice_plot.png")
    #
    # fig_param_importances = optuna.visualization.plot_param_importances(study)
    # fig_param_importances.write_image(f"{dataset_name}/{model_name}/param_importances_plot.png")

    # Extract encoder dimensions
    # encoder_dims = [_best_params[f'encoder_units_{i}'] for i in range(encoder_layers)]

    # Run the model with the best hyperparameters
    # dae, _ = ModelFactory.get_model(
    #     model_name=model_name,
    #     input_size=X_train.shape[1],
    #     latent_dim=_best_params['latent_dim'],
    #     encoder_dims=encoder_dims,
    #     activation_name=ActivationFactory.leaky_relu_NAME,
    #     dropout_rate=_best_params['dropout_rate'],
    #     learning_rate=_best_params['learning_rate'],
    # )

    # dae.fit(X_train, y_train, X_val, y_val)
    #
    # # Save the model
    # dae.save_checkpoint(f"{dataset_name}/{model_name}/{model_name}.ckpt")


if __name__ == "__main__":
    config_path = "../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)

    model_name = ModelFactory.DAE_NAME

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
