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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from models import ActivationFactory

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
    # Determine the columns for preprocessing
    # non_categorical_columns = [col for col in X_train.columns if X_train[col].nunique() > 2]
    # categorical_columns = [col for col in X_train.columns if col not in non_categorical_columns]
    # original_columns = X_train.columns

    # TODO: perform scaling only on non one-hot encoded features
    non_categorical_columns = [col for col in X_train.columns if X_train[col].nunique() > 2]
    categorical_columns = [col for col in X_train.columns if col not in non_categorical_columns]

    real_positive_columns = [col for col in X_train.columns if X_train[col].nunique() > 2 and (X_train[col] > 0).all()]
    real_columns = [col for col in X_train.columns if X_train[col].nunique() > 2 and not (X_train[col] > 0).all()]

    original_columns = X_train.columns

    scaler_pipeline = ColumnTransformer([
        ('minmax_scaler', MinMaxScaler(), real_positive_columns),
        ('standard_scaler', StandardScaler(), real_columns)
    ], remainder='passthrough')

    X_train = pd.DataFrame(scaler_pipeline.fit_transform(X_train),
                           columns=non_categorical_columns + categorical_columns)
    # keep the order of columns
    X_train = X_train[original_columns]

    X_val = pd.DataFrame(scaler_pipeline.transform(X_val),
                         columns=non_categorical_columns + categorical_columns)

    # Hyperparameters to optimize
    latent_dim = trial.suggest_int('latent_dim', 10, 50)
    # latent_dim = dims[dataset_name]

    encoder_units = []
    previous_size = 256
    for i in range(2):
        current_size = trial.suggest_int(f'encoder_units_{i}', 64, previous_size)
        encoder_units.append(current_size)
        previous_size = current_size

    # Enforcing increasing sizes for decoder units
    decoder_units = []
    previous_size = 64
    for i in range(2):
        current_size = trial.suggest_int(f'decoder_units_{i}', previous_size, 256)
        decoder_units.append(current_size)
        previous_size = current_size

    # Convert to tuple
    encoder_units = tuple(encoder_units)
    decoder_units = tuple(decoder_units)

    # activation_name = trial.suggest_categorical('activation_name',
    #                                             [ActivationFactory.relu_NAME, ActivationFactory.leaky_relu_NAME,
    #                                              ActivationFactory.tanh_NAME])
    activation_name = ActivationFactory.leaky_relu_NAME
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = BATCH_SIZE
    n_epochs = trial.suggest_int('n_epochs', 200, 1500)

    dae, _ = ModelFactory.get_model(
        model_name=model_name,
        input_size=X_train.shape[1],
        types_list=types_list,
        latent_dim=latent_dim,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        activation_name=activation_name,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    dae.fit(X_train, y_train, X_val, y_val)

    # y_pred = dae.predict(X_val)
    # accuracy = (y_pred.flatten() == y_val).mean()

    y_val_probs = dae.predict_proba(X_val)
    auc = roc_auc_score(y_val, y_val_probs[:, 1])

    return auc


def optimize_model_for_dataset(dataset_name: str):
    # Load data
    data_path = "../data"
    train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    X, y = train.iloc[:, :-1], train.iloc[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Create a new function with X_train and y_train pre-filled
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, dataset_name=dataset_name)

    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=20)

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
                                                params=['latent_dim',
                                                        'encoder_units_0', 'encoder_units_1',
                                                        'decoder_units_0', 'decoder_units_1',
                                                        'dropout_rate', 'learning_rate',
                                                        'n_epochs'])
    fig_slice.write_image(f"{dataset_name}/{model_name}/slice_plot.png")

    fig_param_importances = optuna.visualization.plot_param_importances(study)
    fig_param_importances.write_image(f"{dataset_name}/{model_name}/param_importances_plot.png")

    best_encoder_units = (_best_params['encoder_units_0'], _best_params['encoder_units_1'])
    best_decoder_units = (_best_params['decoder_units_0'], _best_params['decoder_units_1'])

    # Run the model with the best hyperparameters
    dae, _ = ModelFactory.get_model(
        model_name=model_name,
        input_size=X_train.shape[1],
        types_list=types_list,
        latent_dim=_best_params['latent_dim'],
        encoder_units=best_encoder_units,
        decoder_units=best_decoder_units,
        activation_name=ActivationFactory.leaky_relu_NAME,
        dropout_rate=_best_params['dropout_rate'],
        learning_rate=_best_params['learning_rate'],
    )

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

    n_epochs = _best_params['n_epochs']
    batch_size = BATCH_SIZE

    dae.fit(X_train, y_train, X_val, y_val)

    # Save the model
    dae.save_checkpoint(f"{dataset_name}/{model_name}")


if __name__ == "__main__":
    config_path = "../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)

    model_name = ModelFactory.DAE_Dynamic_TYPE_LIGHTNING_NAME

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
        types_list = dataset['types_list']

        # if dataset_name_ != "Heart Disease":
        #     continue

        optimize_model_for_dataset(dataset_name_)
