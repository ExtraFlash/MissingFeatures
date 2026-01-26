"""
TODO: Implement general optimization functions for my_models
"""

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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from utils import utils

import ast


def objective(trial, X_train, y_train, X_val, y_val,
              model_name: str,
              hyperparameters_dict: dict):
    hypers_values_dict = {}
    for hyper_name, hyper_spec in hyperparameters_dict.items():
        type_ = hyper_spec['type']
        distribution = hyper_spec.get('distribution', None)
        should_log = distribution == 'loguniform'

        if type_ == 'int':
            # Integer sampling
            hypers_values_dict[hyper_name] = trial.suggest_int(
                hyper_name, int(hyper_spec['min']), int(hyper_spec['max']), log=should_log
            )
        elif type_ == 'float':
            # Float sampling
            hypers_values_dict[hyper_name] = trial.suggest_float(
                hyper_name, hyper_spec['min'], hyper_spec['max'], log=should_log
            )
        elif type_ == 'categorical':
            selected_value = trial.suggest_categorical(hyper_name, hyper_spec['values'])

            # Check if the selected value is a string that represents a list
            if isinstance(selected_value, str) and selected_value.startswith('[') and selected_value.endswith(']'):
                hypers_values_dict[hyper_name] = ast.literal_eval(selected_value)
            else:
                # For simple categorical values (int, float, string), use as-is
                hypers_values_dict[hyper_name] = selected_value
        else:
            raise ValueError(f"Unsupported type '{type_}' for hyperparameter '{hyper_name}'")

    model, _ = ModelFactory.get_model(model_name,
                                      input_size=X_train.shape[1],
                                      **hypers_values_dict)

    if ModelFactory.is_train_with_val(model_name):
        model.fit(X_train, y_train, X_val, y_val)
    elif ModelFactory.is_eval_set_format(model_name):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)

    y_val_probs = model.predict_proba(X_val)
    auc = roc_auc_score(y_val, y_val_probs[:, 1])

    return auc


def optimize_model_for_dataset(dataset_name: str, model_name: str):
    # Load data
    data_path = "../data"
    path = f"{data_path}/{dataset_name}"
    train = pd.read_csv(f"{path}/train/data.csv")

    X_train, y_train, X_val, y_val = utils.preprocess_split(train)

    # load config hyperparameters
    with open("config_hyperparameters.json") as f:
        config_hyperparameters = json.load(f)

    # Get the hyperparameters dict for the model
    hyperparameters_dict = config_hyperparameters[model_name]

    # Create a new function with X_train and y_train pre-filled
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                  model_name=model_name,
                                  hyperparameters_dict=hyperparameters_dict)

    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=100)

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
    # params_names = list(_best_params.keys())
    # fig_slice = optuna.visualization.plot_slice(study,
    #                                             params=params_names)
    # fig_slice.write_image(f"{dataset_name}/{model_name}/slice_plot.png")
    # fig_param_importances = optuna.visualization.plot_param_importances(study)
    # fig_param_importances.write_image(f"{dataset_name}/{model_name}/param_importances_plot.png")

    # model, _ = ModelFactory.get_model(model_name,
    #                                   input_size=X_train.shape[1],
    #                                   **_best_params)
    #
    # # Train model
    # if ModelFactory.is_lightning_model(model_name):
    #     model.fit(x=X_train, y=y_train, x_val=X_val, y_val=y_val, show_progress=False)
    #     # Save the model
    #     model.save_checkpoint(f"{dataset_name}/{model_name}/{model_name}.ckpt")
    # else:
    #     model.fit(X_train, y_train)
    #     # Save the model
    #     with open(f"{dataset_name}/{model_name}/model.pkl", 'wb') as f:
    #         pickle.dump(model, f)


def run_experiment(dataset, model_name):
    dataset_name = dataset['name']
    optimize_model_for_dataset(dataset_name, model_name)



if __name__ == "__main__":
    config_path = "../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)

    """
    Only for non-DAE my_models (where the size of one hyperparameter is independent of the other)
    """
    # models_names = [
    #     # ModelFactory.GCN_NAME,
    #     # ModelFactory.Denoising_Graph_Encoder_Name
    #     # ModelFactory.Gradient_Boosting_Classifier_Name,
    #     # ModelFactory.Neural_Network_NAME,
    #     # ModelFactory.Teacher_Students_NAME,
    #     # ModelFactory.Random_Forest_NAME,
    #     # ModelFactory.Ada_Boost_NAME,
    #     # ModelFactory.LGBM_NAME,
    #     # ModelFactory.XGB_NAME,
    #     # ModelFactory.Denoising_Graph_Encoder_Name,
    #     # ModelFactory.Logistic_Regression_NAME
    # ]

    models_names = [ModelFactory.G2_GRAPH_SAGE_NAME]

    models_to_not_optimize = [
        ModelFactory.Complete_Random_Forest_NAME,
        ModelFactory.Complete_Gradient_Boosting_Classifier_Name,
        ModelFactory.Mean_Gradient_Boosting_Classifier_Name,
        ModelFactory.Weighted_Gradient_Boosting_Classifier_Name
    ]

    models_names = [model_name for model_name in models_names if model_name not in models_to_not_optimize]


    # Note: Teacher Students and rest of the my_models are optimized: only optimize gcn, logistic regression

    # get datasets from config
    datasets: list = config['datasets']

    # datasets_names = [
    #     'tic-tac-toe',
    #     'eeg-eye-state',
    #     # 'kc2',
    #     # 'spambase',
    #     # 'PhishingWebsites'
    # ]

    for model_name in models_names:
        for dataset in datasets:
            # if dataset['name'] not in datasets_names:
            #     continue
            print(f"Running experiment for dataset: {dataset['name']} and model: {model_name}")
            run_experiment(dataset, model_name)

    # Use ProcessPoolExecutor for parallelism
    # with ProcessPoolExecutor(max_workers=5) as executor:
    #     tasks = []
    #     for model_name in models_names:
    #         for dataset in datasets:
    #             tasks.append(executor.submit(run_experiment, dataset, model_name))
    #
    #     for future in tqdm(tasks):
    #         future.result()

    # datasets = [datasets[0]]
    # run experiment for each dataset
    # for model_name_ in tqdm(models_names):
        # for dataset in datasets:
            # dataset_name_ = dataset['name']
            # relative_path_ = dataset['relative_path']
            # label_position_ = dataset['label_position']
            # has_header_ = dataset['has_header']
            # has_id_ = dataset['has_id']
            # is_multy_class_ = dataset['is_multy_class']

            # optimize_model_for_dataset(dataset_name_, model_name_)
