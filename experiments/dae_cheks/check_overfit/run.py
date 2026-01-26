import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from my_models import ModelFactory
from my_models import ActivationFactory

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

from utils import utils


def save_results_for_dataset(dataset_name: str, is_multy_class: bool):
    # Load data
    data_path = "../../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    # val = pd.read_csv(f"{data_path}/{dataset_name}/val/data.csv")
    # test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")

    # Split to train and val
    X_train, y_train, X_val, y_val = utils.preprocess_split(train_set)

    model_name = ModelFactory.DAE_NAME

    best_params_path = f"../../../optimized_models/{dataset_name}/{model_name}/best_params.json"
    with open(best_params_path, "r") as file:
        best_params = json.load(file)

    latent_dim = best_params["latent_dim"]
    encoder_units = (best_params["encoder_units_0"], best_params["encoder_units_1"])
    decoder_units = (best_params["decoder_units_0"], best_params["decoder_units_1"])
    activation_name = ActivationFactory.leaky_relu_NAME
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]


    input_size = X_train.shape[1]
    model, loaded = ModelFactory.get_model(model_name, input_size,
                                           dataset_name=dataset_name,
                                           latent_dim=latent_dim,
                                           encoder_units=encoder_units,
                                           decoder_units=decoder_units,
                                           activation_name=activation_name,
                                           dropout_rate=dropout_rate,
                                           learning_rate=learning_rate,
                                           )

    train_results = model.fit(X_train, y_train, X_val, y_val, show_progress=True)

    train_loss = train_results['train_loss']
    train_reconstruct_loss = train_results['train_reconstruct_loss']
    train_mlp_loss = train_results['train_mlp_loss']

    val_loss = train_results['val_loss']
    val_reconstruct_loss = train_results['val_reconstruct_loss']
    val_mlp_loss = train_results['val_mlp_loss']

    y_val_probs = model.predict_proba(X_val)

    auc = roc_auc_score(y_val, y_val_probs[:, 1])
    print(f'{dataset_name} - {model_name} - auc: {auc}')

    # save plot
    # Create a figure with two subplots, one in each column
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: Reconstruct Loss
    train_steps = list(range(len(train_reconstruct_loss)))
    val_steps = list(range(len(val_reconstruct_loss)))
    ax1.plot(train_steps, train_reconstruct_loss, label='Training Loss')
    ax1.plot(val_steps, val_reconstruct_loss, label='Validation Loss')
    # ax1.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')  # Uncomment if needed
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Reconstruct Loss Over Time: {dataset_name}')
    ax1.legend()

    # Second subplot: MLP Loss
    ax2.plot(train_steps, train_mlp_loss, label='Training Loss')
    ax2.plot(val_steps, val_mlp_loss, label='Validation Loss')
    # ax2.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')  # Uncomment if needed
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'MLP Loss Over Time: {dataset_name}')
    ax2.legend()

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_combined_loss_plot.png")



if __name__ == "__main__":
    config_path = "../../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
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

        save_results_for_dataset(dataset_name_, is_multy_class_)
