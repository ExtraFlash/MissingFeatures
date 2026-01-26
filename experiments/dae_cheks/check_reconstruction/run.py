import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from my_models import ModelFactory
from utils import utils

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


def get_results_for_dataset(dataset_name):
    # Load data
    data_path = "../../../data"
    train = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    # val = pd.read_csv(f"{data_path}/{dataset_name}/val/data.csv")
    # test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")

    # Split to train and val
    X_train, y_train, X_val, y_val = utils.preprocess_split(train)

    model_name = ModelFactory.DAE_NAME

    input_size = X_train.shape[1]

    checkpoint_dir = f"../../../optimized_models/{dataset_name}/{model_name}/{model_name}.ckpt"

    model, loaded = ModelFactory.get_model(model_name, input_size, checkpoint_dir=checkpoint_dir,
                                           dataset_name=dataset_name)

    # Fit the model

    # checkpoint_dir = f"../../../optimized_models/{dataset_name}/{dae_name}"
    # input_size = X_train.shape[1]
    # model, loaded = ModelFactory.get_model(dae_name, input_size, checkpoint_dir=checkpoint_dir)

    # get list of features
    features = list(X_train.columns.values)

    remaining_features_amount = 10

    features_to_remove_amount = len(features) - remaining_features_amount

    X_val_missing = X_val.copy()

    # remove features
    features_to_remove = random.sample(features, k=features_to_remove_amount)


    X_val_missing.loc[:, features_to_remove] = 0.0

    # remaining features is 1 if the feature is not removed, 0 otherwise
    remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]

    # Run model on val data
    X_val_missing_torch = torch.from_numpy(X_val_missing.to_numpy()).float()
    X_val_missing_torch = X_val_missing_torch.to(model.model.device)
    X_val_torch = torch.from_numpy(X_val.to_numpy()).float().to(model.model.device)

    mask_vector = np.array(remaining_features)
    mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(X_val_missing_torch.shape[0], -1).to(model.model.device)
    # X_val_constructed, _ = model.network(X_val_torch, mask)
    X_val_constructed = model.reconstruct(X_val_missing_torch, mask_vector)

    mse = nn.MSELoss()(X_val_torch, X_val_constructed)

    print('---------------------------------')
    print(f'{dataset_name}')

    print(f"mse: {mse}")

    # print one sample
    sample_index = 0
    sample = X_val_torch[sample_index].cpu().detach().numpy()
    sample_constructed = X_val_constructed[sample_index].cpu().detach().numpy()
    print(f"sample: {sample}")
    print(f"sample_constructed: {sample_constructed}")

    # get indices where sample is not zero
    sample_indices = np.where(sample != 0)[0]
    # put indices in sample_constructed
    sample_constructed = sample_constructed[sample_indices]

    print(f"sample at indices: {sample[sample_indices]}")
    print(f"sample_constructed at indices: {sample_constructed}")

    # Create a DataFrame with original, reconstructed, and difference values
    results_df = pd.DataFrame({
        'Features': sample_indices,
        'Original': sample[sample_indices],
        'Reconstructed': sample_constructed,
        'Difference': sample[sample_indices] - sample_constructed
    })

    # Save to a CSV file
    results_df.to_csv(f"{dataset_name}_reconstruction_sample_{sample_index}.csv", index=False)
    print(f"Results saved to {dataset_name}_reconstruction_sample_{sample_index}.csv")

    # y_val_predicted = model.predict(X_val_missing.values, mask_vector)
    # y_val_probs = model.predict_proba(X_val_missing.values, mask_vector)


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

        # # continue only for this data
        # if dataset_name_ != "Statlog (German Credit Data)":
        #     continue

        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']

        get_results_for_dataset(dataset_name_)
