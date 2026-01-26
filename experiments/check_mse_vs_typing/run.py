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

from utils import utils


def save_results_for_dataset(dataset_name: str, is_multy_class: bool, types_list, results, models_names):
    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")

    cvs = 5

    for cv in range(cvs):
        print(f"CV: {cv}")
        # Split to train and val
        X_train, y_train, X_val, y_val = utils.preprocess_split(train_set)

        # load dae with best hyperparameters
        dae_name = ModelFactory.DAE_NAME
        best_params_path = f"../../optimized_models/{dataset_name}/{dae_name}/best_params.json"
        with open(best_params_path) as f:
            best_params = json.load(f)
        latent_dim = best_params['latent_dim']
        encoder_units = (best_params['encoder_units_0'], best_params['encoder_units_1'])
        decoder_units = (best_params['decoder_units_0'], best_params['decoder_units_1'])
        dropout_rate = best_params['dropout_rate']
        learning_rate = best_params['learning_rate']
        dae_model, _ = ModelFactory.get_model(dae_name,
                                              input_size=X_train.shape[1],
                                              latent_dim=latent_dim,
                                              encoder_units=encoder_units,
                                              decoder_units=decoder_units,
                                              dropout_rate=dropout_rate,
                                              learning_rate=learning_rate)
        # train dae
        dae_model.fit(X_train, y_train, X_val, y_val)

        # empty list for results
        results_auc = []

        # get list of features
        features = list(X_train.columns.values)

        remaining_features_amount = len(features)

        while remaining_features_amount >= 1:
            # print(f"{len(features)}, model: {model_name}")
            # print(f"len: {len(features)}, X: {X_val}")
            # predict on validation and get score
            features_to_remove_amount = len(features) - remaining_features_amount

            auc_means = []

            for _ in range(10):
                # copy X_val so that the original won't get affected
                X_val_missing = X_val.copy()
                features_to_remove = random.sample(features, k=features_to_remove_amount)
                X_val_missing.loc[:, features_to_remove] = 0.0
                remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]
                mask_vector = np.array(remaining_features)
                y_val_probs = dae_model.predict_proba(X_val_missing, mask_vector)
                auc = roc_auc_score(y_val, y_val_probs[:, 1])
                auc_means.append(auc)

            results_auc.append(np.mean(auc_means))
            remaining_features_amount -= 1

        results[dataset_name][dae_name].append(np.array(results_auc))

        #################### DAE Dynamic Type Lightning ####################

        # load dae with best hyperparameters
        dae_typing_name = ModelFactory.DAE_Dynamic_TYPE_LIGHTNING_NAME
        best_params_path = f"../../optimized_models/{dataset_name}/{dae_typing_name}/best_params.json"
        with open(best_params_path) as f:
            best_params = json.load(f)
        latent_dim = best_params['latent_dim']
        encoder_units = (best_params['encoder_units_0'], best_params['encoder_units_1'])
        decoder_units = (best_params['decoder_units_0'], best_params['decoder_units_1'])
        dropout_rate = best_params['dropout_rate']
        learning_rate = best_params['learning_rate']
        dae_typing_model, _ = ModelFactory.get_model(dae_typing_name,
                                                     input_size=X_train.shape[1],
                                                     latent_dim=latent_dim,
                                                     encoder_units=encoder_units,
                                                     decoder_units=decoder_units,
                                                     dropout_rate=dropout_rate,
                                                     learning_rate=learning_rate,
                                                     types_list=types_list)
        # train dae
        dae_typing_model.fit(X_train, y_train, X_val, y_val)

        # empty list for results
        results_auc = []

        # get list of features
        features = list(X_train.columns.values)

        remaining_features_amount = len(features)

        while remaining_features_amount >= 1:
            # print(f"{len(features)}, model: {model_name}")
            # print(f"len: {len(features)}, X: {X_val}")
            # predict on validation and get score
            features_to_remove_amount = len(features) - remaining_features_amount

            auc_means = []

            for _ in range(10):
                # copy X_val so that the original won't get affected
                X_val_missing = X_val.copy()
                features_to_remove = random.sample(features, k=features_to_remove_amount)
                X_val_missing.loc[:, features_to_remove] = 0.0
                remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]
                mask_vector = np.array(remaining_features)
                y_val_probs = dae_typing_model.predict_proba(X_val_missing, mask_vector)
                auc = roc_auc_score(y_val, y_val_probs[:, 1])
                auc_means.append(auc)

            results_auc.append(np.mean(auc_means))
            remaining_features_amount -= 1

        results[dataset_name][dae_typing_name].append(np.array(results_auc))

    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    X_train, _, _, _ = utils.preprocess_split(train_set)
    features_amount = len(list(X_train.columns.values))
    features_amounts = list(reversed(list(range(1, features_amount + 1))))

    results_df = pd.DataFrame(index=features_amounts, columns=models_names)
    for model_name in models_names:
        auc_results = np.array(results[dataset_name][model_name])
        # save the results
        results_df.loc[:, model_name] = np.mean(auc_results, axis=0)
        results_df.loc[:, 'features_amounts'] = features_amounts

    if not os.path.exists(f"{dataset_name}"):
        os.makedirs(f"{dataset_name}")
    results_df.to_csv(f"{dataset_name}/results.csv", index=False)


def main():
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']

    dataset_names = [dataset['name'] for dataset in datasets]
    models_names = [ModelFactory.DAE_NAME, ModelFactory.DAE_Dynamic_TYPE_LIGHTNING_NAME]

    results_dict = {dataset_name: {model_name: [] for model_name in models_names} for dataset_name in dataset_names}

    # run experiment for each dataset
    for dataset in tqdm(datasets):
        dataset_name_ = dataset['name']
        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']
        types_list_ = dataset['types_list']

        # if dataset_name_ != "Heart Disease":
        #     continue

        save_results_for_dataset(dataset_name_, is_multy_class_, types_list_, results_dict, models_names)


if __name__ == '__main__':
    main()
