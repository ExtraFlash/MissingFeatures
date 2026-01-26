import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from my_models import ModelFactory, ActivationFactory

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

MODELS_NAMES = [
    ModelFactory.Random_Forest_NAME,
    ModelFactory.Ada_Boost_NAME,
    ModelFactory.XGB_NAME
]


def save_results_for_dataset(dataset_name: str, is_multy_class: bool, types_list,
                             results_with_dae_dict,
                             results_without_dae_dict,
                             results_only_dae_dict):
    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")

    # load dae
    dae_name = ModelFactory.DAE_NAME
    # dae_checkpoint_dir = f"../../optimized_models/{dataset_name}/{dae_name}/{dae_name}.ckpt"
    X_train, _, _, _ = utils.preprocess_split(train_set)
    input_size = X_train.shape[1]

    best_params_path = f"../../optimized_models/{dataset_name}/{dae_name}/best_params.json"
    with open(best_params_path, "r") as file:
        best_params = json.load(file)

    cvs = 5

    for cv in range(cvs):
        print(f"CV: {cv}")
        # Split to train and val
        X_train, y_train, X_val, y_val = utils.preprocess_split(train_set)

        # Transform data into latent space
        X_train_before_latent = X_train.copy()
        X_val_before_latent = X_val.copy()

        ############### ONLY DAE ################
        # load dae with best hyperparameters
        latent_dim = best_params['latent_dim']
        encoder_units = (best_params['encoder_units_0'], best_params['encoder_units_1'])
        decoder_units = (best_params['decoder_units_0'], best_params['decoder_units_1'])
        activation_name = ActivationFactory.leaky_relu_NAME
        dropout_rate = best_params['dropout_rate']
        learning_rate = best_params['learning_rate']

        input_size = X_train.shape[1]
        dae_model, loaded = ModelFactory.get_model(dae_name, input_size,
                                                   dataset_name=dataset_name,
                                                   latent_dim=latent_dim,
                                                   encoder_units=encoder_units,
                                                   decoder_units=decoder_units,
                                                   activation_name=activation_name,
                                                   dropout_rate=dropout_rate,
                                                   learning_rate=learning_rate,
                                                   )
        dae_model.fit(X_train_before_latent, y_train, X_val_before_latent, y_val)
        X_train = dae_model.reconstruct(X_train)
        X_val = dae_model.reconstruct(X_val)

        # Get result for each model
        for model_name in MODELS_NAMES:

            ############### WITH DAE ################
            model_with_dae, loaded = ModelFactory.get_model(model_name, input_size,
                                                            dataset_name=dataset_name)
            model_with_dae.fit(X_train, y_train)

            ############### WITHOUT DAE ################
            model_without_dae, loaded = ModelFactory.get_model(model_name, input_size,
                                                               dataset_name=dataset_name)
            model_without_dae.fit(X_train_before_latent, y_train)

            # empty list for results
            results_with_dae_auc = []
            results_without_dae_auc = []
            results_only_dae_auc = []

            # results[dataset_name][model_name].append([])
            # model_checkpoint_dir = f"../../optimized_models/{dataset_name}/{model_name}"

            # get list of features
            features = list(X_train_before_latent.columns.values)

            remaining_features_amount = len(features)

            while remaining_features_amount >= 1:
                # print(f"{len(features)}, model: {model_name}")
                # print(f"len: {len(features)}, X: {X_val}")
                # predict on validation and get score
                features_to_remove_amount = len(features) - remaining_features_amount

                auc_with_dae_means = []
                auc_without_dae_means = []
                auc_dae_only_means = []

                for _ in range(10):
                    # copy X_val so that the original won't get affected
                    X_val_missing = X_val_before_latent.copy()
                    # print(f'X_val_missing: {X_val_missing.shape}')
                    # print(f'remaining_features_amount: {remaining_features_amount}')
                    # print(f'val shape: {X_val.shape}')
                    # print(f'val missing shape: {X_val_missing.shape}')
                    # remove features
                    features_to_remove = random.sample(features, k=features_to_remove_amount)
                    # print(f'features_to_remove: {features_to_remove}')
                    X_val_missing.loc[:, features_to_remove] = 0.0
                    # print(f'val missing shape: {X_val_missing.shape}

                    remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]
                    # print("-" * 20)
                    mask_vector = np.array(remaining_features)

                    ############### WITH DAE ################
                    X_val_missing_latent = dae_model.reconstruct(X_val_missing, mask_vector)
                    y_val_predicted = model_with_dae.predict(X_val_missing_latent)
                    y_val_probs = model_with_dae.predict_proba(X_val_missing_latent)

                    auc = roc_auc_score(y_val, y_val_probs[:, 1])
                    auc_with_dae_means.append(auc)
                    #########################################

                    ############### WITHOUT DAE ################
                    y_val_probs = model_without_dae.predict_proba(X_val_missing)
                    auc = roc_auc_score(y_val, y_val_probs[:, 1])
                    auc_without_dae_means.append(auc)
                    #########################################

                    ############### ONLY DAE ################
                    y_val_probs = dae_model.predict_proba(X_val_missing, mask_vector)
                    auc = roc_auc_score(y_val, y_val_probs[:, 1])
                    auc_dae_only_means.append(auc)

                # append results
                results_with_dae_auc.append(np.mean(auc_with_dae_means))
                results_without_dae_auc.append(np.mean(auc_without_dae_means))
                results_only_dae_auc.append(np.mean(auc_dae_only_means))

                remaining_features_amount -= 1

            results_with_dae_dict[dataset_name][model_name].append(np.array(results_with_dae_auc))
            results_without_dae_dict[dataset_name][model_name].append(np.array(results_without_dae_auc))
            results_only_dae_dict[dataset_name][model_name].append(np.array(results_only_dae_auc))

    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    X_train, _, _, _ = utils.preprocess_split(train_set)
    features_amount = len(list(X_train.columns.values))
    features_amounts = list(reversed(list(range(1, features_amount + 1))))

    results_with_dae_df = pd.DataFrame(index=features_amounts, columns=MODELS_NAMES)
    results_without_dae_df = pd.DataFrame(index=features_amounts, columns=MODELS_NAMES)
    results_only_dae_df = pd.DataFrame(index=features_amounts, columns=MODELS_NAMES)
    names_to_save = ['results_with_dae', 'results_without_dae', 'results_only_dae']
    for i, (df, results) in enumerate(zip([results_with_dae_df, results_without_dae_df, results_only_dae_df], [results_with_dae_dict, results_without_dae_dict, results_only_dae_dict])):
        for model_name in MODELS_NAMES:
            auc_results = np.array(results[dataset_name][model_name])
            # save the results
            df.loc[:, model_name] = np.mean(auc_results, axis=0)
            df.loc[:, 'features_amounts'] = features_amounts

        if not os.path.exists(f"{dataset_name}"):
            os.makedirs(f"{dataset_name}")
        df.to_csv(f"{dataset_name}/{names_to_save[i]}.csv", index=False)


def main():
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']

    dataset_names = [dataset['name'] for dataset in datasets]

    results_with_dae_dict = {dataset_name: {model_name: [] for model_name in MODELS_NAMES} for dataset_name in
                             dataset_names}
    results_without_dae_dict = {dataset_name: {model_name: [] for model_name in MODELS_NAMES} for dataset_name in
                                dataset_names}
    results_only_dae_dict = {dataset_name: {model_name: [] for model_name in MODELS_NAMES} for dataset_name in
                             dataset_names}

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

        save_results_for_dataset(dataset_name_, is_multy_class_, types_list_,
                                 results_with_dae_dict,
                                 results_without_dae_dict,
                                 results_only_dae_dict)


if __name__ == '__main__':
    main()
