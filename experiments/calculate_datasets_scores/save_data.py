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


# def plot_learning_curve(dataset_name, model_name, estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     plt.savefig(f"results_{dataset_name}_{model_name}.png")
#     return plt


def save_results_for_dataset(dataset_name: str, is_multy_class: bool, data_percentage: int):
    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    target_col = train_set.columns[-1]  # Get the last column name
    train_set = train_set.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(frac=data_percentage / 100.0, random_state=42)
    ).reset_index(drop=True)
    # val = pd.read_csv(f"{data_path}/{dataset_name}/val/data.csv")
    # test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")

    cvs = 5

    # Create directories
    os.makedirs(f"data_percentage_{data_percentage}/{dataset_name}", exist_ok=True)

    for model_name in ModelFactory.MODELS:
        os.makedirs(f"data_percentage_{data_percentage}/{dataset_name}/{model_name}", exist_ok=True)

        for cv in range(cvs):
            os.makedirs(f"data_percentage_{data_percentage}/{dataset_name}/{model_name}/cv_{cv}", exist_ok=True)

    for cv in range(cvs):
        # Split to train and val
        seed = ModelFactory.SEEDS[cv]
        X_train, y_train, X_val, y_val = utils.preprocess_split(train_set, seed)

        features_amount = len(list(X_train.columns.values))

        # 'auc': {'model1': [scores], 'model2': [scores],...}
        # results = {
        #     'auc': {},
        #     'accuracy': {}
        # }
        #
        # results_available = {
        #     'auc': {}
        # }
        # get all my_models names
        models_names = ModelFactory.MODELS
        # models_names = ModelFactory.MODELS
        # for each model get list of scores
        for model_name in models_names:
            print(f'dataset: {dataset_name}, Loading model {model_name}')
            auc_results, accuracy_results, results_available_auc = get_model_results(model_name, dataset_name, X_train,
                                                                                     y_train, X_val, y_val,
                                                                                     is_multy_class)
            # results['auc'][model_name] = auc_results
            # results['accuracy'][model_name] = accuracy_results
            # results_available['auc'][model_name] = results_available_auc

            model_results = {
                'num_features': {},
                'auc': {},
                # 'accuracy': {}
            }

            model_results_available = {
                'num_features': {},
                'auc': {},
                # 'accuracy': {}
            }

            model_results['num_features'] = list(reversed(list(range(1, features_amount + 1))))
            model_results['auc'] = auc_results

            metric_df = pd.DataFrame(model_results)
            # Save data as csv
            metric_df.to_csv(f"data_percentage_{data_percentage}/{dataset_name}/{model_name}/cv_{cv}/auc.csv", index=False)

            # max auc with filtered features amounts
            model_results_available['num_features'] = [num for num in reversed(range(1, features_amount + 1)) if num % 5 == 1]
            model_results_available['auc'] = results_available_auc

            metric_df_available = pd.DataFrame(model_results_available)
            # Save data as csv
            metric_df_available.to_csv(f"data_percentage_{data_percentage}/{dataset_name}/{model_name}/cv_{cv}/auc_available.csv", index=False)

        # Save results
        # features_amount = len(list(X_train.columns.values))
        # features_amounts = list(reversed(list(range(1, features_amount + 1))))

        # metrics = ['auc', 'accuracy']
        # for metric in metrics:
        #     # Create Dataframe
        #     data_dict = {
        #         'num_features': features_amounts
        #     }
        #     for model_name in models_names:
        #         if len(results[metric][model_name]) == 0:  # for example if metric is auc and dataset is multiclass
        #             continue
        #         data_dict[model_name] = results[metric][model_name]
        #     # print(data_dict)
        #     metric_df = pd.DataFrame(data_dict)
        #     # Create directory
        #     if not os.path.exists(f"{dataset_name}/cv_{cv}"):
        #         os.makedirs(f"{dataset_name}/cv_{cv}")

            # Save data as csv
        #     metric_df.to_csv(f"{dataset_name}/cv_{cv}/{metric}.csv", index=False)
        #
        # # auc max
        # filtered_features_amounts = [num for num in reversed(range(1, features_amount + 1)) if num % 5 == 1]
        # # Create Dataframe
        # data_dict = {
        #     'num_features': filtered_features_amounts
        # }
        # for model_name in models_names:
        #     if len(results_available['auc'][model_name]) == 0:  # for example if metric is auc and dataset is multiclass
        #         continue
        #     data_dict[model_name] = results_available['auc'][model_name]
        #
        # metric_df = pd.DataFrame(data_dict)
        #
        # # Save data as csv
        # metric_df.to_csv(f"{dataset_name}/cv_{cv}/auc_available.csv", index=False)


def get_model_results(model_name: str, dataset_name: str, X_train, y_train, X_val, y_val, is_multy_class):
    # print(f'X_train: {X_train.shape}')
    # print(f'y_train: {y_train.shape}')
    # print(f'X_val: {X_val.shape}')
    # print(f'y_val: {y_val.shape}')
    # get the model instance

    optimized_model_name = ModelFactory.get_optimized_model_name(model_name)

    checkpoint_dir = f"../../optimized_models/{dataset_name}/{optimized_model_name}"

    best_params_path = f"{checkpoint_dir}/best_params.json"
    with open(best_params_path) as f:
        best_params = json.load(f)
    if 'n_estimators' in best_params:
        best_params['n_estimators'] = min(best_params['n_estimators'], 100)
    model, loaded = ModelFactory.get_model(model_name, X_train.shape[1], dataset_name=dataset_name,
                                           types_list=types_list, **best_params)
    if ModelFactory.is_train_with_val(model_name):
        model.fit(X_train, y_train, X_val, y_val)
    elif ModelFactory.is_eval_set_format(model_name):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)

    # else:
    #     if ModelFactory.is_lightning_model(model_name):
    #         checkpoint_dir = checkpoint_dir + f"/{model_name}.ckpt"
    #
    #     input_size = X_train.shape[1]
    #     model, loaded = ModelFactory.get_model(model_name, input_size, checkpoint_dir=checkpoint_dir, dataset_name=dataset_name, types_list=types_list)
    #     # train the model if not loaded
    #     if not loaded:
    #         # if ModelFactory.is_lightning_model(model_name):
    #         #     model.fit(X_train, y_train, X_val, y_val)
    #         # else:
    #         # model.fit(X_train, y_train)
    #         raise Exception(f"Model {model_name} is not loaded")

    # get list of features
    features = list(X_train.columns.values)
    # print(f"features: {features}, model: {model_name}")

    results_auc = []
    results_accuracy = []

    results_available_auc = []

    remaining_features_amount = len(features)

    while remaining_features_amount >= 1:
        print(f'remaining features: {remaining_features_amount}')
        # print(f"{len(features)}, model: {model_name}")
        # print(f"len: {len(features)}, X: {X_val}")
        # predict on validation and get score

        features_to_remove_amount = len(features) - remaining_features_amount
        # K times: randomly remove features and get results
        auc_means = []
        accuracy_means = []
        for trial in range(10):
            # copy X_val so that the original won't get affected
            X_val_missing = X_val.copy()
            # print(f'X_val_missing: {X_val_missing.shape}')
            # print(f'remaining_features_amount: {remaining_features_amount}')
            # print(f'val shape: {X_val.shape}')
            # print(f'val missing shape: {X_val_missing.shape}')
            # remove features
            features_to_remove = random.sample(features, k=features_to_remove_amount)
            # print(f'features_to_remove: {features_to_remove}')
            # print(f'val missing shape: {X_val_missing.shape}')

            # TODO: support masked my_models, graph masked, and basic my_models that support None as mask
            """
            pipeline:
            if model is masked model:
            get prediction using X_val_missing and mask vector
            if model supports nones:
            set nones in X_val_missing and get predictions
            if model supports unmasked columns
            """

            if ModelFactory.is_masked_model(model_name):
                X_val_missing.loc[:, features_to_remove] = 0.0
                # remaining features is 1 if the feature is not removed, 0 otherwise
                remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]

                mask_vector = np.array(remaining_features)
                y_val_predicted = model.predict(X_val_missing.values, mask_vector)
                y_val_probs = model.predict_proba(X_val_missing.values, mask_vector)
            elif ModelFactory.is_model_supports_nans(model_name):
                # set missing features to Nan
                X_val_missing.loc[:, features_to_remove] = np.nan
                y_val_predicted = model.predict(X_val_missing)
                y_val_probs = model.predict_proba(X_val_missing)


            elif ModelFactory.is_model_supports_unmasked_columns(model_name):
                unmasked_columns = []
                for i, feature in enumerate(features):
                    if feature not in features_to_remove:
                        unmasked_columns.append(i)
                y_val_predicted = model.predict(X_val_missing, unmasked_columns)
                y_val_probs = model.predict_proba(X_val_missing, unmasked_columns)
            else:
                X_val_missing.loc[:, features_to_remove] = 0.0
                y_val_predicted = model.predict(X_val_missing)
                y_val_probs = model.predict_proba(X_val_missing)

            # # get predictions, if DAE model, need to get also the mask vector of missing features
            # if not ModelFactory.is_masked_model(model_name):
            #     y_val_predicted = model.predict(X_val_missing)
            #     y_val_probs = model.predict_proba(X_val_missing)
            # else:
            #     # remaining features is 1 if the feature is not removed, 0 otherwise
            #     remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]
            #
            #     # print("-" * 20)
            #     mask_vector = np.array(remaining_features)
            #     y_val_predicted = model.predict(X_val_missing.values, mask_vector)
            #     y_val_probs = model.predict_proba(X_val_missing.values, mask_vector)

            if not is_multy_class:
                auc = roc_auc_score(y_val, y_val_probs[:, 1])
                auc_means.append(auc)

            accuracy = accuracy_score(y_val, y_val_predicted)
            accuracy_means.append(accuracy)

            # print(f"i: {trial}, remaining features: {remaining_features_amount}")
            # print(f"i==0: {trial==0}, remaining features % 5 == 1: {(remaining_features_amount % 5 == 1)}")
            # print('-----')

            # addition of auc max
            if (trial == 0) and (remaining_features_amount % 5 == 1):
                # get the remaining features
                remaining_features = [feature for feature in features if feature not in features_to_remove]

                # train only on the remaining features
                model_max, loaded_max = ModelFactory.get_model(model_name, len(remaining_features),
                                                               dataset_name=dataset_name,
                                                               types_list=types_list, **best_params)

                X_train_remaining = X_train[remaining_features]
                X_val_remaining = X_val[remaining_features]

                if ModelFactory.is_train_with_val(model_name):
                    model_max.fit(X_train_remaining, y_train, X_val_remaining, y_val)
                elif ModelFactory.is_eval_set_format(model_name):
                    model_max.fit(X_train_remaining, y_train, eval_set=[(X_val_remaining, y_val)])
                else:
                    model_max.fit(X_train_remaining, y_train)

                # get auc result
                y_val_probs = model_max.predict_proba(X_val_remaining)

                auc_remaining = roc_auc_score(y_val, y_val_probs[:, 1])

                results_available_auc.append(auc_remaining)

        auc = np.mean(auc_means)
        results_auc.append(auc)
        accuracy = np.mean(accuracy_means)
        results_accuracy.append(accuracy)

        remaining_features_amount -= 1
    return results_auc, results_accuracy, results_available_auc


if __name__ == "__main__":
    config_path = "../../datasets/config.json"

    data_percentages = [
        100, 75, 50
    ]

    for data_percentage in data_percentages:
        os.makedirs(f"data_percentage_{data_percentage}", exist_ok=True)

    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets'][8:]
    #  datasets_names: list = [
    #      "Tokyo",
    #      "Connectionist Bench",
    #      "Ionosphere",
    #      "Pima Indians Diabetes Database",
    #     "Heart Disease",
    #       "Statlog (German Credit Data)",
    #   ]
    # get only the datasets that are in the list
    # datasets = [dataset for dataset in datasets if dataset['name'] in datasets_names]
    # run experiment for each dataset
    for dataset in tqdm(datasets):
        # if dataset['name'] not in ["electricity"]:
        #     print(f'skipping {dataset["name"]}')
        #     continue
        print(f'current dataset: {dataset["name"]}')
        dataset_name_ = dataset['name']
        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']
        types_list = dataset.get('types_list', None)

        # if dataset_name_ != "Heart Disease":
        #     continue

        for data_percentage in data_percentages:
            save_results_for_dataset(dataset_name_, is_multy_class_, data_percentage)
