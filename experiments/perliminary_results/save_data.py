import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from tqdm import tqdm

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


def save_results_for_dataset(dataset_name: str, is_multy_class: bool):
    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    # val = pd.read_csv(f"{data_path}/{dataset_name}/val/data.csv")
    # test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")

    cvs = 5

    for cv in range(cvs):
        # Split to train and val
        train, val = train_test_split(train_set, test_size=0.2)
        # Scale train and val
        # TODO: perform scaling only on non one-hot encoded features
        non_categorical_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2]
        categorical_columns = [col for col in train.columns[:-1] if col not in non_categorical_columns]


        real_positive_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2 and (train[col] > 0).all()]
        real_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2 and not (train[col] > 0).all()]

        target_column = [train.columns[-1]]
        original_columns = train.columns

        scaler_pipeline = ColumnTransformer([
            ('minmax_scaler', MinMaxScaler(), real_positive_columns),
            ('standard_scaler', StandardScaler(), real_columns)
        ], remainder='passthrough')

        train = pd.DataFrame(scaler_pipeline.fit_transform(train), columns=non_categorical_columns + categorical_columns + target_column)
        # keep the order of columns
        train = train[original_columns]

        val = pd.DataFrame(scaler_pipeline.transform(val), columns=non_categorical_columns + categorical_columns + target_column)
        # keep the order of columns
        val = val[original_columns]

        # train.iloc[:, :-1] = scaler_pipeline.fit_transform(train.iloc[:, :-1])
        # val.iloc[:, :-1] = scaler_pipeline.transform(val.iloc[:, :-1])


        # train.iloc[:, :-1] = standard_scaler.fit_transform(train.iloc[:, :-1])
        # val.iloc[:, :-1] = standard_scaler.transform(val.iloc[:, :-1])


        # Split features and labels
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
        X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]

        # 'auc': {'model1': [scores], 'model2': [scores],...}
        results = {
            'auc': {},
            'accuracy': {}
        }
        # get all models names
        models_names = ModelFactory.MODELS
        # for each model get list of scores
        for model_name in models_names:
            auc_results, accuracy_results = get_model_results(model_name, dataset_name, X_train, y_train, X_val, y_val,
                                                              is_multy_class)
            results['auc'][model_name] = auc_results
            results['accuracy'][model_name] = accuracy_results

        # Create directories
        if not os.path.exists(f"{dataset_name}"):
            os.makedirs(f"{dataset_name}")

        # Save results
        features_amount = len(list(X_train.columns.values))
        features_amounts = list(reversed(list(range(1, features_amount + 1))))

        metrics = ['auc', 'accuracy']
        for metric in metrics:
            # Create Dataframe
            data_dict = {
                'num_features': features_amounts
            }
            for model_name in models_names:
                if len(results[metric][model_name]) == 0:  # for example if metric is auc and dataset is multiclass
                    continue
                data_dict[model_name] = results[metric][model_name]
            # print(data_dict)
            metric_df = pd.DataFrame(data_dict)
            # Create directory
            if not os.path.exists(f"{dataset_name}/cv_{cv}"):
                os.makedirs(f"{dataset_name}/cv_{cv}")

            # Save data as csv
            metric_df.to_csv(f"{dataset_name}/cv_{cv}/{metric}.csv", index=False)


def get_model_results(model_name: str, dataset_name: str, X_train, y_train, X_val, y_val, is_multy_class):
    # print(f'X_train: {X_train.shape}')
    # print(f'y_train: {y_train.shape}')
    # print(f'X_val: {X_val.shape}')
    # print(f'y_val: {y_val.shape}')
    # get the model instance

    checkpoint_dir = f"../../optimized_models/{dataset_name}/{model_name}"

    input_size = X_train.shape[1]
    model, loaded = ModelFactory.get_model(model_name, input_size, checkpoint_dir=checkpoint_dir, dataset_name=dataset_name, types_list=types_list)
    # train the model if not loaded
    if not loaded:
        if ModelFactory.is_lightning_model(model_name):
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)

    # get list of features
    features = list(X_train.columns.values)
    # print(f"features: {features}, model: {model_name}")

    results_auc = []
    results_accuracy = []

    remaining_features_amount = len(features)

    while remaining_features_amount >= 1:
        # print(f"{len(features)}, model: {model_name}")
        # print(f"len: {len(features)}, X: {X_val}")
        # predict on validation and get score

        #
        features_to_remove_amount = len(features) - remaining_features_amount
        # K times: randomly remove features and get results
        auc_means = []
        accuracy_means = []
        for _ in range(10):
            # copy X_val so that the original won't get affected
            X_val_missing = X_val.copy()
            # print(f'X_val_missing: {X_val_missing.shape}')
            # print(f'remaining_features_amount: {remaining_features_amount}')
            # print(f'val shape: {X_val.shape}')
            # print(f'val missing shape: {X_val_missing.shape}')
            # remove features
            features_to_remove = random.sample(features, k=features_to_remove_amount)
            # print(f'features_to_remove: {features_to_remove}')
            X_val_missing.loc[:, features_to_remove] = 0.0
            # print(f'val missing shape: {X_val_missing.shape}')

            # get predictions, if DAE model, need to get also the mask vector of missing features
            if not ModelFactory.is_masked_model(model_name):
                y_val_predicted = model.predict(X_val_missing)
                y_val_probs = model.predict_proba(X_val_missing)
            else:
                # remaining features is 1 if the feature is not removed, 0 otherwise
                remaining_features = [1 if feature not in features_to_remove else 0 for feature in features]

                # print("-" * 20)
                mask_vector = np.array(remaining_features)
                y_val_predicted = model.predict(X_val_missing.values, mask_vector)
                y_val_probs = model.predict_proba(X_val_missing.values, mask_vector)

            if not is_multy_class:
                auc = roc_auc_score(y_val, y_val_probs[:, 1])
                auc_means.append(auc)

            accuracy = accuracy_score(y_val, y_val_predicted)
            accuracy_means.append(accuracy)

        auc = np.mean(auc_means)
        results_auc.append(auc)
        accuracy = np.mean(accuracy_means)
        results_accuracy.append(accuracy)

        remaining_features_amount -= 1
    return results_auc, results_accuracy


if __name__ == "__main__":
    config_path = "../../datasets/config.json"
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
