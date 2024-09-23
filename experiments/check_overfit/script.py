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


def save_results_for_dataset(dataset_name: str, is_multy_class: bool):
    # Load data
    data_path = "../../data"
    train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")
    # val = pd.read_csv(f"{data_path}/{dataset_name}/val/data.csv")
    # test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")

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

    # Split features and labels
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]

    model_name = ModelFactory.DAE_Dynamic_TYPE_NAME

    input_size = X_train.shape[1]
    model, loaded = ModelFactory.get_model(model_name, input_size,
                                           dataset_name=dataset_name, types_list=types_list,
                                           latent_dim=49,
                                           encoder_units=(128, 64),
                                           dropout_rate=0.5,
                                           learning_rate=1e-3,
                                           )

    train_loss_history, val_loss_history = model.fit(X_train, y_train, X_val, y_val, batch_size=32, show_progress=True)

    y_val_probs = model.predict(X_val)

    auc = roc_auc_score(y_val, y_val_probs)
    print(f'{dataset_name} - {model_name} - auc: {auc}')

    # save plot
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='val')
    plt.legend()
    plt.title(f"{dataset_name} - {model_name}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("loss_plot.png")


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

        if dataset_name_ != "Heart Disease":
            continue

        save_results_for_dataset(dataset_name_, is_multy_class_)
