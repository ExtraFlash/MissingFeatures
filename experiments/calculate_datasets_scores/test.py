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


if __name__ == "__main__":
    config_path = "../../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
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

        # Load data
        data_path = "../../data"
        train_set = pd.read_csv(f"{data_path}/{dataset_name_}/train/data.csv")

        print(train_set.shape)