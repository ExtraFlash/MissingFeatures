import numpy as np
import pandas as pd
import os
import pathlib
import json


import os
import pathlib
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

###########################################
# This script performs one-hot encoding without standardization so that we can apply z-score on each fold separately
###########################################

# Example mapping function to convert binary categories to 0 and 1


def binary_mapping(X):
    return np.where(X == X.min(), 0, 1)


def process_data(dataset_name: str, label_position: str, has_header: bool, has_id: bool,
                 positive: str, negative: str, sep):
    # Load data

    # Define a custom delimiter regex
    delimiter_pattern = '\s+' if sep is not None else ','

    path = f"{dataset_name}.csv"

    if not has_header:
        data = pd.read_csv(path, header=None, sep=delimiter_pattern)
    else:
        data = pd.read_csv(path, sep=delimiter_pattern)

    # Remove id column
    if has_id:
        data.drop(data.columns[0], axis=1, inplace=True)

    # shuffle
    data = shuffle(data, random_state=42)
    data.reset_index(drop=True, inplace=True)

    # if we have less than 4 features or less than 200 samples, don't use the dataset
    if (data.shape[1] - 1) < 4 or data.shape[0] < 200:
        print(f"Dataset {dataset_name} has less than 4 features or less than 200 samples")
        return

    # if we have more than 10k rows, take only 10k
    if data.shape[0] > 10000:
        data = data.iloc[:10000]

    # Rearrange columns
    if label_position == "start":
        cols = data.columns.tolist()
        cols = cols[1:] + [cols[0]]
        data = data[cols]

    # Replace categories with numbers for the label
    if positive:  # if there is a positive and negative strings, replace them with 1 and 0
        # categories = data.iloc[:, -1].unique()
        data.iloc[:, -1].replace([positive, negative], [1, 0], inplace=True)

    # If there is a feature with only one unique value, drop it
    for col in data.columns[:-1]:
        if data[col].nunique() == 1:
            data.drop(col, axis=1, inplace=True)

    # Define the custom binary transformer
    binary_transformer = FunctionTransformer(binary_mapping, validate=True)

    # Drop columns with only one unique value
    # columns_to_drop = [col for col in data.columns if data[col].nunique() == 1]
    # data = data.drop(columns=columns_to_drop)

    # One hot encode categorical columns, and transform binary columns to 0 and 1
    categorical_columns = [col for col in data.columns[:-1] if 2 < data[col].nunique() < 20]
    binary_columns = [col for col in data.columns[:-1] if data[col].nunique() == 2]

    categorical_amounts = [data[col].nunique() for col in data.columns[:-1] if 2 < data[col].nunique() < 20]

    # Create a list of tuples to specify the type of each column, one-hot are
    # placed at the beginning of the resulting transformed dataset:   (type, index, amount)
    types_list = []
    idx = 0

    # Add binary features to the types list (they will be handled first)
    for i, col in enumerate(binary_columns):
        types_list.append(('binary', idx))
        idx += 1

    # Add categorical features to the types list
    for i, col in enumerate(categorical_columns):
        types_list.append(('categorical', idx, categorical_amounts[i]))
        idx += categorical_amounts[i]

    one_hot_pipeline = ColumnTransformer([
        ('binary', binary_transformer, binary_columns),
        ('one_hot', OneHotEncoder(), categorical_columns)
    ], remainder='passthrough')

    # data_before = data.copy()

    # notice order of columns change (categorical columns first)
    data = pd.DataFrame(one_hot_pipeline.fit_transform(data))

    # real-positive features
    print('---------------')
    print(f'dataset_name: {dataset_name}')
    for i, col in enumerate(data.columns[:-1]):
        if data[col].nunique() > 2 and (data[col] >= 0).all():
            types_list.append(('real_positive', i, 0))
            print(f"real_positive: {col}")
    # for i, col in enumerate(data.columns[idx:-1]):
    #     types_list.append(('real_positive', i + idx, 0))


    # Split data into train, validation and test sets
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    # train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Standardize data
    # standard_scaler = StandardScaler()
    # data.iloc[:, :-1] = standard_scaler.fit_transform(data.iloc[:, :-1])

    # Create directories
    # data_path = "../data"
    # if not os.path.exists(f"{data_path}/{dataset_name}"):
    #     os.makedirs(f"{data_path}/{dataset_name}")
    # if not os.path.exists(f"{data_path}/{dataset_name}/train"):
    #     os.makedirs(f"{data_path}/{dataset_name}/train")
    # # if not os.path.exists(f"{data_path}/{dataset_name}/val"):
    # #     os.makedirs(f"{data_path}/{dataset_name}/val")
    # # if not os.path.exists(f"{data_path}/{dataset_name}/test"):
    # #     os.makedirs(f"{data_path}/{dataset_name}/test")
    #
    # # update config with the types list
    # config_path = f"../datasets/config.json"
    # with open(config_path) as f:
    #     config = json.load(f)
    # # update the types list in the config
    # for dataset in config['datasets']:
    #     if dataset['name'] == dataset_name:
    #         dataset['types_list'] = types_list
    #         break
    # # save the updated config
    # with open(config_path, 'w') as f:
    #     json.dump(config, f, indent=4)

    # Save data
    data.to_csv(f"{dataset_name}_preprocessed.csv", index=False)
    # val.to_csv(f"{data_path}/{dataset_name}/val/data.csv", index=False)
    # test.to_csv(f"{data_path}/{dataset_name}/test/data.csv", index=False)


dataset_name = "simulation"
label_position = "end"
has_header = True
has_id = False
positive = None
negative = None
sep = None

# create the dataset csv using pandas
n_samples = 20000
n_features = 9
data = np.random.rand(n_samples, n_features + 1)
data[:, -2] = 2.0
# Now make a categorical column that should stay as it is
data[:, -3] = np.random.randint(0, 30, n_samples)
# Now make a binary column that should be transformed
data[:, -4] = np.random.randint(0, 2, n_samples)
# Now make a categorical column that should be one-hot encoded
data[:, -5] = np.random.randint(0, 10, n_samples)
# Make the last column the label
data[:, -1] = np.random.randint(0, 2, n_samples)
data = pd.DataFrame(data)
data.to_csv(f"{dataset_name}.csv", index=False)


process_data(dataset_name, label_position, has_header, has_id, positive, negative, sep)


