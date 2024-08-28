import json
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# config_path = "../datasets/config.json"
#
# with open(config_path) as f:
#     config = json.load(f)
#
# datasets: list = config['datasets']
#
# dataset = datasets[-1]
#
# dataset_name = dataset['name']
# relative_path = dataset['relative_path']
# label_position = dataset['label_position']
# has_header = dataset['has_header']
# has_id = dataset['has_id']
# positive = dataset.get('positive', None)
# negative = dataset.get('negative', None)
#
# # Load data
# datasets_path = "../datasets"
#
# # Define a custom delimiter regex
# delimiter_pattern = '\s+'
#
# path = datasets_path + '/' + relative_path
# if not has_header:
#     data = pd.read_csv(path, header=None, sep=delimiter_pattern)
# else:
#     data = pd.read_csv(path, sep=delimiter_pattern)
#
# # Replace categories with numbers for the label
# if positive:  # if there is a positive and negative strings, replace them with 1 and 0
#     # categories = data.iloc[:, -1].unique()
#     print('nowww')
#     data.iloc[:, -1].replace([positive, negative], [1, 0], inplace=True)
# print(data.head())

# y_true = torch.tensor([[1], [2], [2]]).view(-1)
#
# output = torch.tensor([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]])
# y_pred = torch.argmax(output, dim=1)
# print(y_true)
# print(y_pred)
# # print(type(y_true == y_pred))
# correct_predictions = (y_pred == y_true)
# print(correct_predictions)




# df = pd.DataFrame(
#     {
#         'a': [1, 2, 3],
#         'b': [5, 5, 6],
#         'c': [7, 8, 9],
#     }
# )
#
# categorical_columns = [col for col in df.columns[:-1] if df[col].nunique() < 3]
# non_categorical_columns = [col for col in df.columns[:-1] if col not in categorical_columns]
# target_column = [df.columns[-1]]
#
# one_hot_pipeline = ColumnTransformer([
#     ('one_hot', OneHotEncoder(), categorical_columns)
# ], remainder='passthrough')
#
# df_prepared = pd.DataFrame(one_hot_pipeline.fit_transform(df), columns=one_hot_pipeline.get_feature_names_out())
#
#
#
#
#
# non_categorical_columns = [col for col in df_prepared.columns[:-1] if df_prepared[col].nunique() > 2]
#
# categorical_columns = [col for col in df_prepared.columns[:-1] if col not in non_categorical_columns]
# target_column = [df_prepared.columns[-1]]
#
# print(df_prepared)
# print(non_categorical_columns)
#
# scale_pipeline = ColumnTransformer([
#     ('scale', StandardScaler(), non_categorical_columns)
# ], remainder='passthrough')
#
# df_last = pd.DataFrame(scale_pipeline.fit_transform(df_prepared), columns=non_categorical_columns + categorical_columns + target_column)
# print(df_last)
# df_last = df_last[df_prepared.columns]
# print(df_last)

import torch
import torch.nn as nn

# Create an instance of the nn.Sigmoid module
# sigmoid = nn.Softmax(dim=1)
#
# # Create a sample input tensor
# input_tensor = torch.randn(3, 2)
#
# # Apply the sigmoid activation function
# output_tensor = sigmoid(input_tensor)
#
# loss = nn.CrossEntropyLoss()
# calculated_loss = loss(input_tensor, torch.tensor([0, 1, 0]))
# print(calculated_loss.item())
#
# print("Input Tensor:")
# print(input_tensor)
# print("\nOutput Tensor after applying nn.Sigmoid:")
# print(output_tensor)

def f():
    return 0, 1

a = f()
print(a)
print(type(a))



