import random

import pandas
import pandas as pd
from sklearn.linear_model import LogisticRegression

# data = {
#     'first': [1, 2, 3],
#     'second': [4, 5, 6],
#     'third': [0, 1, 2]
# }
#
# data = pandas.DataFrame(data)
# print(data)
# print('-------------')
# data = data.iloc[:, :-1]
# print(data)

# print(df)
# df.iloc[:, 1] = 0
# print(df)

# logistic_reg = LogisticRegression()
#
# X = data[['first', 'second']]
# y = data[['third']]
# logistic_reg.fit(X, y)
#
# y_pred = logistic_reg.predict_proba(X)
# print(y)
# print(type(y))
# print(y_pred)
# print(type(y_pred))
# print(y_pred.shape)

import torch

x = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
mask = torch.tensor([1,0,1])
print(f"x: {x}")
print(f"x shape: {x.shape}")
print(f"mask: {mask}")
print(f"mask shape: {mask.shape}")
print(f"x * mask: {x * mask}")

