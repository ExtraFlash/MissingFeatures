import pandas as pd
import torch


# df = pd.read_csv("../datasets/Heart Disease/heart_statlog_cleveland_hungary_final.csv")
# print(df)

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mask = torch.tensor([1, 0, 1])
mask = mask.unsqueeze(0).expand(X.shape[0], -1)
mask = mask.unsqueeze(0).expand(X.shape[0], -1)
print(f"x: {X}")
print(f"x shape: {X.shape}")
print(f"mask: {mask}")
print(f"mask shape: {mask.shape}")

con = torch.cat([X, mask], dim=1)
print(f"concat: {con}")