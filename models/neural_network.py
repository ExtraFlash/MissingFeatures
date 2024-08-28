import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Union


class NeuralNetworkModel:
    def __init__(self, input_size=784):
        # input size
        self.input_size = input_size
        # device: whether gpu or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # initialize the network and put into device
        self.network = NeuralNetwork(input_size=input_size, hidden_size=100).to(self.device)
        # define optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        # define loss functions
        self.bce_criterion = nn.BCELoss()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:

        self.network.train()

        # move data to tensors
        x = torch.from_numpy(x.to_numpy())
        y = torch.from_numpy(y.to_numpy())

        # define dataset and loader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(1000):
            for i, (x, y) in enumerate(dataloader):
                # move batch to device
                x, y = x.float().to(self.device), y.float().unsqueeze(1).to(self.device)  # y is transformed to (batch_size, 1)

                # forward step
                output = self.network(x)
                # backward step
                self.optimizer.zero_grad()
                loss = self.bce_criterion(output, y)
                loss.backward()
                self.optimizer.step()

                if epoch % 10 == 0 and i == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

    def predict(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            output = self.network(x)
            predicted = output > 0.5
            predicted = predicted.cpu().detach().numpy().astype(int)

        return predicted

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            output = self.network(x)
            p = output.cpu().detach().numpy()
            proba = np.concatenate([1 - p, p], axis=1)

        return proba



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers=2, num_classes=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out