import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Union
import os

from models import ActivationFactory


# TODO:
# 1) In the preprocessing step, find the categorical features, real features, positive real features
#    and save their indices (also with one-hot for the categorical features)
# 2) Save the indices in a json file for each preprocessed dataset
# 3) Load each dataset in run.py and the json, and only for this dynamic model, input the indices
# 4) Define this model to get the indices and build the architecture accordingly


class DynamicTypeDAE:
    def __init__(self,
                 input_size=784,
                 latent_dim=None,
                 types_list=None,
                 encoder_units=(128, 64),  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units=(64, 128),
                 activation_name=ActivationFactory.relu_NAME,
                 dropout_rate=0.0,
                 learning_rate=1e-3
                 ):
        # input size
        self.input_size = input_size

        # latent dimension
        self.latent_dim = latent_dim

        # types list
        self.types_list = types_list

        # device: whether gpu or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # initialize the network and put into device
        self.network = DynamicTypeDAENetwork(input_size=input_size,
                                             latent_dim=latent_dim,
                                             types_list=types_list,
                                             encoder_units=encoder_units,
                                             decoder_units=decoder_units,
                                             activation_name=activation_name,
                                             dropout_rate=dropout_rate).to(self.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # define losses
        self.mse_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, x_val: pd.DataFrame=None, y_val: pd.DataFrame=None, n_epochs=1000, batch_size=32, show_progress=True):
        """
        Train the model and in each epoch generate a bernouli mask
        :param x:
        :param y:
        :param x_val:
        :param y_val:
        :param n_epochs:
        :param batch_size:
        :param show_progress:
        """

        self.network.train()

        x = torch.from_numpy(x.to_numpy())
        y = torch.from_numpy(y.to_numpy())

        if x_val is not None and y_val is not None:
            x_val = torch.from_numpy(x_val.to_numpy())
            y_val = torch.from_numpy(y_val.to_numpy())
            x_val, y_val = x_val.float().to(self.device), y_val.float().unsqueeze(1).to(self.device)

        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_loss_history = []
        val_loss_history = []

        # Train the model
        for epoch in range(n_epochs):
            # Generate a bernouli mask
            mask_first_epoch = self.generate_mask()
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            for i, (x, y) in enumerate(dataloader):
                # take the mask of size f, add 0 dimension (1, f), expand it to (batch_size, f) and move to device
                mask = mask_first_epoch.unsqueeze(0).expand(x.shape[0], -1).to(self.device)
                # move batch to device
                x, y = x.float().to(self.device), y.float().unsqueeze(1).to(
                    self.device)  # y is transformed to (batch_size, 1)
                # forward step
                constructed, p = self.network(x, mask)
                # backward step
                self.optimizer.zero_grad()
                loss = self.custom_loss(x, y, constructed, p)
                loss.backward()
                self.optimizer.step()

                if show_progress and epoch % 10 == 0 and i == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                epoch_train_loss += loss.item()
                if x_val is not None and y_val is not None:
                    with torch.no_grad():
                        mask = mask_first_epoch.unsqueeze(0).expand(x_val.shape[0], -1).to(self.device)
                        val_loss = self.custom_loss(x_val, y_val, self.network(x_val, mask)[0], self.network(x_val, mask)[1])
                        epoch_val_loss += val_loss.item()
            train_loss_history.append(epoch_train_loss)
            val_loss_history.append(epoch_val_loss)

        if x_val is not None and y_val is not None:
            return train_loss_history, val_loss_history

    def custom_loss(self, x: torch.Tensor, y: torch.Tensor, constructed: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        combined_loss = torch.tensor(data=0., device=self.device)

        first_non_categorical_index = 0

        for tup in self.types_list:
            if tup[0] == 'binary':
                # print(f"tup[1]: {tup[1]}")
                # print(constructed[:, tup[1]])
                # print(f'shape: {constructed[:, tup[1]].shape}')
                # print(x[:, tup[1]])
                # print(f'shape: {x[:, tup[1]].shape}')
                combined_loss = combined_loss + self.bce_criterion(
                    constructed[:, tup[1]],
                    x[:, tup[1]]
                )

            elif tup[0] == 'categorical':  # cross entropy loss takes the logits, therefore we do not perform softmax, only in the predict method
                combined_loss = combined_loss + self.ce_criterion(
                    constructed[:, tup[1]: tup[1] + tup[2]],
                    x[:, tup[1]: tup[1] + tup[2]]
                )
            if tup[0] in {'real_positive', 'real'}:
                first_non_categorical_index = tup[1]
                break


        # if first_non_categorical_index is still 0, then we take the whole x
        combined_loss = combined_loss + self.mse_criterion(constructed[:, first_non_categorical_index:], x[:, first_non_categorical_index:])

        combined_loss = combined_loss + self.bce_criterion(p, y)

        return combined_loss

    def predict(self, x: Union[pd.DataFrame, np.ndarray],
                mask_vector: np.ndarray = None):
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            # if mask_vector is not provided, use a mask of ones to include all features
            if mask_vector is None:
                mask_vector = np.ones(x.shape[1])

            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(self.device)
            constructed, p = self.network(x, mask)
            predicted = p > 0.5
            predicted = predicted.cpu().detach().numpy().astype(int)
            # print(f"Predicted: {predicted}")
        return predicted

    def reconstruct(self, x: Union[pd.DataFrame, np.ndarray, torch.Tensor],
                    mask_vector: np.ndarray):
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            elif isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            x = x.to(self.device)
            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(self.device)
            constructed, p = self.network(x, mask)
            reconstructed = constructed.detach()

            # TODO: check if the following is correct
            for tup in self.types_list:
                if tup[0] == 'binary':
                    reconstructed[:, tup[1]] = reconstructed[:, tup[1]] > 0.5
                elif tup[0] == 'categorical':
                    reconstructed[:, tup[1]: tup[1] + tup[2]] = torch.argmax(reconstructed[:, tup[1]: tup[1] + tup[2]], dim=1)
        return reconstructed

    def predict_proba(self,
                      x: Union[pd.DataFrame, np.ndarray],
                      mask_vector: np.ndarray = None):
        self.network.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            # if mask_vector is not provided, use a mask of ones to include all features
            if mask_vector is None:
                mask_vector = np.ones(x.shape[1])

            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(self.device)
            constructed, p = self.network(x, mask)
            # TODO: check dimensions of p, check concatenation of 1-p and p
            # print(f"p shape: {p.shape}")
            p = p.cpu().detach().numpy()
            # print(f"p: {p}")
            proba = np.concatenate([1 - p, p], axis=1)

            # print(f"proba: {proba}")
            # print(f"proba shape: {proba.shape}")
            # print(f"Predicted: {predicted}")
        return proba

    def generate_mask(self):
        """
        Generate a bernouli mask which is a tensor of size (input_size)
        :return:
        """
        # Generate p
        # Generate a random tensor of size (input_size)
        # p = torch.rand(self.input_size)
        p = np.random.uniform(0.1, 0.9)
        p = torch.full((self.input_size, ), p)
        # p = np.random.beta(2, 2)
        mask = torch.bernoulli(p)
        return mask

    def save_checkpoint(self, checkpoint_dir):
        state = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'model' + ".pth"))

    def load_checkpoint(self, checkpoint_dir):
        loaded = False
        if os.path.exists(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
            if self.device.type == 'cpu':
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(checkpoint_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded = True
        # n_training_seqs = checkpoint['n_training_seqs']
        # loss_history = checkpoint['loss_history']

        return loaded


class DynamicTypeDAENetwork(nn.Module):
    def __init__(self,
                 input_size=784,
                 latent_dim=None,
                 types_list=None,
                 encoder_units=(128, 64),  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units=(64, 128),
                 activation_name=ActivationFactory.relu_NAME,
                 dropout_rate=0.0
                 ):
        super(DynamicTypeDAENetwork, self).__init__()

        if latent_dim is None:
            self.latent_dim = 3
        else:
            self.latent_dim = latent_dim
        latent_dim = self.latent_dim

        # types list
        self.types_list = types_list

        # Encoder
        encoder_layers = nn.ModuleList()
        current_size = input_size * 2
        for units in encoder_units:
            encoder_layers.append(nn.Linear(current_size, units))
            activation = ActivationFactory.get_activation(activation_name)
            encoder_layers.append(activation())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            current_size = units
        encoder_layers.append(nn.Linear(current_size, self.latent_dim))
        encoder_layers.append(activation())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = nn.ModuleList()
        current_size = latent_dim
        for units in decoder_units:
            decoder_layers.append(nn.Linear(current_size, units))
            decoder_layers.append(activation())
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_size = units
        decoder_layers.append(nn.Linear(current_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask):  # assume x is a tensor of size (batch_size, 784), mask is (784)
        # apply mask
        x = x * mask

        # concatenate the masked input with the mask
        concat = torch.cat([x, mask], 1)
        latent = self.encoder(concat)
        constructed = self.decoder(latent)

        # Create a copy to avoid in-place modification
        constructed_copy = constructed.clone()

        # construct according to the types
        for tup in self.types_list:
            if tup[0] == 'binary':
                constructed_copy[:, tup[1]] = torch.sigmoid(constructed[:, tup[1]])
            elif tup[0] == 'real_positive':
                # perform softplus
                constructed_copy[:, tup[1]] = torch.nn.functional.softplus(constructed[:, tup[1]])

        # mlp
        p = self.mlp(latent)
        return constructed_copy, p
