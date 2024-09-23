import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Union
import os
import json
from tqdm import tqdm

from models import ActivationFactory

import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy, AUROC

from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import List, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from pytorch_lightning.loggers import WandbLogger

BATCH_SIZE = 64


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
        # print(f"x shape: {x.shape}")
        # print(f"mask shape: {mask.shape}")
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


class DynamicTypeDAELightning(pl.LightningModule):
    def __init__(self,
                 input_size=784,
                 latent_dim=None,
                 types_list=None,
                 encoder_units=(128, 64),  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units=(64, 128),
                 activation_name=ActivationFactory.relu_NAME,
                 dropout_rate=0.0,
                 learning_rate=0.001
                 ):
        super(DynamicTypeDAELightning, self).__init__()

        self.learning_rate = learning_rate
        self.input_size = input_size
        self.types_list = types_list

        self.mask = None

        # save hyperparameters
        self.save_hyperparameters()

        # network
        self.network = DynamicTypeDAENetwork(input_size=input_size,
                                             latent_dim=latent_dim,
                                             types_list=types_list,
                                             encoder_units=encoder_units,
                                             decoder_units=decoder_units,
                                             activation_name=activation_name,
                                             dropout_rate=dropout_rate)

        # losses
        self.mse_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()

        # metrics
        self.accuracy = Accuracy(task='binary', num_classes=1)
        self.auroc = AUROC(task='binary', num_classes=1)

    def on_train_epoch_start(self):
        """Generate the mask at the start of each training epoch."""
        self.mask = self.generate_mask()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        # Take the batch
        x, y = batch
        # Generate mask
        # If not in train, we should use the full mask
        if stage != 'train':
            mask_vector = np.ones(x.shape[1])
            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        # If we are in train and mask is not generated yet
        elif self.mask is None:
            self.mask = self.generate_mask()
            mask = self.mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        else:
            # If we are in train and mask is already generated
            mask = self.mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)

        # Forward pass
        reconstructed, p = self.network(x, mask)
        # Calculate loss
        loss = self.custom_loss(x, y, reconstructed, p)
        # Calculate metrics
        # Calc accuracy
        predicted = p > 0.5
        acc = self.accuracy(predicted, y)
        # Log
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        y_true = y.cpu().numpy()
        y_flat = y_true.flatten()
        if len(set(y_flat)) > 1:  # Check if there is more than one class present
            auc = self.auroc(p, y)
            self.log(f"{stage}_auc", auc, prog_bar=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate_mask(self):
        """
        Generate a bernouli mask which is a tensor of size (input_size)
        :return:
        """
        # Generate p
        # Generate a random tensor of size (input_size)
        # p = torch.rand(self.input_size)
        p = np.random.uniform(0.1, 0.9)
        p = torch.full((self.input_size,), p)
        # p = np.random.beta(2, 2)
        mask = torch.bernoulli(p)
        return mask

    def custom_loss(self, x: torch.Tensor, y: torch.Tensor, constructed: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Get the device from one of the input tensors
        device = x.device

        combined_loss = torch.tensor(0.0, device=device)

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


class DynamicTypeDAEModel:
    def __init__(self,
                 input_size=784,
                 latent_dim=None,
                 types_list=None,
                 encoder_units=(128, 64),  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units=(64, 128),
                 activation_name=ActivationFactory.relu_NAME,
                 dropout_rate=0.0,
                 learning_rate=0.001
                 ):

        # define lightning model
        self.model = DynamicTypeDAELightning(input_size=input_size,
                                             latent_dim=latent_dim,
                                             types_list=types_list,
                                             encoder_units=encoder_units,
                                             decoder_units=decoder_units,
                                             activation_name=activation_name,
                                             dropout_rate=dropout_rate,
                                             learning_rate=learning_rate)

        self.trainer = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, x_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
            x_test: pd.DataFrame = None, y_test: pd.DataFrame = None,
            show_progress=True):
        # Create the dataset
        dataset = TensorDataset(torch.tensor(x.to_numpy(), dtype=torch.float32),
                                torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Validation data
        if x_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.tensor(x_val.to_numpy(), dtype=torch.float32),
                                        torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1))
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        else:
            val_dataloader = None

        # Initialize loggers and callbacks
        logger = TensorBoardLogger("../test_new_models/logs", name="DAE_model")
        wandb_logger = WandbLogger(project='SomeRandomDAE')
        wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath='checkpoints',
                                              filename='best_model')

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=1000,
            logger=[logger, wandb_logger],
            callbacks=[early_stop_callback, checkpoint_callback],
            devices=1,
            enable_progress_bar=show_progress,
        )

        # Fit the model
        self.trainer.fit(self.model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)

        # Test the model if needed
        if x_test is not None and y_test is not None:
            test_dataset = TensorDataset(torch.tensor(x_test.to_numpy(), dtype=torch.float32),
                                         torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1))
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            self.trainer.test(self.model, dataloaders=test_dataloader)

    def predict(self, x: Union[pd.DataFrame, np.ndarray],
                mask_vector: np.ndarray = None):
        self.model.eval()
        # if mask_vector is not provided, use a mask of ones to include all features
        if mask_vector is None:
            mask_vector = np.ones(x.shape[1])

        if isinstance(x, pd.DataFrame):
            x = torch.from_numpy(x.to_numpy()).float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1)

        with torch.no_grad():
            _, p = self.model.network(x, mask)
            predicted = p > 0.5
            predicted = predicted.cpu().detach().numpy().astype(int)
            return predicted

    def predict_proba(self,
                      x: Union[pd.DataFrame, np.ndarray],
                      mask_vector: np.ndarray = None
                      ):
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = torch.from_numpy(x.to_numpy()).float()
            else:
                x = torch.from_numpy(x).float()

            # if mask_vector is not provided, use a mask of ones to include all features
            if mask_vector is None:
                mask_vector = np.ones(x.shape[1])

            mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1)
            constructed, p = self.model.network(x, mask)
            # TODO: check dimensions of p, check concatenation of 1-p and p
            # print(f"p shape: {p.shape}")
            p = p.cpu().detach().numpy()
            # print(f"p: {p}")
            proba = np.concatenate([1 - p, p], axis=1)

            # print(f"proba: {proba}")
            # print(f"proba shape: {proba.shape}")
            # print(f"Predicted: {predicted}")
        return proba

    def save_checkpoint(self, file_path: str):
        self.trainer.save_checkpoint(file_path)

    def load_checkpoint(self, file_path: str):
        self.model = DynamicTypeDAELightning.load_from_checkpoint(file_path)


if __name__ == '__main__':
    config_path = "../datasets/config.json"
    # get datasets config
    with open(config_path) as f:
        config = json.load(f)
    # get datasets from config
    datasets: list = config['datasets']
    # run experiment for each dataset
    for dataset in tqdm(datasets):
        dataset_name = dataset['name']
        relative_path_ = dataset['relative_path']
        label_position_ = dataset['label_position']
        has_header_ = dataset['has_header']
        has_id_ = dataset['has_id']
        is_multy_class_ = dataset['is_multy_class']
        types_list = dataset['types_list']

        # if dataset_name != "Heart Disease":
        #     continue

        if dataset_name != "Tokyo":
            continue

        # Load data
        data_path = "../data"
        train_set = pd.read_csv(f"{data_path}/{dataset_name}/train/data.csv")

        train, val = train_test_split(train_set, test_size=0.2)
        test = pd.read_csv(f"{data_path}/{dataset_name}/test/data.csv")


        non_categorical_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2]
        categorical_columns = [col for col in train.columns[:-1] if col not in non_categorical_columns]

        real_positive_columns = [col for col in train.columns[:-1] if
                                 train[col].nunique() > 2 and (train[col] > 0).all()]
        real_columns = [col for col in train.columns[:-1] if train[col].nunique() > 2 and not (train[col] > 0).all()]

        target_column = [train.columns[-1]]
        original_columns = train.columns

        scaler_pipeline = ColumnTransformer([
            ('minmax_scaler', MinMaxScaler(), real_positive_columns),
            ('standard_scaler', StandardScaler(), real_columns)
        ], remainder='passthrough')

        train = pd.DataFrame(scaler_pipeline.fit_transform(train),
                             columns=non_categorical_columns + categorical_columns + target_column)
        # keep the order of columns
        train = train[original_columns]

        val = pd.DataFrame(scaler_pipeline.transform(val),
                           columns=non_categorical_columns + categorical_columns + target_column)
        # keep the order of columns
        val = val[original_columns]

        test = pd.DataFrame(scaler_pipeline.transform(test),
                            columns=non_categorical_columns + categorical_columns + target_column)
        # keep the order of columns
        test = test[original_columns]

        # Split features and labels
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
        X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]
        X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

        input_size = X_train.shape[1]

        model = DynamicTypeDAEModel(input_size=input_size,
                                    types_list=types_list,
                                    latent_dim=50,
                                    encoder_units=(128, 64),
                                    decoder_units=(64, 128),
                                    activation_name=ActivationFactory.relu_NAME,
                                    dropout_rate=0.1,
                                    learning_rate=0.001)

        model.fit(X_train, y_train, X_val, y_val, X_test, y_test, show_progress=True)

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_proba[:, 1])

        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")

        p_values = y_test_proba[:, 1]
        plt.hist(p_values, bins=50)
        plt.show()
