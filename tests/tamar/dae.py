import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Union
from typing import Tuple
import os
import json
from tqdm import tqdm

from my_models import ActivationFactory

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
from typing import Tuple

BATCH_SIZE = 64


class DAE_Network(nn.Module):
    def __init__(self,
                 input_size,
                 latent_dim,
                 encoder_units: Tuple[int, int],
                 decoder_units: Tuple[int, int],
                 dropout_rate,
                 activation_name=ActivationFactory.relu_NAME,
                 should_support_missing_values: bool = False):

        super(DAE_Network, self).__init__()

        self.latent_dim = latent_dim

        self.should_support_missing_values = should_support_missing_values

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

    def forward(self, x, mask, mask_for_nones):  # assume x is a tensor of size (batch_size, 784), mask is (784)

        if self.should_support_missing_values:
            # Replace NaNs with 0 before applying the mask
            x_without_nans = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            # apply mask
            x = x_without_nans * mask
            # concatenate the masked input with the mask for nones
            x = torch.cat((x, mask_for_nones * mask), dim=1)
            # latent
            latent = self.encoder(x)
            # reconstructed
            constructed = self.decoder(latent)
        else:
            # apply mask
            x = x * mask

            # concatenate the masked input with the mask
            # print(f"x: {x.shape}, mask: {mask.shape}")
            concat = torch.cat([x, mask], 1)
            # print(f"concat: {concat.shape}")
            latent = self.encoder(concat)
            constructed = self.decoder(latent)

        # mlp
        p = self.mlp(latent)
        return constructed, p, latent


class DAELightning(pl.LightningModule):
    def __init__(self,
                 input_size,
                 latent_dim,
                 encoder_units: Tuple[int, int],  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units: Tuple[int, int],
                 dropout_rate,
                 learning_rate,
                 activation_name=ActivationFactory.relu_NAME,
                 should_support_missing_values: bool = False
                 ):
        super(DAELightning, self).__init__()

        self.learning_rate = learning_rate
        self.input_size = input_size

        self.should_support_missing_values = should_support_missing_values

        self.mask = None

        if should_support_missing_values:
            self.mask_for_nones = None

        # save hyperparameters
        self.save_hyperparameters()

        # network
        self.network = DAE_Network(input_size=input_size,
                                   latent_dim=latent_dim,
                                   encoder_units=encoder_units,
                                   decoder_units=decoder_units,
                                   activation_name=activation_name,
                                   dropout_rate=dropout_rate,
                                   should_support_missing_values=should_support_missing_values)

        # losses
        self.mse_criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()

        # metrics
        self.accuracy = Accuracy(task='binary', num_classes=1)
        self.auroc = AUROC(task='binary', num_classes=1)

    def on_train_epoch_start(self):
        """Generate the mask at the start of each training epoch."""
        if not self.should_support_missing_values:
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
        # TODO: check if this part needed
        # if stage != 'train':
        #     mask_vector = np.ones(x.shape[1])
        #     mask = torch.from_numpy(mask_vector).float().unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        # If we are in train and mask is not generated yet

        # If the model should support missing values, we generate the mask for each batch
        if self.should_support_missing_values:
            self.mask_for_nones = (~torch.isnan(x)).float()
            self.mask = self.generate_mask()


        if self.mask is None:
            self.mask = self.generate_mask()
            mask = self.mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        else:
            # If we are in train and mask is already generated
            mask = self.mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)

        # Forward pass
        reconstructed, p, _ = self.network(x, mask, self.mask_for_nones)
        # Calculate loss
        if not self.should_support_missing_values:
            # Only calculate the reconstruction loss and mlp
            reconstruct_loss, mlp_loss = self.mse_criterion(reconstructed, x), self.bce_criterion(p, y)
        else:
            # mask only for the features that are actually not missing, but additionally masked
            loss_mask = self.mask_for_nones * (1 - mask)
            reconstruct_loss = ((x - reconstructed) ** 2 * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            mlp_loss = self.bce_criterion(p, y)

        loss = reconstruct_loss + mlp_loss
        # Calculate metrics
        # Calc accuracy
        predicted = p > 0.5
        acc = self.accuracy(predicted, y)
        # Log
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{stage}_reconstruct_loss', reconstruct_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{stage}_mlp_loss', mlp_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        y_true = y.cpu().numpy()
        y_flat = y_true.flatten()
        if len(set(y_flat)) > 1:  # Check if there is more than one class present
            auc = self.auroc(p, y)
            self.log(f"{stage}_auc", auc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def generate_mask(self):
        """
        Generate a bernouli mask which is a tensor of size (input_size)
        :return:
        """
        # Generate p
        # Generate a random tensor of size (input_size)
        # p = torch.rand(self.input_size)
        p = np.random.uniform(0.3, 0.9)
        p = torch.full((self.input_size,), p)
        # p = np.random.beta(2, 2)
        mask = torch.bernoulli(p)
        return mask


class DAEModel:
    def __init__(self,
                 input_size,
                 latent_dim,
                 encoder_units: Tuple[int, int],  # encoder -> latent_dim -> decoder (1 hidden layer each)
                 decoder_units: Tuple[int, int],
                 dropout_rate,
                 learning_rate,
                 activation_name=ActivationFactory.relu_NAME,
                 should_support_missing_values=False,
                 use_wandb=True
                 ):

        # define lightning model
        self.model = DAELightning(input_size=input_size,
                                  latent_dim=latent_dim,
                                  encoder_units=encoder_units,
                                  decoder_units=decoder_units,
                                  activation_name=activation_name,
                                  dropout_rate=dropout_rate,
                                  learning_rate=learning_rate,
                                  should_support_missing_values=should_support_missing_values)

        self.trainer = None
        self.tensorboard_logger = None
        self.wandb_logger = None
        self.use_wandb = use_wandb

        self.should_support_missing_values = should_support_missing_values

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, x_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
            x_test: pd.DataFrame = None, y_test: pd.DataFrame = None,
            show_progress=True,
            missing_indicators: List[int] = None) -> Dict:
        self.model.train()

        """
        missing_indicators = ["", "?", "NA"]
        """

        # If we support missing values, we should convert them to np.nan so that the model will recognize them
        if self.should_support_missing_values:
            x = x.replace(missing_indicators, np.nan)
            y = y.replace(missing_indicators, np.nan)

            if x_val is not None:
                x_val = x_val.replace(missing_indicators, np.nan)
            if y_val is not None:
                y_val = y_val.replace(missing_indicators, np.nan)

            if x_test is not None:
                x_test = x_test.replace(missing_indicators, np.nan)
            if y_test is not None:
                y_test = y_test.replace(missing_indicators, np.nan)

        # Create the dataset
        # print(torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1).shape)
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
        self.tensorboard_logger = TensorBoardLogger("../test_new_models/logs", name="DAE_model")
        if self.use_wandb:
            self.wandb_logger = WandbLogger(project='DynamicDAE')
            self.wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

        early_stop_callback = EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.001)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath='checkpoints',
                                              filename='best_model')

        # Use WandB logger only if enabled
        loggers = [self.tensorboard_logger]
        if self.use_wandb:
            loggers.append(self.wandb_logger)

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=2000,
            logger=loggers,
            callbacks=[early_stop_callback, checkpoint_callback],
            devices=[0],
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

        log_dir = self.tensorboard_logger.log_dir
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Get scalars for loss and AUC
        train_loss_values = event_acc.Scalars('train_loss')
        train_reconstruct_loss_values = event_acc.Scalars('train_reconstruct_loss')
        train_mlp_loss_values = event_acc.Scalars('train_mlp_loss')

        val_loss_values = event_acc.Scalars('val_loss')
        val_reconstruct_loss_values = event_acc.Scalars('val_reconstruct_loss')
        val_mlp_loss_values = event_acc.Scalars('val_mlp_loss')

        if x_test is not None and y_test is not None:
            test_loss_values = event_acc.Scalars('test_loss')

        # Extract steps and values for losses
        train_steps = [x.step for x in train_loss_values]
        train_loss = [x.value for x in train_loss_values]
        train_reconstruct_loss = [x.value for x in train_reconstruct_loss_values]
        train_mlp_loss = [x.value for x in train_mlp_loss_values]

        val_steps = [x.step for x in val_loss_values]
        val_loss = [x.value for x in val_loss_values]
        val_reconstruct_loss = [x.value for x in val_reconstruct_loss_values]
        val_mlp_loss = [x.value for x in val_mlp_loss_values]

        if x_test is not None and y_test is not None:
            test_loss = [x.value for x in test_loss_values]

        return {
            'train_loss': train_loss,
            'train_reconstruct_loss': train_reconstruct_loss,
            'train_mlp_loss': train_mlp_loss,
            'train_steps': train_steps,
            'val_loss': val_loss,
            'val_reconstruct_loss': val_reconstruct_loss,
            'val_mlp_loss': val_mlp_loss,
            'val_steps': val_steps,
        }

    def reconstruct(self, x: Union[pd.DataFrame, np.ndarray, torch.Tensor],
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

        constructed, p, _ = self.model.network(x, mask)
        reconstructed = constructed.detach()

        return reconstructed

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
            _, p, _ = self.model.network(x, mask)
            predicted = p > 0.5
            predicted = predicted.cpu().detach().numpy().astype(int)
            return predicted

    def get_latent(self, x: Union[pd.DataFrame, np.ndarray],
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
            _, _, latent = self.model.network(x, mask)
            latent = latent.cpu().detach().numpy().astype(int)
            return latent

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
            constructed, p, _ = self.model.network(x, mask)
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
        self.model = DAELightning.load_from_checkpoint(file_path)
