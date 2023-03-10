from argparse import ArgumentParser
from os import times
from spikingjelly.datasets.n_mnist import NMNIST

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
import torchmetrics

from project.models.autoencoder import AutoEncoderANN, AutoEncoderSNN
from project.utils.transforms_input import Transform
from project.utils.linear_classifier import classification
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Union


class AutoEncoderModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        timesteps: int = 15,
        dataset="cifar10",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.is_ann = type(model) == AutoEncoderANN
        self.transform = Transform(dataset=dataset, is_ann=self.is_ann)

        self.model: Union[AutoEncoderANN, AutoEncoderSNN] = model

    def forward(self, x):  # x = (BCHW) or (TBCHW)
        input = x
        if "dvs" in self.dataset:
            input = input.to(torch.float)
        else:
            input = input / 255

        x = self.transform(input)  # (BTCHW)

        if self.is_ann:
            x = x.sum(1) / 15.0  # BCHW for ANN
        else:
            x = x.permute(1, 0, 2, 3, 4) # TBCHW for SNN

        x_hat = self.model(x)  # BCHW

        if "dvs" in self.dataset:
            input = input.sum(1) / 15.0 # BCHW of input dvs
            
        return x_hat, input

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, x = self(x)

        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, x = self(x)

        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, x = self(x)

        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # def on_validation_epoch_end(self) -> None:
    #     accuracy = classification(self.model.get_encoder(), self.transform, self.train_set, self.val_set, self.dataset, self.is_ann)
    #     self.log('linear_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
