from argparse import ArgumentParser
from os import times
from spikingjelly.datasets.n_mnist import NMNIST

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
import torchmetrics

from project.models.autoencoder import AutoEncoderSNN
from project.utils.transforms_input import Transform


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, model, learning_rate: float=0.001, timesteps: int = 15, dataset="cifar10", **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if "dvs" not in dataset:
            self.transform = Transform()
        else:
            self.transform = torch.nn.Identity()
            
        self.model = model

    def forward(self, x): # x = (BCHW)
        x = self.transform(x) # (TBCHW)
        
        x = self.model(x)

        x = x.mean(0) # (BCHW)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x, reduction="mean")

        self.log('test_loss', loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
