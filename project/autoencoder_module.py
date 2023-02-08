from argparse import ArgumentParser
from os import times
from spikingjelly.datasets.n_mnist import NMNIST

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
import torchmetrics

from project.models.autoencoder import AutoEncoderSNN


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, model, learning_rate: float=0.001, timesteps: int = 15, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # self.model = AutoEncoderSNN(
        #     in_channels=2,
        #     neuron_model=self.hparams.neuron_model,
        #     bias=self.hparams.bias
        # )
        self.model = model

    def forward(self, x):
        # Reshapes the input tensor from (B, T, C, H, W) to (T, B, C, H, W).
        x = x.permute(1, 0, 2, 3, 4)

        # (T, B, C, H, W) --> (T, B, C, H, W)
        x = self.model(x)
        
        x = x.permute(1, 0, 2, 3, 4) # BTCHW

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
