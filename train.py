from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from project.autoencoder_module import AutoEncoderModule
from project.models.autoencoder import AutoEncoderANN, AutoEncoderSNN
from torchvision import datasets, transforms

name = "autoencoder_snn"

def main():
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    if "snn" in name:
        model = AutoEncoderSNN(in_channels=3)
    else:
        model = AutoEncoderANN(in_channels=3)

    module = AutoEncoderModule(
        model=model
    )

    train_trans = transforms.Compose(
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(),
    )
    val_trans = transforms.Compose(
        transforms.ToTensor(),
        transforms.Normalize(),
    )
    
    train_set = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=train_trans)
    val_set = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=val_trans)
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
    )

    trainer = create_trainer()

    trainer.fit(module, train_loader, val_loader)

    # report results in a txt file
    report_path = os.path.join(trainer.default_root_dir, 'train_report.txt')
    report = open(report_path, 'a')

    report.write(
        f"{name} {trainer.checkpoint_callback.best_model_score}\n")
    report.flush()
    report.close()


def create_trainer() -> pl.Trainer:
    # saves the best model checkpoint based on the accuracy in the validation set
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    
    # create trainer
    trainer = pl.Trainer(
        max_epochs=200,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=f"{name}"),
        default_root_dir=f"experiments/{name}",
        precision=16,
    )
    return trainer



if __name__ == "__main__":
    main()
