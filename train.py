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
batch_size = 500
dataset = "cifar10"

def main():
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    train_loader, val_loader = get_dataset(dataset=dataset)

    trainer = create_trainer()
    
    if "snn" in name:
        model = AutoEncoderSNN(in_channels=3)
    else:
        model = AutoEncoderANN(in_channels=3)

    module = AutoEncoderModule(
        model=model,
        dataset=dataset,
        train_set=train_loader,
        val_set=val_loader
    )

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
        monitor="linear_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{linear_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    
    # create trainer
    trainer = pl.Trainer(
        max_epochs=200,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger("experiments", name=f"{name}_{dataset}"),
        default_root_dir=f"experiments/{name}_{dataset}",
        precision=16,
    )
    return trainer


def get_dataset(dataset='cifar10'):
    if dataset == "cifar10":
        trainx = torch.load('cifar10_trainx.pt')
        trainy = torch.load('cifar10_trainy.pt')
        testx = torch.load('cifar10_testx.pt')
        testy = torch.load('cifar10_testy.pt')

        train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch_size)
        test = DataLoader(TensorDataset(testx, testy), batch_size=batch_size)
    elif dataset == 'cifar10_whitening':
        trainx = torch.load('cifar10_trainx_whitening.pt')
        trainy = torch.load('cifar10_trainy_whitening.pt')
        testx = torch.load('cifar10_testx_whitening.pt')
        testy = torch.load('cifar10_testy_whitening.pt')

        train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch_size)
        test = DataLoader(TensorDataset(testx, testy), batch_size=batch_size)
    elif dataset == "dvsgesture":
        trainx = torch.load('dvsgesture_trainx.pt')
        trainy = torch.load('dvsgesture_trainy.pt')
        testx = torch.load('dvsgesture_testx.pt')
        testy = torch.load('dvsgesture_testy.pt')

        train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch_size)
        test = DataLoader(TensorDataset(testx, testy), batch_size=batch_size)
    return train, test


if __name__ == "__main__":
    main()
