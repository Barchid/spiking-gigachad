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
from project.utils.linear_classifier import classification

name = "autoencoder_snn"
batch_size = 500
dataset = "cifar10"
ckpt = ""
is_ann = True
is_1layer = False
is_random = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    module = AutoEncoderModule.load_from_checkpoint(ckpt, model=None, strict=False)

    if "dvs" in module.dataset:
        in_channels = 2
    else:
        in_channels = 3

    print("\n\nREAL CHECKPOINT LOAD")
    if is_ann:
        module = AutoEncoderModule.load_from_checkpoint(
            ckpt, model=AutoEncoderANN(in_channels), strict=False
        )
    else:
        module = AutoEncoderModule.load_from_checkpoint(
            ckpt, model=AutoEncoderSNN(in_channels), strict=False
        )

    train_loader, val_loader = get_dataset(dataset=module.dataset)

    # if "snn" in name:
    #     model = AutoEncoderSNN(in_channels=in_channels)
    # else:
    #     model = AutoEncoderANN(in_channels=in_channels)

    # print(module.model)
    model = module.model

    if is_random:
        model = AutoEncoderANN(in_channels) if is_ann else AutoEncoderSNN(in_channels)

    if is_1layer:
        encoder = model.get_encoder_1layer()
    else:
        encoder = model.get_encoder()
    encoder = encoder.to(device)

    print(encoder)

    classification(encoder, train_loader, val_loader, module.dataset, module.is_ann, is_1layer, is_random)


def get_dataset(dataset="cifar10"):
    if dataset == "cifar10":
        trainx = torch.load("cifar10_trainx.pt")
        trainy = torch.load("cifar10_trainy.pt")
        testx = torch.load("cifar10_testx.pt")
        testy = torch.load("cifar10_testy.pt")

        train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch_size)
        test = DataLoader(TensorDataset(testx, testy), batch_size=batch_size)
    elif dataset == "cifar10_whitening":
        trainx = torch.load("cifar10_trainx_whitening.pt")
        trainy = torch.load("cifar10_trainy_whitening.pt")
        testx = torch.load("cifar10_testx_whitening.pt")
        testy = torch.load("cifar10_testy_whitening.pt")

        train = DataLoader(TensorDataset(trainx, trainy), batch_size=batch_size)
        test = DataLoader(TensorDataset(testx, testy), batch_size=batch_size)
    elif dataset == "dvsgesture":
        trainx = torch.load("dvsgesture_trainx.pt")
        trainy = torch.load("dvsgesture_trainy.pt")
        testx = torch.load("dvsgesture_testx.pt")
        testy = torch.load("dvsgesture_testy.pt")

        train = DataLoader(
            TensorDataset(trainx, trainy), batch_size=batch_size, num_workers=0
        )
        test = DataLoader(
            TensorDataset(testx, testy), batch_size=batch_size, num_workers=0
        )
    return train, test


if __name__ == "__main__":
    main()
