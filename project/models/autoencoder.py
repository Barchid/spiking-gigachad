import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .base_layers import ConvBnSpike, ConvSpike, LinearSpike
from spikingjelly.clock_driven import surrogate, neuron, functional, layer

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)

def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))

class AutoEncoderSNN(nn.Module):
    """"""

    def __init__(
        self, in_channels: int, neuron_model: str = "LIF"
    ):
        super(AutoEncoderSNN, self).__init__()
        self.conv1 = ConvBnSpike(in_channels, 50, kernel_size=5)
        self.pool1 = layer.SeqToANNContainer(nn.MaxPool2d((2, 2), stride=2))
        self.conv2 = ConvBnSpike(50, 100, kernel_size=3, neuron_model=neuron_model)
        self.pool2 = layer.SeqToANNContainer(nn.MaxPool2d((3, 3), stride=3))

        self.flat = nn.Flatten(start_dim=2)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): input tensor of dimension (T, B, C, H, W).

        Returns:
            torch.Tensor: Tensor (of logits) of dimension (B, num_classes)
        """
        functional.reset_net(self)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)

        print(x.shape)
        exit()
        return x
    
    def get_encoder(self):
        return nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2
        )


class AutoEncoderANN(nn.Module):
    """"""

    def __init__(
        self, in_channels: int, neuron_model: str = "LIF"
    ):
        super(AutoEncoderSNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 50, 5, padding=2, bias=False),
            nn.BatchNorm2d(50)
        )
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, 3, padding=1, bias=False),
            nn.BatchNorm2d(100)
        )
        self.pool2 = nn.MaxPool2d((3, 3), stride=3)
        
        self.flat = nn.Flatten(start_dim=1)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): input tensor of dimension (T, B, C, H, W).

        Returns:
            torch.Tensor: Tensor (of logits) of dimension (B, num_classes)
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        print(x.shape)
        exit()
        return x
    
    def get_encoder(self):
        return nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2
        )