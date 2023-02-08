from spikingjelly.clock_driven import surrogate, neuron, functional, layer
from .base_layers import ConvBnSpike, ConvSpike, LinearSpike
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EncoderSNN(nn.Module):
    """Some Information about EncoderSNN"""
    def __init__(self, in_channels=2):
        super(EncoderSNN, self).__init__()
        self.conv1 = ConvBnSpike(in_channels, 32, kernel_size=5, neuron_model="LIF")
        self.pool1 = nn.MaxPool2d((2,2), 2)
        self.stride1 = ConvSpike(20, 20, kernel_size=3, stride=2, bias=True, neuron_model="LIF")

    def forward(self, x):
        functional.reset_net(self)
        
        return x