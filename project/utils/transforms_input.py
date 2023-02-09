from snntorch import spikegen
import tonic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


class TrainableCoding(nn.Module):
    """Some Information about TrainableCoding"""

    def __init__(self):
        super(TrainableCoding, self).__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=(5,5), padding=2, bias=False)

    def forward(self, x: torch.Tensor):
        # CHW
        x = self.conv(x)
        x.unsqueeze_(0)
        x = x.repeat(15, 1, 1, 1, 1)
        # x = spikegen.rate(x, 15)
        return x


class Transform(nn.Module):
    def __init__(self) -> None:
        super(Transform, self).__init__()
        self.trans = transforms.Compose(
            [
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomErasing(),
            ]
        )

        self.code = TrainableCoding()

    def forward(self, input: torch.Tensor):
        # B, C,H,W
        input = self.trans(input)
        input = self.code(input)  # btchw
        return input
