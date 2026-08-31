"""CNN decoder for recovering binary messages from images."""

import torch
from torch import nn


class WatermarkDecoder(nn.Module):
    """Decode one binary message from each image in a batch."""

    def __init__(self, message_length: int = 32) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(128, message_length)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one logit per message bit for each image."""
        x = self.conv1(images)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Global average pooling
        x = x.mean(dim=(2, 3))
        return self.fc(x)