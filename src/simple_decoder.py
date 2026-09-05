"""Original shallow decoder with three local pooling stages."""

import torch
from torch import nn


class SimpleWatermarkDecoder(nn.Module):
    def __init__(
        self,
        message_length: int = 16,
        feature_channels: tuple[int, int, int] = (32, 64, 128),
        pooling: str = "max",
    ) -> None:
        super().__init__()
        if pooling not in {"max", "avg"}:
            raise ValueError("pooling must be 'max' or 'avg'")
        pool = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d
        channels1, channels2, channels3 = feature_channels

        self.conv1 = nn.Conv2d(3, channels1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = pool(kernel_size=2)

        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = pool(kernel_size=2)

        self.conv3 = nn.Conv2d(channels2, channels3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = pool(kernel_size=2)

        self.fc = nn.Linear(channels3, message_length)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.pool1(self.relu1(self.conv1(images)))
        features = self.pool2(self.relu2(self.conv2(features)))
        features = self.pool3(self.relu3(self.conv3(features)))
        features = features.mean(dim=(2, 3))
        return self.fc(features)