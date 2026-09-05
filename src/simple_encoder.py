"""Original shallow encoder for embedding binary messages into images."""

import torch
from torch import nn


class LegacySimpleWatermarkEncoder(nn.Module):
    def __init__(
        self,
        message_length: int = 16,
        image_channels: int = 3,
        feature_channels: tuple[int, int, int] = (64, 64, 32),
        max_delta: float | None = 0.03,
    ) -> None:
        super().__init__()
        self.message_length = message_length
        self.max_delta = max_delta

        channels1, channels2, channels3 = feature_channels
        self.conv1 = nn.Conv2d(image_channels + message_length, channels1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(channels2, channels3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv_out = nn.Conv2d(channels3, image_channels, kernel_size=3, padding=1)

    def forward(
        self,
        images: torch.Tensor,
        messages: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        height, width = images.shape[-2:]
        expanded_messages = messages.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        features = torch.cat([images, expanded_messages], dim=1)

        features = self.relu1(self.conv1(features))
        features = self.relu2(self.conv2(features))
        features = self.relu3(self.conv3(features))

        residual = self.conv_out(features)
        if self.max_delta is not None:
            residual = self.max_delta * torch.tanh(residual)

        return torch.clamp(images + residual * masks, 0.0, 1.0)


class SimpleWatermarkEncoder(nn.Module):
    """Original encoder with content masking from merge e386ab5."""

    def __init__(self, message_length: int = 32, image_channels: int = 3) -> None:
        super().__init__()
        self.message_length = message_length
        self.image_channels = image_channels
        self.conv1 = nn.Conv2d(image_channels + message_length, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv_out = nn.Conv2d(32, image_channels, kernel_size=3, padding=1)

    def forward(
        self,
        images: torch.Tensor,
        messages: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        height, width = images.shape[-2:]
        expanded_messages = messages.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        features = torch.cat([images, expanded_messages], dim=1)
        features = self.relu1(self.conv1(features))
        features = self.relu2(self.conv2(features))
        features = self.relu3(self.conv3(features))
        residual = self.conv_out(features)
        return torch.clamp(images + residual * masks, 0.0, 1.0)