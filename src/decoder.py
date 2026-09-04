"""CNN decoder for recovering binary messages from images."""

import torch
from torch import nn


class WatermarkDecoder(nn.Module):
    """Decode one binary message from each image in a batch."""

    def __init__(
        self,
        message_length: int = 16,
        image_channels: int = 3,
        feature_channels: int = 40,
    ) -> None:
        super().__init__()
        self.message_length = message_length

        self.conv1 = nn.Conv2d(image_channels + 1, feature_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_channels)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_channels)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(feature_channels)
        self.relu5 = nn.ReLU()

        self.conv_out = nn.Conv2d(feature_channels, message_length, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(message_length)
        self.relu_out = nn.ReLU()

        self.fc = nn.Linear(message_length, message_length)

    def _masked_global_average(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Average each feature channel over image content only."""
        content_sum = (features * masks).sum(dim=(2, 3))
        content_pixels = masks.sum(dim=(2, 3)).clamp_min(1.0)
        return content_sum / content_pixels

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per message bit for each image."""
        x = torch.cat([images, masks], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.relu_out(x)

        x = self._masked_global_average(x, masks)
        return self.fc(x)