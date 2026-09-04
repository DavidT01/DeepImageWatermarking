"""CNN encoder for hiding binary messages into images."""

import torch
from torch import nn

class WatermarkEncoder(nn.Module):
    """Embed a binary message into an image batch as a watermark."""

    def __init__(
        self,
        message_length: int = 16,
        image_channels: int = 3,
        feature_channels: int = 40,
        max_delta: float | None = None,
    ) -> None:
        super().__init__()
        self.message_length = message_length
        self.image_channels = image_channels
        self.max_delta = max_delta

        self.conv1 = nn.Conv2d(image_channels, feature_channels, kernel_size=3, padding=1)
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

        fusion_channels = feature_channels + message_length + image_channels + 1
        self.conv5 = nn.Conv2d(fusion_channels, feature_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(feature_channels)
        self.relu5 = nn.ReLU()

        self.conv_out = nn.Conv2d(
            feature_channels,
            image_channels,
            kernel_size=1,
        )

    def forward(
        self,
        images: torch.Tensor,
        messages: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Embed binary messages into input images and return watermarked images.

        Args:
            images: Batch of images with shape (N, C, H, W) and values in [0, 1].
            messages: Batch of binary messages with shape (N, message_length).
            masks: Batch of content masks with shape (N, 1, H, W).

        Returns:
            Watermarked images with shape (N, C, H, W) and values in [0, 1].
        """
        _, _, height, width = images.shape

        expanded_messages = messages.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

        x = self.conv1(images)
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

        x = torch.cat([x, expanded_messages, images, masks], dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        residual = self.conv_out(x)
        if self.max_delta is not None:
            residual = self.max_delta * torch.tanh(residual)

        watermarked_images = torch.clamp(images + residual * masks, 0.0, 1.0)
        return watermarked_images
