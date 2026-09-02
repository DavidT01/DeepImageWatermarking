"""CNN encoder for hiding binary messages into images."""

import torch
from torch import nn

class WatermarkEncoder(nn.Module):
    """Embed a binary message into an image batch as a watermark."""

    def __init__(self, message_length: int = 32, image_channels: int = 3) -> None:
        super().__init__()
        self.message_length = message_length
        self.image_channels = image_channels

        in_channels = image_channels + message_length

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
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

        x = torch.cat([images, expanded_messages], dim=1)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        residual = self.conv_out(x)

        watermarked_images = torch.clamp(images + residual * masks, 0.0, 1.0)
        return watermarked_images
