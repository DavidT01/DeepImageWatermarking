"""Image attacks used for robustness training and evaluation."""

import random

import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def gaussian_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise to a batch of images."""
    noise = torch.randn_like(images) * std
    return (images + noise).clamp(0, 1)


def gaussian_blur(
    images: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Blur a batch of images with a Gaussian kernel."""
    blurred = TF.gaussian_blur(images, kernel_size=kernel_size, sigma=sigma)
    return blurred.clamp(0, 1)


def downscale(images: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Downscale images and restore their original size."""
    height, width = images.shape[-2:]
    scaled_size = (round(height * scale_factor), round(width * scale_factor))

    scaled = F.interpolate(
        images,
        size=scaled_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    restored = F.interpolate(
        scaled,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return restored.clamp(0, 1)


def rotation(images: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a batch of images by the same angle."""
    rotated = TF.rotate(
        images,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    return rotated.clamp(0, 1)


def apply_attack(images: torch.Tensor, config: dict) -> torch.Tensor:
    """Apply the attack described by a configuration dictionary."""
    name = config["name"]

    if name == "none":
        return images
    if name == "gaussian_noise":
        return gaussian_noise(images, std=config["std"])
    if name == "gaussian_blur":
        return gaussian_blur(
            images,
            kernel_size=config["kernel_size"],
            sigma=config["sigma"],
        )
    if name == "downscale":
        return downscale(images, scale_factor=config["scale_factor"])
    if name == "rotation":
        return rotation(images, angle=config["angle"])

    raise ValueError(f"Unknown attack: {name}")


def apply_random_attack(images: torch.Tensor, configs: list[dict]) -> torch.Tensor:
    """Apply one randomly selected attack configuration."""
    return apply_attack(images, random.choice(configs))
