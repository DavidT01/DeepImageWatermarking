"""Metrics for image quality assessment."""

import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
import torch

def mean_squared_error(images: torch.Tensor, watermarked_images: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Squared Error (MSE) between two batches of images.

    Args:
        images: Original images tensor of shape (N, C, H, W) or (C, H, W).
        watermarked_images: Watermarked images tensor of shape matching `images`.

    Returns:
        Scalar PyTorch tensor containing average MSE across the batch.
    """
    return torch.mean((images - watermarked_images) ** 2)

def peak_signal_noise_ratio(images: torch.Tensor, watermarked_images: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two image batches.

    Args:
        images: Original images tensor of shape (N, C, H, W) or (C, H, W).
        watermarked_images: Watermarked images tensor of shape matching `images`.
        max_val: Maximum possible pixel value (default 1.0).

    Returns:
        Scalar PyTorch tensor containing average PSNR in dB.
    """
    mse = mean_squared_error(images, watermarked_images)
    if mse.item() == 0.0:
        return torch.tensor(float("inf"), device=images.device)
    return 10.0 * torch.log10((max_val ** 2) / mse)

def structural_similarity_index(images: torch.Tensor, watermarked_images: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Calculate Structural Similarity Index (SSIM) between two image batches.

    Args:
        images: Original images tensor of shape (N, C, H, W) or (C, H, W).
        watermarked_images: Watermarked images tensor of shape matching `images`.
        max_val: Maximum possible pixel value (default 1.0).

    Returns:
        Scalar PyTorch tensor containing average SSIM across the batch.
    """
    img1 = images.detach().cpu().numpy()
    img2 = watermarked_images.detach().cpu().numpy()

    if img1.ndim == 3:
        img1 = img1[np.newaxis, ...]
        img2 = img2[np.newaxis, ...]

    scores = []
    for i in range(img1.shape[0]):
        im1 = np.transpose(img1[i], (1, 2, 0))
        im2 = np.transpose(img2[i], (1, 2, 0))
        score = skimage_ssim(im1, im2, channel_axis=-1, data_range=max_val)
        scores.append(score)

    return torch.tensor(np.mean(scores), dtype=torch.float32, device=images.device)

mse = mean_squared_error
psnr = peak_signal_noise_ratio
ssim = structural_similarity_index
