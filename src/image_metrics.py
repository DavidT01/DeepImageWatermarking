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

def visualize_quality(image: torch.Tensor, watermarked_image: torch.Tensor, diff_factor: float = 10.0,
                      save_path: str | Path | None = None, show: bool = False) -> plt.Figure:
    """Visualize original image, watermarked image and amplified difference.

    Args:
        image: Original image tensor of shape (C, H, W) or (1, C, H, W).
        watermarked_image: Watermarked image tensor of shape matching `image`.
        diff_factor: Amplification factor for difference visualization.
        save_path: Optional file path to save the generated plot.
        show: Whether to display the plot with plt.show() or not.

    Returns:
        Matplotlib Figure object.
    """
    if image.ndim == 4:
        image = image[0]
    if watermarked_image.ndim == 4:
        watermarked_image = watermarked_image[0]

    img_orig = image.detach().cpu().permute(1, 2, 0).numpy()
    img_wm = watermarked_image.detach().cpu().permute(1, 2, 0).numpy()

    diff = np.clip(np.abs(img_orig - img_wm) * diff_factor, 0.0, 1.0)
    img_orig = np.clip(img_orig, 0.0, 1.0)
    img_wm = np.clip(img_wm, 0.0, 1.0)

    cmap = "gray" if img_orig.shape[2] == 1 else None
    if cmap == "gray":
        img_orig = img_orig.squeeze(axis=-1)
        img_wm = img_wm.squeeze(axis=-1)
        diff = diff.squeeze(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_orig, cmap=cmap)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img_wm, cmap=cmap)
    axes[1].set_title("Watermarked Image")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap=cmap)
    axes[2].set_title(f"Difference (x{diff_factor:g})")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig

mse = mean_squared_error
psnr = peak_signal_noise_ratio
ssim = structural_similarity_index
