import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.image_metrics import mse, psnr, ssim, visualize_quality

class ImageMetricsTest(unittest.TestCase):
    def test_identical_images(self) -> None:
        images = torch.rand(2, 3, 128, 128)

        mse_val = mse(images, images)
        psnr_val = psnr(images, images)
        ssim_val = ssim(images, images)

        self.assertAlmostEqual(mse_val.item(), 0.0, places=6)
        self.assertTrue(torch.isinf(psnr_val))
        self.assertAlmostEqual(ssim_val.item(), 1.0, places=4)

    def test_known_constant_difference(self) -> None:
        images = torch.ones(2, 3, 128, 128) * 0.5
        watermarked = torch.ones(2, 3, 128, 128) * 0.6

        mse_val = mse(images, watermarked)
        psnr_val = psnr(images, watermarked)

        self.assertAlmostEqual(mse_val.item(), 0.01, places=5)
        self.assertAlmostEqual(psnr_val.item(), 20.0, places=4)

    def test_ssim_different_images(self) -> None:
        images = torch.rand(2, 3, 128, 128)
        watermarked = images + torch.randn_like(images) * 0.05
        watermarked = torch.clamp(watermarked, 0.0, 1.0)

        ssim_val = ssim(images, watermarked)

        self.assertLess(ssim_val.item(), 1.0)
        self.assertGreater(ssim_val.item(), 0.5)

    def test_single_and_batch_tensor_input(self) -> None:
        single_orig = torch.rand(3, 64, 64)
        single_wm = single_orig + 0.01

        batch_orig = torch.rand(2, 3, 64, 64)
        batch_wm = batch_orig + 0.01

        single_ssim = ssim(single_orig, single_wm)
        batch_ssim = ssim(batch_orig, batch_wm)

        self.assertIsInstance(single_ssim, torch.Tensor)
        self.assertIsInstance(batch_ssim, torch.Tensor)

    def test_visualize_quality(self) -> None:
        image = torch.rand(3, 64, 64)
        watermarked = torch.clamp(image + 0.05, 0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "comparison.png"
            fig = visualize_quality(image,watermarked, diff_factor=10.0, save_path=save_path, show=False)

            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(save_path.exists())
            plt.close(fig)
