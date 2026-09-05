import unittest
from unittest.mock import patch

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.noise import (
    apply_attack,
    apply_random_attack,
)


class ImageAttackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configs = [
            {"name": "none"},
            {"name": "gaussian_noise", "std": 0.05},
            {"name": "gaussian_blur", "kernel_size": 5, "sigma": 1.0},
            {"name": "downscale", "scale_factor": 0.75},
            {"name": "rotation", "angle": 5.0},
        ]

    def test_attacks_preserve_tensor_properties_and_gradients(self) -> None:
        for config in self.configs:
            with self.subTest(attack=config["name"]):
                images = torch.rand(
                    2,
                    3,
                    128,
                    128,
                    requires_grad=True,
                )

                masks = torch.ones(2, 1, 128, 128)
                attacked, attacked_masks = apply_attack(images, masks, config)

                self.assertEqual(attacked.shape, images.shape)
                self.assertEqual(attacked_masks.shape, masks.shape)
                self.assertEqual(attacked_masks.dtype, masks.dtype)
                self.assertEqual(attacked_masks.device, masks.device)
                self.assertEqual(attacked.dtype, images.dtype)
                self.assertEqual(attacked.device, images.device)
                self.assertGreaterEqual(attacked.min().item(), 0.0)
                self.assertLessEqual(attacked.max().item(), 1.0)

                attacked.mean().backward()
                self.assertIsNotNone(images.grad)

                if config["name"] != "none":
                    self.assertFalse(torch.equal(attacked, images))

    def test_random_attack_uses_one_configuration(self) -> None:
        images = torch.rand(2, 3, 128, 128)
        masks = torch.ones(2, 1, 128, 128)

        attacked, attacked_masks = apply_random_attack(images, masks, [{"name": "none"}])

        self.assertIs(attacked, images)
        self.assertIs(attacked_masks, masks)

    def test_rotation_keeps_content_mask_aligned(self) -> None:
        images = torch.zeros(1, 3, 32, 32)
        masks = torch.zeros(1, 1, 32, 32)
        images[:, :, 8:24, 8:24] = 1
        masks[:, :, 8:24, 8:24] = 1

        attacked_images, attacked_masks = apply_attack(
            images,
            masks,
            {"name": "rotation", "angle": 30.0},
        )

        self.assertEqual(attacked_images.shape, images.shape)
        self.assertEqual(attacked_masks.shape, masks.shape)
        self.assertFalse(torch.equal(attacked_masks, masks))
        self.assertTrue(
            torch.all((attacked_masks == 0) | (attacked_masks == 1))
        )
        torch.testing.assert_close(
            attacked_masks,
            TF.rotate(masks, angle=30.0, interpolation=InterpolationMode.NEAREST, fill=0.0),
        )
        torch.testing.assert_close(
            attacked_images,
            TF.rotate(images, angle=30.0, interpolation=InterpolationMode.BILINEAR, fill=0.0),
        )

    def test_non_geometric_attack_reuses_mask(self) -> None:
        images = torch.rand(1, 3, 32, 32)
        masks = torch.ones(1, 1, 32, 32)

        for config in self.configs:
            if config["name"] == "rotation":
                continue
            with self.subTest(attack=config["name"]):
                _, attacked_masks = apply_attack(images, masks, config)
                self.assertIs(attacked_masks, masks)

    def test_random_attack_selects_once_for_image_and_mask(self) -> None:
        images = torch.rand(2, 3, 32, 32)
        masks = torch.zeros(2, 1, 32, 32)
        masks[:, :, 8:24, 8:24] = 1
        config = {"name": "rotation", "angle": 30.0}
        expected_images, expected_masks = apply_attack(images, masks, config)

        with patch("src.noise.random.choice", return_value=config) as choose:
            attacked_images, attacked_masks = apply_random_attack(images, masks, self.configs)

        choose.assert_called_once_with(self.configs)
        torch.testing.assert_close(attacked_images, expected_images)
        torch.testing.assert_close(attacked_masks, expected_masks)

    def test_unknown_attack(self) -> None:
        images = torch.rand(2, 3, 128, 128)
        masks = torch.ones(2, 1, 128, 128)

        with self.assertRaises(ValueError):
            apply_attack(images, masks, {"name": "unknown"})
