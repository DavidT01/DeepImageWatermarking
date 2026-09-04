import unittest

import torch

from src.noise import (
    apply_attack,
    apply_attack_with_mask,
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

                attacked = apply_attack(images, config)

                self.assertEqual(attacked.shape, images.shape)
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

        attacked = apply_random_attack(images, [{"name": "none"}])

        self.assertIs(attacked, images)

    def test_rotation_keeps_content_mask_aligned(self) -> None:
        images = torch.zeros(1, 3, 32, 32)
        masks = torch.zeros(1, 1, 32, 32)
        images[:, :, 8:24, 8:24] = 1
        masks[:, :, 8:24, 8:24] = 1

        attacked_images, attacked_masks = apply_attack_with_mask(
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

    def test_non_geometric_attack_reuses_mask(self) -> None:
        images = torch.rand(1, 3, 32, 32)
        masks = torch.ones(1, 1, 32, 32)

        _, attacked_masks = apply_attack_with_mask(
            images,
            masks,
            {"name": "gaussian_noise", "std": 0.01},
        )

        self.assertIs(attacked_masks, masks)

    def test_unknown_attack(self) -> None:
        images = torch.rand(2, 3, 128, 128)

        with self.assertRaises(ValueError):
            apply_attack(images, {"name": "unknown"})
