import unittest

import torch

from src.noise import apply_attack, apply_random_attack


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

    def test_unknown_attack(self) -> None:
        images = torch.rand(2, 3, 128, 128)

        with self.assertRaises(ValueError):
            apply_attack(images, {"name": "unknown"})
