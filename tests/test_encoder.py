import unittest

import torch

from src.encoder import WatermarkEncoder

class WatermarkEncoderTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(4, 3, 128, 128)
        messages = torch.randint(0, 2, (4, 32), dtype=torch.float32)

        watermarked = encoder(images, messages)

        self.assertEqual(watermarked.shape, (4, 3, 128, 128))

    def test_output_value_range(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(4, 3, 128, 128)
        messages = torch.randint(0, 2, (4, 32), dtype=torch.float32)

        watermarked = encoder(images, messages)

        self.assertGreaterEqual(watermarked.min().item(), 0.0)
        self.assertLessEqual(watermarked.max().item(), 1.0)

    def test_backward(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(2, 3, 128, 128, requires_grad=True)
        messages = torch.randint(0, 2, (2, 32), dtype=torch.float32)

        watermarked = encoder(images, messages)
        loss = watermarked.mean()
        loss.backward()

        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
        self.assertIsNotNone(images.grad)
