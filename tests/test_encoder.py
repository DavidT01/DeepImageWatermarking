import unittest

import torch

from src.encoder import WatermarkEncoder

class WatermarkEncoderTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(4, 3, 128, 128)
        messages = torch.randint(0, 2, (4, 32), dtype=torch.float32)
        masks = torch.ones(4, 1, 128, 128)

        watermarked = encoder(images, messages, masks)

        self.assertEqual(watermarked.shape, (4, 3, 128, 128))

    def test_output_value_range(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(4, 3, 128, 128)
        messages = torch.randint(0, 2, (4, 32), dtype=torch.float32)
        masks = torch.ones(4, 1, 128, 128)

        watermarked = encoder(images, messages, masks)

        self.assertGreaterEqual(watermarked.min().item(), 0.0)
        self.assertLessEqual(watermarked.max().item(), 1.0)

    def test_backward(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(2, 3, 128, 128, requires_grad=True)
        messages = torch.randint(0, 2, (2, 32), dtype=torch.float32)
        masks = torch.ones(2, 1, 128, 128)

        watermarked = encoder(images, messages, masks)
        loss = watermarked.mean()
        loss.backward()

        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
        self.assertIsNotNone(images.grad)

    def test_padding_is_unchanged(self) -> None:
        encoder = WatermarkEncoder()
        images = torch.rand(2, 3, 128, 128)
        messages = torch.randint(0, 2, (2, 32), dtype=torch.float32)
        masks = torch.zeros(2, 1, 128, 128)
        masks[:, :, 32:96, :] = 1

        watermarked = encoder(images, messages, masks)
        padding = masks.expand_as(images) == 0

        self.assertTrue(torch.equal(watermarked[padding], images[padding]))
