import unittest

import torch
from torch import nn

from src.decoder import WatermarkDecoder
from src.encoder import WatermarkEncoder
from src.noise import apply_attack

class WatermarkPipelineTest(unittest.TestCase):
    def test_encoder_decoder_pipeline(self) -> None:
        encoder = WatermarkEncoder()
        decoder = WatermarkDecoder()

        images = torch.rand(2, 3, 128, 128)
        messages = torch.randint(0, 2, (2, 32), dtype=torch.float32)
        masks = torch.ones(2, 1, 128, 128)

        watermarked = encoder(images, messages, masks)
        logits = decoder(watermarked)

        loss = nn.BCEWithLogitsLoss()(logits, messages)
        loss.backward()

        for parameter in encoder.parameters():
            self.assertIsNotNone(parameter.grad)
        for parameter in decoder.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_pipeline_output_shapes_and_values(self) -> None:
        encoder = WatermarkEncoder()
        decoder = WatermarkDecoder()

        batch_size = 4
        images = torch.rand(batch_size, 3, 128, 128)
        messages = torch.randint(0, 2, (batch_size, 32), dtype=torch.float32)
        masks = torch.ones(batch_size, 1, 128, 128)

        watermarked = encoder(images, messages, masks)
        self.assertEqual(watermarked.shape, (batch_size, 3, 128, 128))
        self.assertGreaterEqual(watermarked.min().item(), 0.0)
        self.assertLessEqual(watermarked.max().item(), 1.0)

        logits_clean = decoder(watermarked)
        self.assertEqual(logits_clean.shape, (batch_size, 32))

        attack_config = {"name": "gaussian_noise", "std": 0.05}
        noisy_watermarked = apply_attack(watermarked, attack_config)
        self.assertEqual(noisy_watermarked.shape, (batch_size, 3, 128, 128))
        self.assertGreaterEqual(noisy_watermarked.min().item(), 0.0)
        self.assertLessEqual(noisy_watermarked.max().item(), 1.0)

        logits_noisy = decoder(noisy_watermarked)
        self.assertEqual(logits_noisy.shape, (batch_size, 32))
