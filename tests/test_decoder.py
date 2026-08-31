import unittest

import torch
from torch import nn

from src.decoder import WatermarkDecoder


class WatermarkDecoderTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        decoder = WatermarkDecoder()
        images = torch.rand(4, 3, 128, 128)

        logits = decoder(images)

        self.assertEqual(logits.shape, (4, 32))

    def test_backward(self) -> None:
        decoder = WatermarkDecoder()
        images = torch.rand(2, 3, 128, 128)
        messages = torch.randint(0, 2, (2, 32), dtype=torch.float32)

        logits = decoder(images)
        loss = nn.BCEWithLogitsLoss()(logits, messages)
        loss.backward()

        for parameter in decoder.parameters():
            self.assertIsNotNone(parameter.grad)