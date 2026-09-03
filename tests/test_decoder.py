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

    def test_custom_feature_channels(self) -> None:
        decoder = WatermarkDecoder(feature_channels=(16, 32, 64))

        self.assertEqual(decoder.conv1.out_channels, 16)
        self.assertEqual(decoder.conv2.out_channels, 32)
        self.assertEqual(decoder.conv3.out_channels, 64)
        self.assertEqual(decoder.fc.in_features, 64)