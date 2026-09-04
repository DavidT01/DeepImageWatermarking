import unittest

import torch
from torch import nn

from src.decoder import WatermarkDecoder


class WatermarkDecoderTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        decoder = WatermarkDecoder()
        images = torch.rand(4, 3, 128, 128)
        masks = torch.ones(4, 1, 128, 128)

        logits = decoder(images, masks)

        self.assertEqual(logits.shape, (4, 16))

    def test_backward(self) -> None:
        decoder = WatermarkDecoder()
        images = torch.rand(2, 3, 128, 128)
        masks = torch.ones(2, 1, 128, 128)
        messages = torch.randint(0, 2, (2, 16), dtype=torch.float32)

        logits = decoder(images, masks)
        loss = nn.BCEWithLogitsLoss()(logits, messages)
        loss.backward()

        for parameter in decoder.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_architecture(self) -> None:
        decoder = WatermarkDecoder(feature_channels=24)

        self.assertEqual(decoder.conv1.in_channels, 4)
        self.assertEqual(decoder.conv5.out_channels, 24)
        self.assertEqual(decoder.conv_out.out_channels, 16)
        self.assertEqual(decoder.fc.in_features, 16)
        self.assertEqual(
            sum(isinstance(module, nn.BatchNorm2d) for module in decoder.modules()),
            6,
        )
        self.assertFalse(
            any(
                isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
                for module in decoder.modules()
            )
        )

    def test_masked_global_average(self) -> None:
        decoder = WatermarkDecoder(message_length=2)
        features = torch.tensor(
            [[[[1.0, 3.0], [100.0, 100.0]], [[2.0, 4.0], [200.0, 200.0]]]]
        )
        masks = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

        pooled = decoder._masked_global_average(features, masks)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))