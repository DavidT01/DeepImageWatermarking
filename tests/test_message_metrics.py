import unittest

import torch

from src.message_metrics import (
    bit_error_rate,
    exact_message_accuracy,
    logits_to_bits,
)


class MessageMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [
                [2.0, -1.0, -0.5, -2.0],
                [-2.0, 1.0, 1.0, -0.1],
            ]
        )
        self.targets = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )

    def test_logits_to_bits(self) -> None:
        logits = torch.tensor([[-1.0, 0.0, 3.0]])

        bits = logits_to_bits(logits)

        expected = torch.tensor([[False, True, True]])
        self.assertTrue(torch.equal(bits, expected))

    def test_bit_error_rate(self) -> None:
        self.assertAlmostEqual(bit_error_rate(self.logits, self.targets).item(), 0.125)

    def test_exact_message_accuracy(self) -> None:
        self.assertAlmostEqual(
            exact_message_accuracy(self.logits, self.targets).item(),
            0.5,
        )