import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.decoder import WatermarkDecoder
from src.encoder import WatermarkEncoder
from src.evaluate import evaluate_model, evaluate_scenarios, load_models
from src.train import TrainConfig, save_checkpoint


class MessageEncoder(nn.Module):
    def __init__(self, change_padding=False):
        super().__init__()
        self.change_padding = change_padding

    def forward(self, images, messages, masks):
        watermarked = images.clone()
        watermarked[:, 0, 8, : messages.size(1)] = messages
        if self.change_padding:
            watermarked += (1 - masks) * 0.5
        return watermarked


class MessageDecoder(nn.Module):
    def forward(self, images):
        bits = images[:, 0, 8, :32]
        logits = bits * 20 - 10
        flipped = images[:, 1, 8, 0] > 0.5
        logits[flipped] *= -1
        return logits


class EvaluationUtilitiesTest(unittest.TestCase):
    def test_load_models(self) -> None:
        encoder = WatermarkEncoder(message_length=8)
        decoder = WatermarkDecoder(message_length=8)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters())
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(
                path,
                encoder,
                decoder,
                optimizer,
                epoch=1,
                best_val_loss=0.5,
                config=TrainConfig(message_length=8),
            )
            loaded_encoder, loaded_decoder, config = load_models(path, "cpu")

        for expected, actual in zip(
            encoder.parameters(),
            loaded_encoder.parameters(),
        ):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(
            decoder.parameters(),
            loaded_decoder.parameters(),
        ):
            torch.testing.assert_close(actual, expected)

        self.assertFalse(loaded_encoder.training)
        self.assertFalse(loaded_decoder.training)
        self.assertEqual(config["message_length"], 8)

    def test_evaluate_model_is_deterministic_and_ignores_padding(self) -> None:
        images = torch.zeros(3, 3, 32, 32)
        images[2, 1, 8, 0] = 1
        masks = torch.zeros(3, 1, 32, 32)
        masks[:, :, 8:24, :] = 1
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        clean_padding = evaluate_model(
            MessageEncoder(),
            MessageDecoder(),
            loader,
            "cpu",
        )
        changed_padding = evaluate_model(
            MessageEncoder(change_padding=True),
            MessageDecoder(),
            loader,
            "cpu",
        )
        repeated = evaluate_model(
            MessageEncoder(),
            MessageDecoder(),
            loader,
            "cpu",
        )

        self.assertEqual(clean_padding, repeated)
        self.assertAlmostEqual(clean_padding["ber"], 1 / 3)
        self.assertAlmostEqual(clean_padding["exact_accuracy"], 2 / 3)
        self.assertEqual(clean_padding["mse"], changed_padding["mse"])
        self.assertEqual(clean_padding["psnr"], changed_padding["psnr"])
        self.assertEqual(clean_padding["ssim"], changed_padding["ssim"])
        self.assertEqual(clean_padding["num_images"], 3)

    def test_evaluate_scenarios(self) -> None:
        images = torch.zeros(2, 3, 32, 32)
        masks = torch.ones(2, 1, 32, 32)
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)
        applied_attacks = []
        scenarios = {
            "clean": [],
            "noise-blur": [
                {"name": "noise"},
                {"name": "blur"},
            ],
        }

        def record_attack(batch, config):
            applied_attacks.append(config["name"])
            return batch

        with patch("src.evaluate.apply_attack", side_effect=record_attack):
            results = evaluate_scenarios(
                MessageEncoder(),
                MessageDecoder(),
                loader,
                "cpu",
                scenarios,
            )

        self.assertEqual([result["scenario"] for result in results], list(scenarios))
        self.assertEqual(results[1]["attacks"], scenarios["noise-blur"])
        self.assertEqual(applied_attacks, ["noise", "blur"])
