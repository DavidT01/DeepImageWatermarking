import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import TrainConfig, fit, load_checkpoint, save_checkpoint, validate


class PaddingOnlyEncoder(nn.Module):
    def forward(self, images, messages, masks):
        return images + (1 - masks) * 0.5


class MarkerDecoder(nn.Module):
    def forward(self, images):
        markers = images[:, 0, 8, 0].unsqueeze(1)
        return torch.where(markers > 0.5, 1.0, -1.0).expand(-1, 32)


class TrainingUtilitiesTest(unittest.TestCase):
    def test_validation_ignores_padding_and_weights_samples(self) -> None:
        images = torch.zeros(3, 3, 32, 32)
        images[2, 0, 8, 0] = 1
        masks = torch.zeros(3, 1, 32, 32)
        masks[:, :, 8:24, :] = 1
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        stats = validate(
            PaddingOnlyEncoder(),
            MarkerDecoder(),
            loader,
            nn.BCEWithLogitsLoss(),
            torch.device("cpu"),
            fixed_messages=torch.zeros(3, 32),
        )

        self.assertEqual(stats["image_loss"], 0.0)
        self.assertEqual(stats["psnr"], float("inf"))
        self.assertAlmostEqual(stats["ber"], 1 / 3)
        self.assertAlmostEqual(stats["exact_accuracy"], 2 / 3)

    def test_fit_uses_stable_validation_and_separate_outputs(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        masks = torch.ones(2, 1, 16, 16)
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TrainConfig(
                experiment_name="smoke",
                epochs=2,
                batch_size=2,
                encoder_channels=(32, 24, 16),
                decoder_channels=(16, 32, 64),
                learning_rate=0.0,
                device="cpu",
                checkpoint_dir=str(root / "checkpoints"),
                log_path=str(root / "experiments.csv"),
                attack_configs=[{"name": "none"}],
            )

            encoder, decoder, history = fit(loader, loader, config=config)
            checkpoint_path = root / "checkpoints" / "smoke" / "last.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            with (root / "experiments.csv").open(
                newline="",
                encoding="utf-8",
            ) as log_file:
                rows = list(csv.DictReader(log_file))

            self.assertTrue(checkpoint_path.exists())

        self.assertEqual(history[0]["val_loss"], history[1]["val_loss"])
        self.assertEqual(encoder.conv1.out_channels, 32)
        self.assertEqual(decoder.conv3.out_channels, 64)
        self.assertEqual(checkpoint["config"]["experiment_name"], "smoke")
        self.assertEqual(
            checkpoint["config"]["encoder_channels"],
            (32, 24, 16),
        )
        self.assertEqual(
            checkpoint["config"]["decoder_channels"],
            (16, 32, 64),
        )
        self.assertEqual(checkpoint["config"]["attack_configs"], [{"name": "none"}])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["experiment"] == "smoke" for row in rows))
        self.assertEqual(rows[0]["encoder_channels"], "[32, 24, 16]")
        self.assertEqual(rows[0]["decoder_channels"], "[16, 32, 64]")

    def test_checkpoint_restores_random_state(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        encoder = nn.Linear(2, 2).to(device)
        decoder = nn.Linear(2, 2).to(device)
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
                epoch=3,
                best_val_loss=0.25,
                config=TrainConfig(),
            )

            expected_cpu = torch.rand(4)
            expected_cuda = (
                torch.rand(4, device=device)
                if device.type == "cuda"
                else None
            )
            next_epoch, best_loss = load_checkpoint(
                path,
                encoder,
                decoder,
                optimizer,
                device,
            )

        torch.testing.assert_close(torch.rand(4), expected_cpu)
        if expected_cuda is not None:
            torch.testing.assert_close(torch.rand(4, device=device), expected_cuda)
        self.assertEqual(next_epoch, 4)
        self.assertEqual(best_loss, 0.25)
