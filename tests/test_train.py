import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import (
    TrainConfig,
    _append_log,
    fit,
    load_checkpoint,
    save_checkpoint,
    validate,
)


class PaddingOnlyEncoder(nn.Module):
    def forward(self, images, messages, masks):
        return images + (1 - masks) * 0.5


class MarkerDecoder(nn.Module):
    def forward(self, images, masks):
        markers = images[:, 0, 8, 0].unsqueeze(1)
        return torch.where(markers > 0.5, 1.0, -1.0).expand(-1, 16)


class TrainingUtilitiesTest(unittest.TestCase):
    def test_fit_stops_at_target_validation_ber(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        masks = torch.ones(2, 1, 16, 16)
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TrainConfig(
                experiment_name="target-ber",
                epochs=3,
                batch_size=2,
                encoder_channels=8,
                decoder_channels=8,
                learning_rate=0.0,
                device="cpu",
                checkpoint_dir=str(root / "checkpoints"),
                log_path=str(root / "experiments.csv"),
                checkpoint_metric="ber",
                target_val_ber=1.01,
            )

            with redirect_stdout(io.StringIO()):
                _, _, history = fit(loader, loader, config=config)

            checkpoint = torch.load(
                root / "checkpoints" / "target-ber" / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(len(history), 1)
        self.assertEqual(checkpoint["best_val_metric"], history[0]["val_ber"])
        self.assertEqual(checkpoint["config"]["checkpoint_metric"], "ber")

    def test_log_columns_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiments.csv"
            _append_log(path, {"epoch": 1, "loss": 0.5})

            with self.assertRaises(ValueError):
                _append_log(path, {"epoch": 2, "ber": 0.25})

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
            fixed_messages=torch.zeros(3, 16),
        )

        self.assertEqual(stats["image_loss"], 0.0)
        self.assertEqual(stats["psnr"], float("inf"))
        self.assertAlmostEqual(stats["ber"], 1 / 3)
        self.assertAlmostEqual(stats["exact_accuracy"], 2 / 3)

    def test_fit_uses_stable_validation_and_separate_outputs(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        masks = torch.ones(2, 1, 16, 16)
        dataset = TensorDataset(images, masks)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TrainConfig(
                experiment_name="smoke",
                epochs=2,
                batch_size=2,
                encoder_channels=24,
                decoder_channels=32,
                encoder_max_delta=0.03,
                learning_rate=0.0,
                device="cpu",
                checkpoint_dir=str(root / "checkpoints"),
                log_path=str(root / "experiments.csv"),
                attack_configs=[{"name": "none"}],
            )

            output = io.StringIO()
            with patch("src.train.validate", wraps=validate) as validate_mock:
                with redirect_stdout(output):
                    encoder, decoder, history = fit(
                        train_loader,
                        val_loader,
                        config=config,
                    )
            first_val_messages = validate_mock.call_args_list[0].kwargs[
                "fixed_messages"
            ]
            second_val_messages = validate_mock.call_args_list[1].kwargs[
                "fixed_messages"
            ]
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

        torch.testing.assert_close(first_val_messages, second_val_messages)
        self.assertEqual(encoder.conv1.out_channels, 24)
        self.assertEqual(decoder.conv5.out_channels, 32)
        self.assertEqual(checkpoint["config"]["experiment_name"], "smoke")
        self.assertEqual(
            checkpoint["config"]["encoder_channels"],
            24,
        )
        self.assertEqual(
            checkpoint["config"]["decoder_channels"],
            32,
        )
        self.assertEqual(checkpoint["config"]["encoder_max_delta"], 0.03)
        self.assertEqual(checkpoint["config"]["attack_configs"], [{"name": "none"}])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["experiment"] == "smoke" for row in rows))
        self.assertEqual(rows[0]["encoder_channels"], "24")
        self.assertEqual(rows[0]["decoder_channels"], "32")
        self.assertEqual(rows[0]["encoder_max_delta"], "0.03")
        self.assertGreaterEqual(history[0]["epoch_seconds"], 0.0)
        self.assertIn("Epoch 1/2", output.getvalue())
        self.assertIn("val_BER=", output.getvalue())
        self.assertIn("Training time:", output.getvalue())

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
