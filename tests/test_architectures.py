import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.decoder import WatermarkDecoder
from src.encoder import WatermarkEncoder
from src.evaluate import load_models
from src.simple_decoder import LegacySimpleWatermarkDecoder, SimpleWatermarkDecoder
from src.simple_encoder import LegacySimpleWatermarkEncoder, SimpleWatermarkEncoder
from src.train import TrainConfig, build_models, fit, save_checkpoint


class ArchitectureSelectionTest(unittest.TestCase):
    def test_architecture_defaults(self) -> None:
        advanced_config = TrainConfig()
        encoder, decoder = build_models(advanced_config)
        self.assertEqual(advanced_config.architecture, "advanced")
        self.assertEqual(advanced_config.encoder_channels, 40)
        self.assertIsInstance(encoder, WatermarkEncoder)
        self.assertIsInstance(decoder, WatermarkDecoder)

        simple_config = TrainConfig(architecture="simple")
        encoder, decoder = build_models(simple_config)
        self.assertEqual(simple_config.encoder_channels, (64, 64, 32))
        self.assertEqual(simple_config.decoder_channels, (32, 64, 128))
        self.assertEqual(simple_config.decoder_normalization, "none")
        self.assertIsInstance(encoder, SimpleWatermarkEncoder)
        self.assertIsInstance(decoder, SimpleWatermarkDecoder)
        self.assertEqual(decoder.fc.out_features, 16)

    def test_original_masking_without_residual_bound(self) -> None:
        encoder, decoder = build_models(TrainConfig(architecture="simple"))
        self.assertFalse(hasattr(encoder, "max_delta"))
        self.assertEqual(SimpleWatermarkEncoder().message_length, 32)
        self.assertEqual(SimpleWatermarkDecoder().fc.out_features, 32)
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.zero_()
            encoder.conv_out.bias.fill_(0.2)
            images = torch.full((2, 3, 16, 16), 0.5)
            masks = torch.zeros(2, 1, 16, 16)
            masks[:, :, 4:12, :] = 1
            marked = encoder(images, torch.zeros(2, 16), masks)
            torch.testing.assert_close(marked, images + 0.2 * masks)
            self.assertEqual(decoder(marked, masks).shape, (2, 16))

    def test_original_matches_unbounded_legacy_and_roundtrips(self) -> None:
        config = TrainConfig(architecture="simple")
        encoder, decoder = build_models(config)
        legacy_encoder = LegacySimpleWatermarkEncoder(message_length=16, max_delta=None)
        legacy_decoder = LegacySimpleWatermarkDecoder(message_length=16)
        legacy_encoder.load_state_dict(encoder.state_dict())
        legacy_decoder.load_state_dict(decoder.state_dict())
        images = torch.rand(2, 3, 16, 16)
        masks = torch.ones(2, 1, 16, 16)
        masks[:, :, :4, :] = 0
        messages = torch.zeros(2, 16)
        with torch.no_grad():
            marked = encoder(images, messages, masks)
            torch.testing.assert_close(marked, legacy_encoder(images, messages, masks), rtol=0, atol=0)
            torch.testing.assert_close(decoder(marked), legacy_decoder(marked), rtol=0, atol=0)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.pt"
            save_checkpoint(path, encoder, decoder, optimizer, 0, 0.5, config)
            restored_encoder, restored_decoder, restored_config = load_models(path, "cpu")
        self.assertEqual(restored_config["simple_version"], "original")
        self.assertIsInstance(restored_encoder, SimpleWatermarkEncoder)
        self.assertIsInstance(restored_decoder, SimpleWatermarkDecoder)
        with torch.no_grad():
            torch.testing.assert_close(restored_encoder(images, messages, masks), marked)
            torch.testing.assert_close(restored_decoder(marked), decoder(marked))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(architecture="unknown")
        for config in (
            TrainConfig(architecture="simple", encoder_channels=40),
            TrainConfig(architecture="simple", decoder_channels=(8, 8)),
            TrainConfig(architecture="advanced", encoder_channels=(8, 8, 8)),
            TrainConfig(architecture="simple", decoder_pooling="unknown"),
            TrainConfig(architecture="simple", encoder_max_delta=0.03),
            TrainConfig(architecture="simple", decoder_pooling="avg"),
            TrainConfig(architecture="simple", simple_version="unknown"),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_models(config)

    def test_simple_training_resume_and_loading(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        masks = torch.zeros(2, 1, 16, 16)
        masks[:, :, 4:12, :] = 1
        loader = DataLoader(TensorDataset(images, masks), batch_size=2)

        for pooling in ("max", "avg"):
            with self.subTest(pooling=pooling), tempfile.TemporaryDirectory() as directory:
                config = TrainConfig(
                    architecture="simple",
                    simple_version="legacy",
                    decoder_pooling=pooling,
                    encoder_channels=(8, 8, 4),
                    decoder_channels=(4, 8, 8),
                    encoder_max_delta=0.03,
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    checkpoint_dir=str(Path(directory) / "checkpoints"),
                    log_path=str(Path(directory) / "log.csv"),
                )
                with redirect_stdout(io.StringIO()):
                    encoder, decoder, history = fit(loader, loader, config)
                self.assertEqual(len(history), 1)
                self.assertTrue(all(
                    parameter.grad is not None
                    for model in (encoder, decoder)
                    for parameter in model.parameters()
                ))
                checkpoint = Path(config.checkpoint_dir) / config.experiment_name / "last.pt"
                loaded_encoder, loaded_decoder, saved_config = load_models(checkpoint, "cpu")
                self.assertEqual(saved_config["architecture"], "simple")
                self.assertEqual(saved_config["decoder_pooling"], pooling)
                self.assertIsInstance(
                    loaded_decoder.pool1, nn.MaxPool2d if pooling == "max" else nn.AvgPool2d
                )
                messages = torch.zeros(2, 16)
                with torch.no_grad():
                    marked = loaded_encoder(images, messages, masks)
                    torch.testing.assert_close(marked, encoder(images, messages, masks))
                    torch.testing.assert_close(
                        loaded_decoder(marked, masks), decoder(marked, masks)
                    )
                    self.assertEqual(loaded_decoder(marked, masks).shape, (2, 16))
                    self.assertLessEqual((marked - images).abs().max().item(), 0.030001)
                    padding = masks.expand_as(images) == 0
                    self.assertTrue(torch.equal(marked[padding], images[padding]))

                config.epochs = 2
                with redirect_stdout(io.StringIO()):
                    _, _, continued = fit(loader, loader, config, resume_from=checkpoint)
                self.assertEqual([row["epoch"] for row in continued], [2])

    def test_legacy_checkpoint_loading(self) -> None:
        for architecture, normalization, pooling in (
            ("simple", "none", "max"),
            ("simple", "none", "avg"),
            ("advanced", "batch", "max"),
            ("advanced", "none", "max"),
        ):
            with self.subTest(architecture=architecture, normalization=normalization, pooling=pooling):
                config = TrainConfig(
                    architecture=architecture,
                    simple_version="legacy",
                    encoder_channels=(8, 8, 4) if architecture == "simple" else 8,
                    decoder_channels=(4, 8, 8) if architecture == "simple" else 8,
                    decoder_normalization=normalization,
                    decoder_pooling=pooling,
                )
                encoder, decoder = build_models(config)
                encoder.eval()
                decoder.eval()
                optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "legacy.pt"
                    save_checkpoint(path, encoder, decoder, optimizer, 0, 0.5, config)
                    checkpoint = torch.load(path, weights_only=False)
                    for key in ("architecture", "encoder_channels", "decoder_channels", "simple_version"):
                        del checkpoint["config"][key]
                    torch.save(checkpoint, path)
                    loaded_encoder, loaded_decoder, inferred = load_models(path, "cpu")
                self.assertEqual(inferred["architecture"], architecture)
                images = torch.rand(2, 3, 16, 16)
                masks = torch.ones(2, 1, 16, 16)
                messages = torch.zeros(2, 16)
                with torch.no_grad():
                    marked = encoder(images, messages, masks)
                    torch.testing.assert_close(loaded_encoder(images, messages, masks), marked)
                    torch.testing.assert_close(loaded_decoder(marked, masks), decoder(marked, masks))