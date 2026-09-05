import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from src.data import WatermarkDataset, get_dataloaders


class WatermarkDatasetTest(unittest.TestCase):
    def test_image_and_content_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            raw_dir = data_dir / "raw"
            raw_dir.mkdir()

            image_path = raw_dir / "sample.png"
            Image.new("RGB", (200, 100), color="white").save(image_path)

            splits_file = data_dir / "splits.json"
            splits_file.write_text(json.dumps({"train": ["sample.png"]}))

            dataset = WatermarkDataset(data_dir, splits_file)
            image, mask = dataset[0]

            self.assertEqual(image.shape, (3, 128, 128))
            self.assertEqual(mask.shape, (1, 128, 128))
            self.assertEqual(mask.dtype, torch.float32)
            self.assertTrue(torch.all(mask[:, :32, :] == 0))
            self.assertTrue(torch.all(mask[:, 32:96, :] == 1))
            self.assertTrue(torch.all(mask[:, 96:, :] == 0))

    def test_shapes_color_conversion_and_odd_padding(self) -> None:
        cases = [
            ((100, 200), "RGB", (0, 128, 32, 96)),
            ((128, 65), "L", (31, 96, 0, 128)),
            ((32, 32), "L", (0, 128, 0, 128)),
        ]
        for size, mode, bounds in cases:
            with self.subTest(size=size, mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "raw").mkdir()
                Image.new(mode, size, color="white").save(root / "raw" / "sample.png")
                split_path = root / "splits.json"
                split_path.write_text(json.dumps({"train": ["sample.png"]}))
                dataset = WatermarkDataset(root, split_path)
                image, mask = dataset[0]

                expected_mask = torch.zeros(1, 128, 128)
                top, bottom, left, right = bounds
                expected_mask[:, top:bottom, left:right] = 1
                self.assertEqual(len(dataset), 1)
                self.assertEqual(image.shape, (3, 128, 128))
                self.assertEqual(image.dtype, torch.float32)
                torch.testing.assert_close(mask, expected_mask)
                torch.testing.assert_close(image, expected_mask.expand(3, -1, -1))

    def test_loaders_keep_all_validation_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            Image.new("RGB", (32, 32), color="white").save(root / "raw" / "sample.png")
            split_path = root / "splits.json"
            split_path.write_text(json.dumps({
                split: ["sample.png"] * 3 for split in ("train", "val", "test")
            }))
            train, validation, test = get_dataloaders(root, split_path, batch_size=2, num_workers=0)
            self.assertIsNotNone(train)
            self.assertIsNotNone(validation)
            self.assertIsNotNone(test)
            assert train is not None and validation is not None and test is not None
            self.assertIsInstance(train.sampler, RandomSampler)
            self.assertTrue(train.drop_last)
            self.assertEqual([images.size(0) for images, _ in train], [2])
            for loader in (validation, test):
                self.assertIsInstance(loader.sampler, SequentialSampler)
                self.assertFalse(loader.drop_last)
                self.assertEqual([images.size(0) for images, _ in loader], [2, 1])

            train, _, _ = get_dataloaders(
                root, split_path, batch_size=2, num_workers=0, shuffle_train=False,
            )
            assert train is not None
            self.assertIsInstance(train.sampler, SequentialSampler)

    def test_missing_split_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = root / "splits.json"
            split_path.write_text(json.dumps({"train": ["sample.png"]}))
            with self.assertRaises(ValueError):
                WatermarkDataset(root, split_path, split="val")
            train, validation, test = get_dataloaders(root, split_path, num_workers=0)
            self.assertIsNotNone(train)
            self.assertIsNone(validation)
            self.assertIsNone(test)
