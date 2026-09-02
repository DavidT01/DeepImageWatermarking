import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data import WatermarkDataset


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
