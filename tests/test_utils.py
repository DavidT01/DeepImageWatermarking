import os
import random
import unittest

import numpy as np
import torch

from src.utils import set_seed


class ReproducibilityTest(unittest.TestCase):
    def test_seed_repeats_python_numpy_and_torch_sequences(self) -> None:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        hash_seed = os.environ.get("PYTHONHASHSEED")
        deterministic = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        devices = list(range(torch.cuda.device_count()))
        try:
            with torch.random.fork_rng(devices=devices):
                set_seed()
                python_values = [random.random() for _ in range(4)]
                numpy_values = np.random.rand(4)
                torch_values = torch.rand(4)
                cuda_values = [torch.rand(4, device=f"cuda:{device}") for device in devices]

                set_seed()
                self.assertEqual([random.random() for _ in range(4)], python_values)
                np.testing.assert_array_equal(np.random.rand(4), numpy_values)
                torch.testing.assert_close(torch.rand(4), torch_values, rtol=0, atol=0)
                for device, expected in zip(devices, cuda_values):
                    torch.testing.assert_close(
                        torch.rand(4, device=f"cuda:{device}"), expected, rtol=0, atol=0,
                    )
                if devices:
                    self.assertTrue(torch.backends.cudnn.deterministic)
                    self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = benchmark
            if hash_seed is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = hash_seed