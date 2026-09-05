import ast
import inspect
import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


def notebook_function(name):
    path = Path(__file__).resolve().parents[1] / "notebooks" / "01_data.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        tree = ast.parse("".join(source) if isinstance(source, list) else source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {"os": os, "json": json, "random": random}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Missing notebook function: {name}")


class DataNotebookTest(unittest.TestCase):
    def test_existing_split_is_loaded_without_overwriting_or_scanning(self):
        create_splits = notebook_function("create_splits")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "splits.json"
            original = b'{"train": ["val2017/old.jpg"], "val": [], "test": []}\n'
            output.write_bytes(original)
            with patch("os.listdir", side_effect=AssertionError("Unexpected rescan")), redirect_stdout(io.StringIO()):
                result = create_splits(str(Path(directory) / "missing"), str(output))
            self.assertEqual(result, json.loads(original))
            self.assertEqual(output.read_bytes(), original)

    def test_new_split_is_independent_of_file_order_and_global_rng(self):
        create_splits = notebook_function("create_splits")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "val2017"
            source.mkdir()
            names = [f"{index:03}.jpg" for index in range(20)]
            for name in names:
                (source / name).touch()
            state = random.getstate()
            try:
                with patch("os.listdir", return_value=names), redirect_stdout(io.StringIO()):
                    first = create_splits(str(source), str(Path(directory) / "first.json"))
                self.assertEqual(random.getstate(), state)
                random.seed(1234)
                changed_state = random.getstate()
                with patch("os.listdir", return_value=list(reversed(names))), redirect_stdout(io.StringIO()):
                    second = create_splits(str(source), str(Path(directory) / "second.json"))
                self.assertEqual(random.getstate(), changed_state)
            finally:
                random.setstate(state)
            self.assertEqual(first, second)
            self.assertEqual([len(first[key]) for key in ("train", "val", "test")], [16, 2, 2])
            paths = first["train"] + first["val"] + first["test"]
            self.assertEqual(len(set(paths)), 20)

    def test_regeneration_requires_overwrite_and_rejects_invalid_ratios(self):
        create_splits = notebook_function("create_splits")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "val2017"
            source.mkdir()
            (source / "new.jpg").touch()
            output = Path(directory) / "splits.json"
            original = b'{"train": ["old.jpg"], "val": [], "test": []}'
            output.write_bytes(original)
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                create_splits(str(source), str(output), train_ratio=-0.1, val_ratio=0.1,
                              test_ratio=1.0, overwrite=True)
            self.assertEqual(output.read_bytes(), original)
            with redirect_stdout(io.StringIO()):
                result = create_splits(str(source), str(output), overwrite=True)
            self.assertEqual(json.loads(output.read_bytes()), result)
            self.assertNotEqual(output.read_bytes(), original)

    def test_download_reports_all_existing_images(self):
        download_dataset = notebook_function("download_dataset")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "val2017"
            source.mkdir()
            for name in ("first.jpg", "second.jpg", "ignored.txt"):
                (source / name).touch()
            output = io.StringIO()
            with redirect_stdout(output):
                download_dataset(directory)
            self.assertIn("2 images", output.getvalue())
            self.assertNotIn("n", inspect.signature(download_dataset).parameters)