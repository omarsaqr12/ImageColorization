"""Offline demo checks; never download model weights in the test suite."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.colorize_image import load_rgb, ROOT


class ImageDemoTests(unittest.TestCase):
    def test_rgba_input_converts_to_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            Image.new("RGBA", (5, 3), (20, 30, 40, 255)).save(path)
            array = load_rgb(path)
            self.assertEqual(array.shape, (3, 5, 3))
            np.testing.assert_array_equal(array[0, 0], [20, 30, 40])

    def test_grayscale_input_converts_to_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.png"
            Image.new("L", (4, 2), 80).save(path)
            array = load_rgb(path)
            self.assertEqual(array.shape, (2, 4, 3))
            self.assertTrue((array == 80).all())

    def test_missing_input_fails_before_pretrained_download(self):
        script = ROOT / "scripts" / "colorize_image.py"
        result = subprocess.run(
            [sys.executable, str(script), "does-not-exist.jpg", "--output", "out.png",
             "--download-pretrained"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Input image not found", result.stderr)

    def test_help_runs_without_importing_torch(self):
        script = ROOT / "scripts" / "colorize_image.py"
        result = subprocess.run([sys.executable, str(script), "--help"],
                                capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--download-pretrained", result.stdout)


if __name__ == "__main__":
    unittest.main()
