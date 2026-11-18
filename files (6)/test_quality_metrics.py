"""Numerical edge cases and report export checks; no training required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from quality_metrics import image_metrics, stage_metrics, export_metrics, METRIC_NAMES


class QualityMetricTests(unittest.TestCase):
    def test_neutral_constant_images(self):
        for size in ((1, 1), (7, 13), (20, 20)):
            for value in (0, 128, 255):
                scores = image_metrics(Image.new("RGB", size, (value,) * 3))
                for name, value in scores.items():
                    self.assertAlmostEqual(value, 0.0, places=10, msg=name)

    def test_known_histogram_and_contrast(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
        scores = image_metrics(Image.fromarray(gray).convert("RGB"))
        self.assertAlmostEqual(scores["entropy"], 8.0)
        self.assertAlmostEqual(scores["rms_contrast"], np.sqrt((256**2 - 1) / 12) / 255)
        self.assertAlmostEqual(scores["average_gradient"], 1 / (255 * np.sqrt(2)))

    def test_components_and_small_images_are_finite(self):
        rng = np.random.default_rng(42)
        for shape in ((1, 3, 3), (9, 7, 3), (23, 31, 3)):
            scores = image_metrics(Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)))
            self.assertTrue(all(np.isfinite(value) for value in scores.values()))
            self.assertAlmostEqual(scores["uiqm"], 0.0282 * scores["uicm"] +
                                   0.2953 * scores["uism"] + 3.5753 * scores["uiconm"])
            self.assertTrue(0 <= scores["entropy"] <= 8)

    def test_delta_baselines_and_exports(self):
        original = Image.new("RGB", (20, 20), (64, 64, 64))
        enhanced = Image.fromarray(np.tile(np.arange(20, dtype=np.uint8), (20, 1))).convert("RGB")
        results = stage_metrics([("Original", original), ("UICE-MIRNet", enhanced),
                                 ("+ CLAHE (final)", enhanced)])
        for name in METRIC_NAMES:
            self.assertEqual(results[0][name + "_change_from_original"], 0)
            self.assertEqual(results[1][name + "_change_from_mirnet"], 0)
            self.assertEqual(results[2][name + "_change_from_mirnet"], 0)
            self.assertAlmostEqual(results[1][name + "_change_from_original"],
                                   results[1][name] - results[0][name])
        report = {"images": [{"test_image": "test.png", "results": results}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"
            export_metrics(report, path)
            self.assertIn("metric_definitions", json.loads(path.read_text()))
            with path.with_suffix(".csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(float(rows[1]["entropy"]), results[1]["entropy"])


if __name__ == "__main__":
    unittest.main()
