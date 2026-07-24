import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.evaluation.manifests import ManifestValidationError
from src.evaluation.v2_photorealistic import (
    _natural_photo_key,
    fingerprint_photo,
)


class V2PhotorealisticTests(unittest.TestCase):
    def test_fingerprint_photo_uses_actual_png_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hygiene_example_01.png"
            Image.new("RGB", (640, 640), "white").save(path)
            first = fingerprint_photo(path)
            Image.new("RGB", (640, 640), "black").save(path)
            second = fingerprint_photo(path)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(len(first[0]), 64)
        self.assertEqual(len(first[1]), 16)

    def test_fingerprint_photo_rejects_small_or_non_png_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            small = Path(tmpdir) / "small.png"
            Image.new("RGB", (320, 320), "white").save(small)
            with self.assertRaisesRegex(ManifestValidationError, "at least 640px"):
                fingerprint_photo(small)
            jpeg = Path(tmpdir) / "large.jpg"
            Image.new("RGB", (640, 640), "white").save(jpeg)
            with self.assertRaisesRegex(ManifestValidationError, "readable PNG"):
                fingerprint_photo(jpeg)

    def test_photo_sorting_uses_numeric_suffix(self):
        names = [
            Path("facility_fault_10.png"),
            Path("facility_fault_2.png"),
            Path("facility_fault_01.png"),
        ]
        self.assertEqual(
            [path.name for path in sorted(names, key=_natural_photo_key)],
            ["facility_fault_01.png", "facility_fault_2.png", "facility_fault_10.png"],
        )


if __name__ == "__main__":
    unittest.main()
