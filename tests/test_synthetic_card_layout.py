import tempfile
import unittest
from pathlib import Path

from scripts.build_exploration_pool_v4 import WIDTH, FONT
from src.evaluation.synthetic_card_layout import layout_card, write_card


class SyntheticCardLayoutTests(unittest.TestCase):
    def test_long_style_retains_last_glyph_pixels(self):
        text = "STYLE INDUSTRIAL MODERN"
        item = layout_card([text])[0]
        self.assertEqual(item["scale"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card.ppm"
            write_card(path, [text], 1)
            pixels = path.read_bytes().split(b"\n", 3)[3]
            start_x = 20 + (len(text) - 1) * 12
            for y, bits in enumerate(FONT["N"]):
                for x, bit in enumerate(bits):
                    if bit == "1":
                        pos = ((27 + y * 2) * WIDTH + start_x + x * 2) * 3
                        self.assertEqual(pixels[pos:pos + 3], bytes((25, 31, 42)))

    def test_rejects_overflow_and_missing_glyph_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.ppm"
            for lines in (["A" * 29], ["A"] * 8, ["中文"], []):
                with self.assertRaises(ValueError):
                    write_card(path, lines, 1)
                self.assertFalse(path.exists())

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card.ppm"
            write_card(path, ["STYLE MODERN"], 1)
            with self.assertRaises(FileExistsError):
                write_card(path, ["STYLE RUSTIC"], 1)

    def test_price_underscore_is_drawn_as_underscore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card.ppm"
            write_card(path, ["MID_RANGE"], 1)
            pixels = path.read_bytes().split(b"\n", 3)[3]
            x = 20 + 3 * 18
            for column in range(15):
                pos = ((27 + 18) * WIDTH + x + column) * 3
                self.assertEqual(pixels[pos:pos + 3], bytes((25, 31, 42)))


if __name__ == "__main__":
    unittest.main()
