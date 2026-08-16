from __future__ import annotations

import unittest

from src.training.week6_data import _split_name


class Week6DataLockTests(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        first = _split_name("sample-1", seed=20260814, validation_fraction=0.05)
        second = _split_name("sample-1", seed=20260814, validation_fraction=0.05)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation"})

    def test_split_changes_only_through_explicit_inputs(self) -> None:
        values = {
            _split_name(f"sample-{index}", seed=20260814, validation_fraction=0.05)
            for index in range(1000)
        }
        self.assertEqual(values, {"train", "validation"})


if __name__ == "__main__":
    unittest.main()
