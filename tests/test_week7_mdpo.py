from __future__ import annotations

import math
import unittest

from src.training.week7_mdpo import dpo_loss_and_coefficient


class Week7MdpoTests(unittest.TestCase):
    def test_zero_policy_shift_has_log_two_loss_and_negative_chosen_gradient(self) -> None:
        loss, coefficient, margin = dpo_loss_and_coefficient(-2.0, -3.0, -2.0, -3.0, 0.1)
        self.assertAlmostEqual(loss, math.log(2.0))
        self.assertAlmostEqual(coefficient, -0.05)
        self.assertEqual(margin, 0.0)

    def test_positive_policy_reference_margin_reduces_loss(self) -> None:
        loss, coefficient, margin = dpo_loss_and_coefficient(-1.0, -3.0, -2.0, -3.0, 0.1)
        self.assertGreater(margin, 0.0)
        self.assertLess(loss, math.log(2.0))
        self.assertLess(coefficient, 0.0)


if __name__ == "__main__":
    unittest.main()
