from __future__ import annotations

import unittest

from src.training.week7_preference import _evidence_overlap, _split_pairs


class Week7PreferenceTests(unittest.TestCase):
    def test_evidence_overlap_requires_every_locked_reference(self) -> None:
        row = {
            "target": {"context_state": {"historical_image_reference": ["VISIBLE-ONE", "VISIBLE-TWO"]}}
        }
        self.assertTrue(_evidence_overlap(row, {"observed_evidence": ["visible-one", "VISIBLE-TWO"]}))
        self.assertFalse(_evidence_overlap(row, {"observed_evidence": ["VISIBLE-ONE"]}))

    def test_split_holds_out_one_pair_per_scenario_and_chosen_role(self) -> None:
        pairs = []
        for scenario in ("image_product_search", "after_sales", "itinerary_planning"):
            for role in ("multitask", "week6"):
                for index in range(2):
                    pairs.append({
                        "sample_id": f"{scenario}-{role}-{index}",
                        "parent_scenario": scenario,
                        "chosen_model_role": role,
                    })
        _split_pairs(pairs, {"split": {"seed": 7}})
        validation = [pair for pair in pairs if pair["split"] == "validation"]
        train = [pair for pair in pairs if pair["split"] == "train"]
        self.assertEqual(len(validation), 6)
        self.assertEqual(len(train), 6)
        self.assertEqual(
            {(pair["parent_scenario"], pair["chosen_model_role"]) for pair in validation},
            {
                (scenario, role)
                for scenario in ("image_product_search", "after_sales", "itinerary_planning")
                for role in ("multitask", "week6")
            },
        )


if __name__ == "__main__":
    unittest.main()
