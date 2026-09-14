import unittest
from pathlib import Path
import tempfile

from scripts.analyze_week8_facility_routing import lf_sha256, route_facility_review


CONFIG = {
    "protocol": "product_visual_observation_v3",
    "facility_refinement": {
        "protocol": "visual_facility_review_v1",
        "mode": "replace",
        "eligibility": "non_food_scene",
        "max_attempts": 2,
        "max_new_tokens": 128,
        "system_prompt": "system",
        "task_prompt": "task",
    },
}


def observation(subject="dining_space", facilities=None):
    return {
        "subject_kind": subject,
        "subject_fact": "visible room",
        "style_evidence": [],
        "facility_evidence": facilities or [],
        "price_text": [],
    }


class Week8FacilityRoutingTradeoffTests(unittest.TestCase):
    def test_analyzer_source_hash_is_line_ending_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            lf = Path(folder) / "lf.py"
            crlf = Path(folder) / "crlf.py"
            lf.write_bytes(b"one\ntwo\n")
            crlf.write_bytes(b"one\r\ntwo\r\n")
            self.assertEqual(lf_sha256(lf), lf_sha256(crlf))

    def test_no_review_and_all_eligible_are_bounded(self):
        value = observation(facilities=[{"label": "seating", "fact": "wooden chairs"}])
        self.assertFalse(route_facility_review("no_review", value, CONFIG))
        self.assertTrue(route_facility_review("all_eligible", value, CONFIG))
        self.assertFalse(route_facility_review(
            "all_eligible", observation(subject="food_closeup"), CONFIG
        ))

    def test_evidence_conflict_uses_observation_fact(self):
        conflicted = observation(facilities=[{
            "label": "bar", "fact": "cocktail glasses on table",
        }])
        supported = observation(facilities=[{
            "label": "bar", "fact": "fixed serving bar",
        }])
        self.assertTrue(route_facility_review("evidence_conflict", conflicted, CONFIG))
        self.assertFalse(route_facility_review("evidence_conflict", supported, CONFIG))
        self.assertFalse(route_facility_review(
            "evidence_conflict", {**conflicted, "subject_kind": "food_closeup"}, CONFIG
        ))

    def test_observable_uncertainty_is_limited_to_identifiable_empty_venues(self):
        self.assertTrue(route_facility_review(
            "observable_uncertainty", observation(subject="retail_space"), CONFIG
        ))
        self.assertFalse(route_facility_review(
            "observable_uncertainty", observation(subject="visitor_attraction"), CONFIG
        ))


if __name__ == "__main__":
    unittest.main()
