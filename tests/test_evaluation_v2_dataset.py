import copy
import unittest
from pathlib import Path

from src.evaluation.v2_dataset import _curate_after_sales, _version_record


def record(sample_id, stratum, source_type, issue_type=None, status="completed"):
    return {
        "sample_id": sample_id,
        "sampling_stratum": stratum,
        "source_type": source_type,
        "annotation_status": status,
        "annotator": "annotator-a" if status == "completed" else None,
        "annotation": {"issue_type": issue_type} if status == "completed" else None,
        "dataset_version": "week3_evaluation_v1",
        "review_status": "pending",
        "reviewer": None,
        "notes": "original",
    }


class V2DatasetTest(unittest.TestCase):
    def test_version_record_does_not_mutate_v1(self):
        original = record("one", "hotel", "public_yelp", "unknown")
        snapshot = copy.deepcopy(original)
        updated = _version_record(original, completed=False, note="v2")
        self.assertEqual(original, snapshot)
        self.assertEqual(updated["dataset_version"], "week3_evaluation_v2")
        self.assertEqual(updated["annotation_status"], "pending")
        self.assertIsNone(updated["annotation"])

    def test_after_sales_keeps_supported_v1_and_fills_exact_strata(self):
        frozen = []
        for index in range(6):
            frozen.append(record(f"public-{index}", "hygiene_stain", "public_yelp", "hygiene_stain"))
        for stratum in ("attraction_closure", "transport_delay"):
            for index in range(37):
                frozen.append(record(f"{stratum}-{index}", stratum, "business_synthetic", stratum))
        frozen.append(record("bad", "facility_damage", "public_yelp", "unknown"))
        replacements = []
        for stratum, count in {
            "hygiene_stain": 38,
            "facility_damage": 38,
            "attraction_closure": 37,
            "transport_delay": 37,
        }.items():
            for index in range(count):
                replacements.append(record(f"new-{stratum}-{index}", stratum, "business_synthetic", status="pending"))
        curated, removed = _curate_after_sales(frozen, replacements)
        self.assertEqual(len(curated), 150)
        self.assertEqual(removed, ["bad"])
        self.assertEqual(sum(row["annotation_status"] == "completed" for row in curated), 80)
        self.assertEqual(sum(row["annotation_status"] == "pending" for row in curated), 70)
        self.assertEqual({row["source_type"] for row in curated}, {"public_yelp", "business_synthetic"})


if __name__ == "__main__":
    unittest.main()
