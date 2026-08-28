import unittest

from src.data.week8_visual_holdout import choose_untouched, identity_projection
from src.training.week7_data import IDENTITY_FIELDS


class VisualHoldoutTests(unittest.TestCase):
    def test_retrieval_raw_group_and_product_namespaced_group_collide(self):
        self.assertEqual(identity_projection({"group_id": "abc"})["group_id"],
                         identity_projection({"group_id": "yelp-business:abc"})["group_id"])

    def test_five_dimension_exclusion_includes_older_versions(self):
        consumed = {key: set() for key in IDENTITY_FIELDS}
        rows = [{"source_id": str(i), "group_id": "yelp-business:g" + str(i),
                 "image_sha256": str(i) * 64, "image_path": str(i) + ".jpg"} for i in range(6)]
        consumed["group_id"].add("g0")
        consumed["image_sha256"].add("1" * 64)
        consumed["source_id"].add("2")
        chosen, audit = choose_untouched(rows, consumed, 3, "seed", "new-image-only-template")
        self.assertEqual({row["source_id"] for row in chosen}, {"3", "4", "5"})
        self.assertEqual(audit["rejected_count"], 3)
        consumed["constraint_template_id"].add("new-image-only-template")
        with self.assertRaises(ValueError):
            choose_untouched(rows, consumed, 3, "seed", "new-image-only-template")

    def test_fixed_selection_cannot_drop_samples_or_duplicate_groups(self):
        consumed = {key: set() for key in IDENTITY_FIELDS}
        rows = [{"source_id": str(i), "group_id": "same", "image_sha256": str(i) * 64, "image_path": "a.jpg"} for i in range(2)]
        with self.assertRaises(ValueError):
            choose_untouched(rows, consumed, 3, "seed", "template")
        with self.assertRaises(ValueError):
            choose_untouched(rows, consumed, 2, "seed", "template")

    def test_selection_is_independent_of_labels_and_input_order(self):
        consumed = {key: set() for key in IDENTITY_FIELDS}
        rows = [{"source_id": str(i), "group_id": str(i), "image_sha256": str(i) * 64,
                 "image_path": "a.jpg", "target": "secret"} for i in range(8)]
        first, _ = choose_untouched(rows, consumed, 4, "seed", "template")
        second, _ = choose_untouched([{**row, "target": "changed"} for row in reversed(rows)], consumed, 4, "seed", "template")
        self.assertEqual(first, second)
        self.assertNotIn("target", str(first))


if __name__ == "__main__":
    unittest.main()
