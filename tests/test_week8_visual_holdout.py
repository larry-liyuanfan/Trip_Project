import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data.week8_visual_holdout import choose_untouched, identity_projection, load_history
from src.training.week7_data import IDENTITY_FIELDS, sha256_file
from src.inference.product_observation import canonical_config_sha256


class VisualHoldoutTests(unittest.TestCase):
    def test_failed_visual_final_still_excludes_every_identity_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def put(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
            put("audit.json", {"source": {"product_config_path": "source.json"}})
            put("source.json", {"dataset": {"source_paths": {}}})
            files = {}
            for name in ("train/image_product_search.jsonl", "development/image_product_search.jsonl"):
                path = root / "training" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
                files[name] = {"sha256": sha256_file(path), "count": 0}
            put("training/dataset_lock.json", {"files": files})
            row = {"sample_id": "old-final", "source_id": "source", "group_id": "yelp-business:group",
                   "image_sha256": "a" * 64, "constraint_template_id": None, "target": "must_not_be_used"}
            put("previous/manifest.jsonl", row)
            lock = {"manifest_sha256": sha256_file(root / "previous/manifest.jsonl"), "count": 1}
            lock["lock_sha256"] = canonical_config_sha256(lock)
            put("previous/dataset_lock.json", lock)
            config = {"source_audit_config": "audit.json", "previous_product_versions": [], "previous_retrieval_versions": [],
                      "previous_training_version": "training", "previous_visual_holdouts": [{"path": "previous", "lock_sha256": lock["lock_sha256"]}]}
            consumed = {key: set() for key in IDENTITY_FIELDS}
            evidence = {"files": [], "superseded_week7_identity_manifests": []}
            fresh = {"historical_exclusion_evidence": evidence.copy()}
            with patch("src.data.week8_visual_holdout._verify_identity", return_value=(fresh, None, None)), \
                 patch("src.data.week8_visual_holdout.load_consumed_identities", return_value=(consumed, evidence)), \
                 patch("src.data.week8_visual_holdout.add_superseded_identities"):
                result, audit, _, _ = load_history(root, config)
                self.assertEqual(result["group_id"], {"group"})
                self.assertEqual(result["source_id"], {"source"})
                self.assertEqual(result["sample_id"], {"old-final"})
                self.assertEqual(result["image_sha256"], {"a" * 64})
                self.assertEqual(result["constraint_template_id"], set())
                self.assertNotIn("must_not_be_used", str(audit))
                (root / "previous/manifest.jsonl").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    load_history(root, config)

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

    def test_untemplated_photos_remain_null_not_artificially_isolated(self):
        consumed = {key: set() for key in IDENTITY_FIELDS}
        rows = [{"source_id": "s", "group_id": "g", "image_sha256": "a" * 64, "image_path": "a.jpg"}]
        chosen, audit = choose_untouched(rows, consumed, 1, "seed", None)
        self.assertIsNone(chosen[0]["constraint_template_id"])
        self.assertEqual(audit["template_identity_status"], "not_applicable_untemplated_images")
        rows[0]["constraint_template_id"] = "real-template"
        with self.assertRaises(ValueError):
            choose_untouched(rows, consumed, 1, "seed", None)


if __name__ == "__main__":
    unittest.main()
