import copy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from src.data.week8_unlabeled_pool import (
    PROTOCOL, ROW_FIELDS, build_pool, choose_source_identities, validate_pool_images, verified_pool, validate_pool_snapshot,
)
from src.data.week8_visual_holdout import build_holdout, read_json, validate_holdout
from src.inference.product_observation import canonical_config_sha256
from src.training.week7_data import IDENTITY_FIELDS, sha256_file


class UnlabeledPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.consumed = {key: set() for key in IDENTITY_FIELDS}
        self.history = {"files": [], "week8_history": [{"path": "failed-final", "scope": "identity_fields_only"}]}

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def image(self, name, color):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(path)
        return path

    def test_selection_ignores_caption_labels_and_input_order(self):
        rows = [{"photo_id": str(i), "business_id": str(i // 2), "caption": "hidden", "label": "food"} for i in range(12)]
        groups = {str(i) for i in range(6)}
        first, _ = choose_source_identities(iter(rows), groups, self.consumed, 4, "seed")
        second, _ = choose_source_identities(({**row, "caption": "changed", "label": "inside"} for row in reversed(rows)),
                                              groups, self.consumed, 4, "seed")
        self.assertEqual(first, second)
        self.assertEqual(len({row["group_id"] for row in first}), 4)
        self.assertTrue(all(set(row) == {"source_id", "group_id", "photo_id"} for row in first))

    def test_selection_rejects_historical_sources_groups_and_bad_identifiers(self):
        self.consumed["source_id"].add("yelp-photo:p1")
        self.consumed["group_id"].add("g2")
        rows = [{"photo_id": "p1", "business_id": "g1"}, {"photo_id": "p2", "business_id": "g2"},
                {"photo_id": "../../escape", "business_id": "g3"}, {"photo_id": "p4", "business_id": "g4"},
                {"photo_id": "p5", "business_id": "g5"}]
        selected, counts = choose_source_identities(rows, {"g1", "g2", "g3", "g4"}, self.consumed, 5, "seed")
        self.assertEqual([row["photo_id"] for row in selected], ["p4"])
        self.assertEqual(counts["historical_source_or_group"], 2)
        self.assertEqual(counts["invalid_identity"], 1)
        self.assertEqual(counts["outside_ota_scope"], 1)
        for invalid in (0, -1, 10001, True):
            with self.assertRaises(ValueError):
                choose_source_identities(rows, set(), self.consumed, invalid, "seed")

    def test_image_validation_keeps_rejections_not_silent_replacements(self):
        red = self.image("pool/raw/photos/p1.jpg", "red")
        self.image("pool/raw/photos/p2.jpg", "blue")
        self.image("pool/raw/photos/p3.jpg", "blue")
        (self.root / "pool/raw/photos/p4.jpg").write_bytes(b"not an image")
        self.consumed["image_sha256"].add(sha256_file(red))
        candidates = [{"photo_id": f"p{i}", "source_id": f"yelp-photo:p{i}", "group_id": f"g{i}"} for i in range(1, 6)]
        accepted, rejected = validate_pool_images(self.root, self.root / "pool", candidates, self.consumed, "pool-v1")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(set(accepted[0]), ROW_FIELDS)
        self.assertIsNone(accepted[0]["constraint_template_id"])
        self.assertEqual([row["reason"] for row in rejected], ["historical_image_hash", "duplicate_image_hash",
                                                               "unreadable_image", "missing_archive_image"])

    def prepare_build(self):
        images = [self.image(f"source/{i}.jpg", color) for i, color in enumerate(("red", "blue", "green"))]
        tar_data = io.BytesIO()
        with tarfile.open(fileobj=tar_data, mode="w:gz") as archive:
            for i, image in enumerate(images):
                archive.add(image, arcname=f"photos/p{i}.jpg")
        archive_path = self.root / "existing.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("Yelp Photos/photos.tar", tar_data.getvalue())
        business, photos = self.put("business.parquet", []), self.put("photos.parquet", [])
        config = {"protocol": PROTOCOL, "dataset_version": "pool-v1", "output_root": "pool", "seed": "seed",
                  "maximum_candidate_images": 3, "minimum_accepted_images": 3, "maximum_business_groups": 10,
                  "labels_generated": False, "human_annotation_count": 0, "archive_environment_variable": "TRIP_POOL_TEST_ARCHIVE",
                  "archive_size": archive_path.stat().st_size, "archive_sha256": sha256_file(archive_path), "history": {},
                  "photos": {"path": "photos.parquet", "sha256": sha256_file(photos)},
                  "business": {"path": "business.parquet", "sha256": sha256_file(business)}}
        config_path = self.put("pool-config.json", config)
        audit = {"source": {"fresh_identity_path": "unused.jsonl"}, "category_terms": {
            "existing": {"hotel": [], "attraction": [], "restaurant": ["restaurants"]},
            "strict_expansion": {"hotel": [], "attraction": [], "restaurant": []}}}
        self.put("unused.jsonl", {})
        fresh = {"identity_manifest": {"sha256": "old-source"}}
        def rows(path, columns):
            if columns == ["business_id", "categories"]:
                return iter([{"business_id": f"g{i}", "categories": ["Restaurants"]} for i in range(3)])
            self.assertEqual(columns, ["photo_id", "business_id"])
            return iter([{"photo_id": f"p{i}", "business_id": f"g{i}"} for i in range(3)])
        self.enterContext(patch.dict(os.environ, {"TRIP_POOL_TEST_ARCHIVE": str(archive_path)}))
        self.enterContext(patch("src.data.week8_unlabeled_pool.parquet_rows", side_effect=rows))
        self.enterContext(patch("src.data.week8_unlabeled_pool.load_history", return_value=(self.consumed, self.history, audit, fresh)))
        self.enterContext(patch("src.data.week8_unlabeled_pool.source_hashes", return_value={"source": "stable"}))
        self.enterContext(patch("src.data.week8_visual_holdout.load_history", return_value=(self.consumed, self.history, audit, fresh)))
        return config_path

    def test_build_archive_pool_and_new_holdout_keep_identity_binding(self):
        config_path = self.prepare_build()
        result = build_pool(self.root, config_path)
        self.assertEqual(result["status"], "PASS")
        declaration = {"config": "pool-config.json", "lock_sha256": result["lock_sha256"]}
        rows, lock = verified_pool(self.root, declaration, self.history)
        self.assertEqual(len(rows), 3)
        self.assertFalse(lock["labels_generated"])
        with self.assertRaises(FileExistsError):
            build_pool(self.root, config_path)
        final = {"protocol": "test", "dataset_version": "new-final", "output_root": "final", "sample_count": 2,
                 "seed": "final-seed", "template_id": None, "human_annotation_count": 0,
                 "label_source": "model_generated_silver", "no_tuning_after_final": True, "unlabeled_source_pool": declaration}
        final_path = self.put("final.json", final)
        sealed = build_holdout(self.root, final_path)
        self.assertEqual(sealed["count"], 2)
        selected, final_lock = validate_holdout(self.root, final)
        self.assertEqual(final_lock["source_identity_sha256"], lock["files"]["identity_manifest.jsonl"])
        self.consumed["group_id"].add(selected[0]["group_id"])
        with self.assertRaisesRegex(ValueError, "historical"):
            validate_holdout(self.root, final)

    def test_pool_refuses_changed_archive_history_or_artifact(self):
        path = self.prepare_build()
        config = read_json(path)
        config["archive_sha256"] = "bad"
        self.put("bad-config.json", config)
        with self.assertRaisesRegex(ValueError, "archive identity"):
            build_pool(self.root, self.root / "bad-config.json")
        result = build_pool(self.root, path)
        declaration = {"config": "pool-config.json", "lock_sha256": result["lock_sha256"]}
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verified_pool(self.root, declaration, {**self.history, "later_consumed": True})
        (self.root / "pool/identity_manifest.jsonl").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verified_pool(self.root, declaration, self.history)

    def test_insufficient_pool_is_preserved_but_cannot_supply_final(self):
        path = self.prepare_build()
        config = read_json(path)
        config["minimum_accepted_images"] = 4
        self.put("pool-config.json", config)
        result = build_pool(self.root, path)
        self.assertEqual(result["status"], "INSUFFICIENT_SOURCE")
        with self.assertRaisesRegex(ValueError, "insufficient"):
            verified_pool(self.root, {"config": "pool-config.json", "lock_sha256": result["lock_sha256"]}, self.history)

    def test_portable_snapshot_rejects_changed_members_or_unrelated_final(self):
        path = self.prepare_build()
        result = build_pool(self.root, path)
        rows, lock = verified_pool(self.root, {"config": "pool-config.json", "lock_sha256": result["lock_sha256"]}, self.history)
        data = {"unlabeled_source_pool_lock_sha256": lock["lock_sha256"], "source_identity_sha256": lock["files"]["identity_manifest.jsonl"]}
        read = lambda name: (self.root / "pool" / name).read_text(encoding="utf-8")
        digest = lambda name: sha256_file(self.root / "pool" / name)
        validate_pool_snapshot(data, rows[:1], lock, read, digest)
        unrelated = copy.deepcopy(rows[:1])
        unrelated[0]["image_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "does not belong"):
            validate_pool_snapshot(data, unrelated, lock, read, digest)
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            validate_pool_snapshot(data, rows[:1], lock, read, lambda name: "bad")

    def test_snapshot_rejects_missing_rejection_audit_even_if_lock_rehashed(self):
        path = self.prepare_build()
        result = build_pool(self.root, path)
        rows, lock = verified_pool(self.root, {"config": "pool-config.json", "lock_sha256": result["lock_sha256"]}, self.history)
        del lock["files"]["rejections.jsonl"]
        lock["lock_sha256"] = canonical_config_sha256({k: v for k, v in lock.items() if k != "lock_sha256"})
        data = {"unlabeled_source_pool_lock_sha256": lock["lock_sha256"], "source_identity_sha256": lock["files"]["identity_manifest.jsonl"]}
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            validate_pool_snapshot(data, rows[:1], lock, lambda name: "", lambda name: lock["files"][name])
