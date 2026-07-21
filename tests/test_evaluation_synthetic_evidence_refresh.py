import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.build_week3_candidate_manifests import _candidate_recipe_versions
from scripts.manage_week3_annotations import export_packet
from scripts.refresh_week3_synthetic_evidence import run_refresh
from src.evaluation.manifests import (
    ManifestValidationError,
    build_exclusion_rows,
    write_jsonl,
)
from src.evaluation.synthetic_evidence_refresh import (
    execute_synthetic_evidence_refresh,
    plan_synthetic_evidence_refresh,
)


class SyntheticEvidenceRefreshTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config = {
            "dataset_version": "week3_evaluation_v1",
            "paths": {
                "exclusion_manifest": "data/eval/registry/evaluation_exclusion_manifest.jsonl",
                "codings_dir": "data/eval/codings",
                "sampling_logs_dir": "data/eval/logs",
            },
            "candidate_sources": {
                "synthetic_recipe_version": "week3_business_evidence_v1",
                "after_sales_synthetic_recipe_version": "week3_after_sales_evidence_v2",
            },
            "scenarios": {
                "image_product_search": {
                    "manifest_path": "data/eval/manifests/image_product_search_v1.jsonl",
                    "target_count": 1,
                    "sampling": {"quotas": {"hotel": 1}},
                },
                "after_sales": {
                    "manifest_path": "data/eval/manifests/after_sales_v1.jsonl",
                    "target_count": 2,
                    "sampling": {
                        "quotas": {
                            "hygiene_stain": 0,
                            "facility_damage": 0,
                            "attraction_closure": 1,
                            "transport_delay": 1,
                        }
                    },
                },
                "itinerary_planning": {
                    "manifest_path": "data/eval/manifests/itinerary_planning_v1.jsonl",
                    "target_count": 1,
                    "sampling": {"quotas": {"paired_reference_and_text": 1}},
                },
            },
        }
        self.records = {
            "image_product_search": [
                self._record(
                    scenario="image_product_search",
                    sample_id="product-1",
                    source_id="yelp:product:1",
                    image_path="data/eval/images/image_product_search/product.png",
                    color=(30, 60, 90),
                    sampling_stratum="hotel",
                )
            ],
            "after_sales": [
                self._record(
                    scenario="after_sales",
                    sample_id="after-closure",
                    source_id="synthetic:attraction_closure:0000",
                    image_path="data/eval/images/after_sales/attraction_closure_0000.png",
                    color=(120, 40, 40),
                    sampling_stratum="attraction_closure",
                    source_type="business_synthetic",
                    recipe_version="week3_business_evidence_v1",
                ),
                self._record(
                    scenario="after_sales",
                    sample_id="after-delay",
                    source_id="synthetic:transport_delay:0000",
                    image_path="data/eval/images/after_sales/transport_delay_0000.png",
                    color=(180, 120, 20),
                    sampling_stratum="transport_delay",
                    source_type="business_synthetic",
                    recipe_version="week3_business_evidence_v1",
                ),
            ],
            "itinerary_planning": [
                self._record(
                    scenario="itinerary_planning",
                    sample_id="itinerary-1",
                    source_id="yelp:itinerary:1",
                    image_path="data/eval/images/itinerary_planning/itinerary.png",
                    color=(40, 120, 70),
                    sampling_stratum="paired_reference_and_text",
                    text_constraints="2 days; budget 1000",
                    recipe_version="week3_business_evidence_v1",
                )
            ],
        }
        self._write_live_artifacts()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _record(
        self,
        *,
        scenario,
        sample_id,
        source_id,
        image_path,
        color,
        sampling_stratum,
        source_type="public_yelp",
        text_constraints=None,
        recipe_version=None,
    ):
        path = self.root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=color).save(path, format="PNG")
        image_bytes = path.read_bytes()
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        perceptual_hash = f"{int.from_bytes(bytes(color), 'big'):016x}"[-16:]
        return {
            "sample_id": sample_id,
            "scenario": scenario,
            "source_type": source_type,
            "source_id": source_id,
            "source_license": "test-license",
            "image_sha256": image_sha256,
            "input": {
                "images": [
                    {
                        "path": image_path,
                        "sha256": image_sha256,
                        "perceptual_hash": perceptual_hash,
                    }
                ],
                "text_constraints": text_constraints,
            },
            "split": "evaluation",
            "dataset_version": "week3_evaluation_v1",
            "annotation_status": "pending",
            "annotator": None,
            "review_status": "pending",
            "reviewer": None,
            "file_status": "valid",
            "annotation": None,
            "notes": "fixture",
            "sampling_stratum": sampling_stratum,
            "provenance": {
                "source_uri": f"fixture://{source_id}",
                "source_version": recipe_version or "fixture-v1",
                "group_id": f"group:{source_id}",
                "synthetic_recipe_version": recipe_version,
                "constraint_template_id": None,
                "pii_review_status": "not_applicable",
            },
        }

    def _write_live_artifacts(self):
        for scenario, records in self.records.items():
            path = self.root / self.config["scenarios"][scenario]["manifest_path"]
            write_jsonl(path, records)
        all_records = [record for records in self.records.values() for record in records]
        write_jsonl(
            self.root / self.config["paths"]["exclusion_manifest"],
            build_exclusion_rows(all_records),
        )
        codings = self.root / self.config["paths"]["codings_dir"]
        write_jsonl(
            codings / "after_sales_annotation_packet.jsonl",
            export_packet(
                self.records["after_sales"],
                scenario="after_sales",
                stage="annotation",
            ),
        )
        write_jsonl(
            codings / "after_sales_annotation_suggested.jsonl",
            export_packet(
                self.records["after_sales"],
                scenario="after_sales",
                stage="annotation",
                include_suggestions=True,
            ),
        )

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def _live_target_snapshot(self):
        paths = [
            self.root / image["path"]
            for record in self.records["after_sales"]
            for image in record["input"]["images"]
        ]
        paths.extend(
            [
                self.root / self.config["scenarios"]["after_sales"]["manifest_path"],
                self.root / self.config["paths"]["exclusion_manifest"],
                self.root / "data/eval/codings/after_sales_annotation_packet.jsonl",
                self.root / "data/eval/codings/after_sales_annotation_suggested.jsonl",
            ]
        )
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in paths
        }

    def test_plan_preserves_identity_and_updates_only_v2_fields(self):
        originals = copy.deepcopy(self.records["after_sales"])

        plan = plan_synthetic_evidence_refresh(
            root=self.root,
            config=self.config,
            run_id="refresh-001",
        )

        planned = plan.manifests["after_sales"]
        self.assertEqual([row["sample_id"] for row in planned], ["after-closure", "after-delay"])
        self.assertEqual([row["source_id"] for row in planned], [row["source_id"] for row in originals])
        self.assertEqual(len(plan.images), 2)
        for before, after in zip(originals, planned):
            self.assertEqual(after["annotation_status"], "pending")
            self.assertEqual(after["review_status"], "pending")
            self.assertEqual(after["sampling_stratum"], before["sampling_stratum"])
            self.assertNotEqual(after["image_sha256"], before["image_sha256"])
            self.assertEqual(after["image_sha256"], after["input"]["images"][0]["sha256"])
            self.assertNotEqual(
                after["input"]["images"][0]["perceptual_hash"],
                before["input"]["images"][0]["perceptual_hash"],
            )
            self.assertEqual(
                after["provenance"]["source_version"],
                "week3_after_sales_evidence_v2",
            )
            self.assertEqual(
                after["provenance"]["synthetic_recipe_version"],
                "week3_after_sales_evidence_v2",
            )

    def test_plan_rejects_completed_annotation_before_writing(self):
        record = self.records["after_sales"][0]
        record.update(
            annotation_status="completed",
            annotator="human-a",
            annotation={
                "issue_type": "attraction_closure",
                "severity": "medium",
                "key_information": [],
                "ocr_ground_truth": [],
            },
        )
        self._write_live_artifacts()
        before = self._snapshot()

        with self.assertRaisesRegex(ManifestValidationError, "completed annotation"):
            plan_synthetic_evidence_refresh(
                root=self.root,
                config=self.config,
                run_id="refresh-001",
            )

        self.assertEqual(self._snapshot(), before)

    def test_plan_rejects_target_ui_draft_before_writing(self):
        write_jsonl(
            self.root / "data/eval/codings/ui_drafts/annotation-human.jsonl",
            [
                {
                    "sample_id": "after-closure",
                    "stage": "annotation",
                    "actor": "human-a",
                    "payload": {},
                }
            ],
        )
        before = self._snapshot()

        with self.assertRaisesRegex(ManifestValidationError, "UI draft.*after-closure"):
            plan_synthetic_evidence_refresh(
                root=self.root,
                config=self.config,
                run_id="refresh-001",
            )

        self.assertEqual(self._snapshot(), before)

    def test_plan_rejects_nonpending_review_before_writing(self):
        record = self.records["after_sales"][0]
        record.update(
            annotation_status="completed",
            annotator="human-a",
            annotation={
                "issue_type": "attraction_closure",
                "severity": "medium",
                "key_information": [],
                "ocr_ground_truth": [],
            },
            review_status="validated",
            reviewer="human-b",
        )
        self._write_live_artifacts()
        before = self._snapshot()

        with self.assertRaisesRegex(ManifestValidationError, "non-pending review"):
            plan_synthetic_evidence_refresh(
                root=self.root,
                config=self.config,
                run_id="refresh-001",
            )

        self.assertEqual(self._snapshot(), before)

    def test_plan_rejects_wrong_target_count(self):
        self.records["after_sales"].pop()
        self._write_live_artifacts()

        with self.assertRaisesRegex(ManifestValidationError, "target count"):
            plan_synthetic_evidence_refresh(
                root=self.root,
                config=self.config,
                run_id="refresh-001",
            )

    def test_plan_rejects_wrong_source_pattern(self):
        self.records["after_sales"][0]["source_id"] = "synthetic:wrong:0000"
        self._write_live_artifacts()

        with self.assertRaisesRegex(ManifestValidationError, "source_id pattern"):
            plan_synthetic_evidence_refresh(
                root=self.root,
                config=self.config,
                run_id="refresh-001",
            )

    def test_plan_rejects_manually_filled_annotation_packet(self):
        packet_path = self.root / "data/eval/codings/after_sales_annotation_packet.jsonl"
        rows = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["annotation"]["issue_type"] = "attraction_closure"
        write_jsonl(packet_path, rows)
        before = self._snapshot()

        with self.assertRaisesRegex(ManifestValidationError, "manual packet content"):
            plan_synthetic_evidence_refresh(
                root=self.root,
                config=self.config,
                run_id="refresh-001",
            )

        self.assertEqual(self._snapshot(), before)

    def test_check_only_plan_creates_no_files(self):
        before = self._snapshot()

        plan_synthetic_evidence_refresh(
            root=self.root,
            config=self.config,
            run_id="refresh-001",
        )

        self.assertEqual(self._snapshot(), before)

    def test_execute_backs_up_and_replaces_every_affected_artifact(self):
        before = self._live_target_snapshot()
        plan = plan_synthetic_evidence_refresh(
            root=self.root,
            config=self.config,
            run_id="refresh-001",
        )

        result = execute_synthetic_evidence_refresh(
            plan,
            root=self.root,
            config=self.config,
        )

        after = self._live_target_snapshot()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["target_count"], 2)
        self.assertEqual(result["recipe_version"], "week3_after_sales_evidence_v2")
        self.assertNotEqual(after, before)
        backup_root = self.root / result["backup_path"]
        for relative_path, original_bytes in before.items():
            self.assertEqual((backup_root / relative_path).read_bytes(), original_bytes)
        audit = json.loads((self.root / result["audit_path"]).read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(len(audit["images"]), 2)
        self.assertFalse((self.root / "data/eval/.staging/refresh-001").exists())

    def test_execute_rolls_back_all_replaced_artifacts_on_failure(self):
        before = self._live_target_snapshot()
        plan = plan_synthetic_evidence_refresh(
            root=self.root,
            config=self.config,
            run_id="refresh-001",
        )
        calls = 0

        def fail_on_fourth(source, destination):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError("injected replacement failure")
            os.replace(source, destination)

        with self.assertRaisesRegex(ManifestValidationError, "rolled back"):
            execute_synthetic_evidence_refresh(
                plan,
                root=self.root,
                config=self.config,
                replace=fail_on_fourth,
            )

        self.assertEqual(self._live_target_snapshot(), before)
        self.assertFalse((self.root / "data/eval/.staging/refresh-001").exists())
        self.assertFalse(
            (self.root / "data/eval/logs/after_sales_synthetic_refresh_v2.json").exists()
        )

    def test_execute_rejects_live_drift_before_creating_backup(self):
        plan = plan_synthetic_evidence_refresh(
            root=self.root,
            config=self.config,
            run_id="refresh-001",
        )
        changed_path = self.root / plan.images[0].relative_path
        changed_path.write_bytes(changed_path.read_bytes() + b"changed")

        with self.assertRaisesRegex(ManifestValidationError, "changed after refresh planning"):
            execute_synthetic_evidence_refresh(
                plan,
                root=self.root,
                config=self.config,
            )

        self.assertFalse(
            (self.root / "data/eval/backups/synthetic-evidence-v1-refresh-001").exists()
        )

    def test_check_only_cli_service_returns_ready_without_writes(self):
        before = self._snapshot()

        result = run_refresh(
            config=self.config,
            root=self.root,
            run_id="refresh-001",
            check_only=True,
        )

        self.assertEqual(
            result,
            {
                "status": "ready",
                "run_id": "refresh-001",
                "recipe_version": "week3_after_sales_evidence_v2",
                "target_count": 2,
            },
        )
        self.assertEqual(self._snapshot(), before)

    def test_full_builder_uses_separate_itinerary_and_after_sales_recipes(self):
        itinerary, after_sales = _candidate_recipe_versions(
            self.config["candidate_sources"]
        )

        self.assertEqual(itinerary, "week3_business_evidence_v1")
        self.assertEqual(after_sales, "week3_after_sales_evidence_v2")


if __name__ == "__main__":
    unittest.main()
