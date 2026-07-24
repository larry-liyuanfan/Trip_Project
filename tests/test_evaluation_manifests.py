import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import scripts.prepare_week3_evaluation as prepare_week3_evaluation
from scripts.prepare_week3_evaluation import run_candidate_sampling
from scripts.validate_week3_evaluation import validate_configured_manifests
from src.evaluation.config import load_evaluation_config
from src.evaluation.manifests import (
    EvaluationCollisionError,
    ManifestValidationError,
    build_exclusion_rows,
    is_release_eligible,
    load_manifest,
    reject_evaluation_collisions,
    summarize_counts,
    validate_release_provenance,
    validate_manifest_record,
    write_jsonl,
)
from src.evaluation.sampling import stratified_sample


class EvaluationManifestTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def _record(self, scenario="image_product_search", sample_id="sample-001"):
        image_hash = hashlib.sha256(f"{scenario}:{sample_id}".encode("utf-8")).hexdigest()
        image_path = f"data/eval/images/{sample_id}.jpg"
        return {
            "sample_id": sample_id,
            "scenario": scenario,
            "source_type": "synthetic_fixture",
            "source_id": f"source-{sample_id}",
            "source_license": "synthetic-test-only",
            "image_sha256": image_hash,
            "input": {
                "images": [{"path": image_path, "sha256": image_hash}],
                "text_constraints": "两天行程，第二天18:00前结束" if scenario == "itinerary_planning" else None,
            },
            "split": "evaluation",
            "dataset_version": "week3-eval-v1",
            "annotation_status": "pending",
            "annotator": None,
            "review_status": "pending",
            "reviewer": None,
            "file_status": "pending",
            "annotation": None,
            "notes": None,
        }

    def test_pending_image_product_record_matches_common_contract(self):
        record = {
            "sample_id": "image-product-001",
            "scenario": "image_product_search",
            "source_type": "synthetic_fixture",
            "source_id": "source-001",
            "source_license": "synthetic-test-only",
            "image_sha256": "a" * 64,
            "input": {
                "images": [
                    {
                        "path": "data/eval/images/image-product-001.jpg",
                        "sha256": "a" * 64,
                    }
                ],
                "text_constraints": None,
            },
            "split": "evaluation",
            "dataset_version": "week3-eval-v1",
            "annotation_status": "pending",
            "annotator": None,
            "review_status": "pending",
            "reviewer": None,
            "file_status": "pending",
            "annotation": None,
            "notes": None,
        }

        validated = validate_manifest_record(record)

        self.assertEqual(validated, record)

    def test_pending_record_requires_inference_input(self):
        record = self._record()
        del record["input"]

        with self.assertRaisesRegex(ManifestValidationError, "input"):
            validate_manifest_record(record)

    def test_itinerary_input_requires_raw_text_constraints(self):
        record = self._record("itinerary_planning")
        record["input"]["text_constraints"] = None

        with self.assertRaisesRegex(ManifestValidationError, "text_constraints"):
            validate_manifest_record(record)

    def test_completed_annotations_validate_for_each_scenario(self):
        annotations = {
            "image_product_search": {
                "business_category": "hotel",
                "style_tags": ["modern"],
                "visible_facilities": ["pool"],
                "price_range": "premium",
            },
            "after_sales": {
                "issue_type": "facility_damage",
                "severity": "high",
                "key_information": ["窗户破损"],
                "ocr_ground_truth": None,
            },
            "itinerary_planning": {
                "reference_images": ["data/samples/images/cafe_001.jpg"],
                "text_constraints": ["两天"],
                "style_preferences": ["慢节奏"],
                "hard_constraints": ["第二天18:00前结束"],
                "soft_constraints": ["优先咖啡馆"],
                "required_itinerary_elements": ["交通方式"],
            },
        }

        for index, (scenario, annotation) in enumerate(annotations.items()):
            with self.subTest(scenario=scenario):
                record = self._record(scenario, f"completed-{index}")
                record.update(
                    annotation_status="completed",
                    annotator="human-reviewer-01",
                    review_status="validated",
                    reviewer="independent-reviewer-01",
                    file_status="valid",
                    annotation=annotation,
                )
                self.assertEqual(validate_manifest_record(record), record)

    def test_rejects_invalid_completed_annotation(self):
        record = self._record("after_sales")
        record.update(
            annotation_status="completed",
            annotator="human-reviewer-01",
            file_status="valid",
            annotation={
                "issue_type": "facility_damage",
                "key_information": [],
                "ocr_ground_truth": None,
            },
        )

        with self.assertRaisesRegex(ManifestValidationError, "severity"):
            validate_manifest_record(record)

    def test_rejects_invalid_image_hash_for_image_scenario(self):
        record = self._record()
        record["image_sha256"] = "not-a-sha256"

        with self.assertRaisesRegex(ManifestValidationError, "image_sha256"):
            validate_manifest_record(record)

    def test_manifest_validation_checks_hash_against_actual_image_bytes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "data" / "eval" / "images" / "sample.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"synthetic image bytes")
            expected_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            record = self._record(sample_id="byte-checked")
            record["image_sha256"] = expected_hash
            record["input"]["images"] = [
                {"path": "data/eval/images/sample.jpg", "sha256": expected_hash}
            ]

            self.assertEqual(validate_manifest_record(record, root=root), record)

            wrong_hash = "0" * 64
            record["image_sha256"] = wrong_hash
            record["input"]["images"][0]["sha256"] = wrong_hash
            with self.assertRaisesRegex(ManifestValidationError, "image bytes"):
                validate_manifest_record(record, root=root)

    def test_rejects_validated_review_without_completed_annotation_and_valid_file(self):
        record = self._record()
        record["review_status"] = "validated"

        with self.assertRaisesRegex(ManifestValidationError, "validated review"):
            validate_manifest_record(record)

    def test_validated_review_requires_distinct_independent_reviewer(self):
        record = self._record()
        record.update(
            annotation_status="completed",
            annotator="human-01",
            review_status="validated",
            reviewer="human-01",
            file_status="valid",
            annotation={
                "business_category": "hotel",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
            },
        )

        with self.assertRaisesRegex(ManifestValidationError, "independent reviewer"):
            validate_manifest_record(record)

    def test_load_manifest_rejects_duplicate_sample_ids(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            record = self._record()
            path.write_text(
                "\n".join(json.dumps(record) for _ in range(2)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ManifestValidationError, "duplicate sample_id"):
                load_manifest(path)

    def test_summarize_counts_keeps_five_statuses_distinct(self):
        pending = self._record(sample_id="pending")
        annotated = self._record(sample_id="annotated")
        annotated.update(
            annotation_status="completed",
            annotator="human-reviewer-01",
            file_status="valid",
            annotation={
                "business_category": "restaurant",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
            },
            provenance={
                "source_uri": "fixture://annotated",
                "source_version": "fixture-v1",
                "group_id": "group-annotated",
                "synthetic_recipe_version": None,
                "constraint_template_id": None,
                "pii_review_status": "verified",
            },
        )
        validated = self._record(sample_id="validated")
        validated.update(
            annotation_status="completed",
            annotator="human-reviewer-02",
            review_status="validated",
            reviewer="independent-reviewer-02",
            file_status="valid",
            annotation={
                "business_category": "attraction",
                "style_tags": ["historic"],
                "visible_facilities": [],
                "price_range": "unknown",
            },
            provenance={
                "source_uri": "fixture://validated",
                "source_version": "fixture-v1",
                "group_id": "group-validated",
                "synthetic_recipe_version": None,
                "constraint_template_id": None,
                "pii_review_status": "verified",
            },
        )

        counts = summarize_counts(
            [pending, annotated, validated],
            target_count=200,
            tested_sample_ids={"annotated", "validated", "not-in-manifest"},
        )

        self.assertEqual(
            counts,
            {
                "target_count": 200,
                "candidate_count": 3,
                "annotated_count": 2,
                "validated_count": 2,
                "tested_count": 2,
            },
        )

    def test_summarize_counts_keeps_frozen_unknown_and_pending_pii_eligible(self):
        record = self._record(sample_id="pii-pending")
        record.update(
            annotation_status="completed",
            annotator="human-a",
            review_status="validated",
            reviewer="legacy-reviewer",
            file_status="valid",
            annotation={
                "business_category": "hotel",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
            },
            provenance={
                "source_uri": "fixture://pii-pending",
                "source_version": "fixture-v1",
                "group_id": "group-pii-pending",
                "synthetic_recipe_version": None,
                "constraint_template_id": None,
                "pii_review_status": "pending",
            },
        )

        counts = summarize_counts([record], target_count=1)

        self.assertEqual(counts["annotated_count"], 1)
        self.assertEqual(counts["validated_count"], 1)

    def test_rejected_review_is_not_release_eligible(self):
        record = self._record(sample_id="rejected")
        record.update(
            annotation_status="completed",
            annotator="human-a",
            review_status="rejected",
            reviewer="human-b",
            file_status="valid",
            annotation={
                "business_category": "hotel",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
            },
            provenance={
                "source_uri": "fixture://rejected",
                "source_version": "fixture-v1",
                "group_id": "group-rejected",
                "synthetic_recipe_version": None,
                "constraint_template_id": None,
                "pii_review_status": "verified",
            },
        )

        self.assertFalse(is_release_eligible(record))
        self.assertEqual(summarize_counts([record], target_count=1)["validated_count"], 0)

    def test_exclusion_rows_cover_all_registered_evaluation_candidates(self):
        later = self._record(sample_id="sample-b")
        earlier = self._record("itinerary_planning", sample_id="sample-a")

        rows = build_exclusion_rows([later, earlier])

        self.assertEqual([row["sample_id"] for row in rows], ["sample-a", "sample-b"])
        self.assertEqual(
            rows[0],
            {
                "sample_id": "sample-a",
                "scenario": "itinerary_planning",
                "source_id": "source-sample-a",
                "image_path": earlier["input"]["images"][0]["path"],
                "image_sha256": earlier["input"]["images"][0]["sha256"],
                "dataset_version": "week3-eval-v1",
            },
        )

    def test_exclusion_rows_include_every_itinerary_reference_image(self):
        record = self._record("itinerary_planning", sample_id="multi-image")
        second_hash = hashlib.sha256(b"second image").hexdigest()
        record["input"]["images"].append(
            {"path": "data/eval/images/multi-image-2.jpg", "sha256": second_hash}
        )

        rows = build_exclusion_rows([record])

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["image_sha256"] for row in rows},
            {record["image_sha256"], second_hash},
        )
        self.assertEqual(
            {row["image_path"] for row in rows},
            {
                "data/eval/images/multi-image.jpg",
                "data/eval/images/multi-image-2.jpg",
            },
        )

    def test_registry_rejects_duplicate_source_ids_within_or_across_scenarios(self):
        for second_scenario in ("image_product_search", "after_sales"):
            with self.subTest(second_scenario=second_scenario):
                first = self._record("image_product_search", sample_id="source-first")
                second = self._record(second_scenario, sample_id="source-second")
                second["source_id"] = first["source_id"]

                with self.assertRaisesRegex(ManifestValidationError, "duplicate source_id"):
                    build_exclusion_rows([first, second])

    def test_registry_rejects_duplicate_image_hashes_within_or_across_scenarios(self):
        for second_scenario in ("image_product_search", "after_sales"):
            with self.subTest(second_scenario=second_scenario):
                first = self._record("image_product_search", sample_id="hash-first")
                second = self._record(second_scenario, sample_id="hash-second")
                duplicate_hash = first["input"]["images"][0]["sha256"]
                second["image_sha256"] = duplicate_hash
                second["input"]["images"][0]["sha256"] = duplicate_hash

                with self.assertRaisesRegex(ManifestValidationError, "duplicate image_sha256"):
                    build_exclusion_rows([first, second])

    def test_training_candidate_source_id_collision_is_rejected(self):
        exclusion_rows = build_exclusion_rows([self._record(sample_id="held-out")])
        training_candidates = [
            {
                "source_id": "source-held-out",
                "image_sha256": "b" * 64,
            }
        ]

        with self.assertRaisesRegex(EvaluationCollisionError, "source_id"):
            reject_evaluation_collisions(training_candidates, exclusion_rows)

    def test_training_candidate_image_hash_collision_is_rejected(self):
        held_out = self._record(sample_id="held-out")
        exclusion_rows = build_exclusion_rows([held_out])
        training_candidates = [
            {
                "source_id": "different-source",
                "image_sha256": held_out["input"]["images"][0]["sha256"],
            }
        ]

        with self.assertRaisesRegex(EvaluationCollisionError, "image_sha256"):
            reject_evaluation_collisions(training_candidates, exclusion_rows)

    def test_non_colliding_training_candidate_passes(self):
        exclusion_rows = build_exclusion_rows([self._record(sample_id="held-out")])

        reject_evaluation_collisions(
            [{"source_id": "training-source", "image_sha256": "b" * 64}],
            exclusion_rows,
        )

    def test_release_provenance_requires_group_source_version_and_image_fingerprint(self):
        record = self._record(sample_id="release-ready")
        with self.assertRaisesRegex(ManifestValidationError, "provenance"):
            validate_release_provenance([record])

        record["provenance"] = {
            "source_uri": "https://example.test/source/release-ready",
            "source_version": "fixture-v1",
            "group_id": "business-001",
            "synthetic_recipe_version": None,
            "constraint_template_id": None,
            "pii_review_status": "not_applicable",
        }
        record["input"]["images"][0]["perceptual_hash"] = "0123456789abcdef"
        validated = validate_release_provenance([record])
        self.assertEqual(validated[0]["provenance"]["group_id"], "business-001")

    def test_release_provenance_rejects_duplicate_groups_and_near_duplicate_images(self):
        first = self._record(sample_id="release-a")
        second = self._record(sample_id="release-b")
        for record, fingerprint in (
            (first, "0000000000000000"),
            (second, "0000000000000001"),
        ):
            record["provenance"] = {
                "source_uri": None,
                "source_version": "fixture-v1",
                "group_id": record["sample_id"],
                "synthetic_recipe_version": "recipe-v1",
                "constraint_template_id": None,
                "pii_review_status": "verified",
            }
            record["input"]["images"][0]["perceptual_hash"] = fingerprint

        with self.assertRaisesRegex(ManifestValidationError, "near-duplicate"):
            validate_release_provenance([first, second], max_perceptual_distance=1)

        second["input"]["images"][0]["perceptual_hash"] = "ffffffffffffffff"
        second["provenance"]["group_id"] = first["provenance"]["group_id"]
        with self.assertRaisesRegex(ManifestValidationError, "duplicate group_id"):
            validate_release_provenance([first, second])

    def test_release_provenance_preserves_pending_pii_as_reported_limitation(self):
        record = self._record(sample_id="pii-pending")
        record["provenance"] = {
            "source_uri": "yelp://review/review-1",
            "source_version": "source-v1",
            "group_id": "business-1",
            "synthetic_recipe_version": None,
            "constraint_template_id": None,
            "pii_review_status": "pending",
        }
        record["input"]["images"][0]["perceptual_hash"] = "0123456789abcdef"

        validated = validate_release_provenance([record])
        self.assertEqual(
            validated[0]["provenance"]["pii_review_status"],
            "pending",
        )

    def test_training_candidates_are_rejected_by_group_template_or_near_duplicate(self):
        record = self._record("itinerary_planning", sample_id="held-out")
        record["provenance"] = {
            "source_uri": None,
            "source_version": "fixture-v1",
            "group_id": "trip-group-1",
            "synthetic_recipe_version": "recipe-v1",
            "constraint_template_id": "constraint-template-1",
            "pii_review_status": "verified",
        }
        record["input"]["images"][0]["perceptual_hash"] = "0000000000000000"
        exclusion_rows = build_exclusion_rows([record])

        cases = (
            {"group_id": "trip-group-1", "source_id": "new-1"},
            {
                "constraint_template_id": "constraint-template-1",
                "source_id": "new-2",
            },
            {
                "image_perceptual_hash": "0000000000000001",
                "source_id": "new-3",
            },
        )
        for candidate in cases:
            candidate["image_sha256"] = None
            with self.subTest(candidate=candidate), self.assertRaises(
                EvaluationCollisionError
            ):
                reject_evaluation_collisions(
                    [candidate],
                    exclusion_rows,
                    max_perceptual_distance=1,
                )

    def test_write_jsonl_persists_utf8_rows(self):
        rows = build_exclusion_rows([self._record(sample_id="held-out")])
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry" / "exclusions.jsonl"

            write_jsonl(path, rows)

            payload = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["sample_id"], "held-out")

    def test_stratified_sampling_is_repeatable_and_input_order_independent(self):
        candidates = [
            {
                "source_type": "synthetic_fixture",
                "source_id": f"hotel-{index}",
                "source_license": "synthetic-test-only",
                "image_sha256": f"{index + 1:064x}",
                "input": {
                    "images": [
                        {
                            "path": f"data/eval/images/hotel-{index}.jpg",
                            "sha256": f"{index + 1:064x}",
                        }
                    ],
                    "text_constraints": None,
                },
                "coverage_group": "hotel",
            }
            for index in range(4)
        ] + [
            {
                "source_type": "synthetic_fixture",
                "source_id": f"restaurant-{index}",
                "source_license": "synthetic-test-only",
                "image_sha256": f"{index + 101:064x}",
                "input": {
                    "images": [
                        {
                            "path": f"data/eval/images/restaurant-{index}.jpg",
                            "sha256": f"{index + 101:064x}",
                        }
                    ],
                    "text_constraints": None,
                },
                "coverage_group": "restaurant",
            }
            for index in range(3)
        ]
        kwargs = {
            "scenario": "image_product_search",
            "dataset_version": "week3-eval-v1",
            "seed": 20260713,
            "stratum_field": "coverage_group",
            "quotas": {"hotel": 2, "restaurant": 1},
        }

        first_records, first_log = stratified_sample(candidates, **kwargs)
        second_records, second_log = stratified_sample(list(reversed(candidates)), **kwargs)

        self.assertEqual(first_records, second_records)
        self.assertEqual(first_log, second_log)
        self.assertEqual(len(first_records), 3)
        self.assertEqual(first_log["selected_total"], 3)

    def test_sampling_forces_pending_human_states(self):
        candidates = [
            {
                "source_type": "synthetic_fixture",
                "source_id": "candidate-001",
                "source_license": "synthetic-test-only",
                "image_sha256": "c" * 64,
                "input": {
                    "images": [
                        {
                            "path": "data/eval/images/candidate-001.jpg",
                            "sha256": "c" * 64,
                        }
                    ],
                    "text_constraints": None,
                },
                "coverage_group": "hotel",
                "annotation_status": "completed",
                "annotator": "model",
                "review_status": "validated",
                "annotation": {"fabricated": True},
            }
        ]

        records, _ = stratified_sample(
            candidates,
            scenario="image_product_search",
            dataset_version="week3-eval-v1",
            seed=7,
            stratum_field="coverage_group",
            quotas={"hotel": 1},
        )

        self.assertEqual(records[0]["annotation_status"], "pending")
        self.assertIsNone(records[0]["annotator"])
        self.assertEqual(records[0]["review_status"], "pending")
        self.assertIsNone(records[0]["annotation"])

    def test_sampling_preserves_inference_input_for_pending_records(self):
        candidate = {
            "source_type": "synthetic_fixture",
            "source_id": "itinerary-candidate-001",
            "source_license": "synthetic-test-only",
            "image_sha256": "f" * 64,
            "input": {
                "images": [
                    {
                        "path": "data/eval/images/reference-001.jpg",
                        "sha256": "f" * 64,
                    }
                ],
                "text_constraints": "三天，必须包含博物馆",
            },
            "coverage_group": "paired_reference_and_text",
        }

        records, _ = stratified_sample(
            [candidate],
            scenario="itinerary_planning",
            dataset_version="week3-eval-v1",
            seed=7,
            stratum_field="coverage_group",
            quotas={"paired_reference_and_text": 1},
        )

        self.assertEqual(records[0]["input"], candidate["input"])

    def test_sampling_computes_hashes_from_actual_image_bytes(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "data" / "eval" / "images"
            image_dir.mkdir(parents=True)
            first_path = image_dir / "reference-001.jpg"
            second_path = image_dir / "reference-002.jpg"
            first_path.write_bytes(b"first image bytes")
            second_path.write_bytes(b"second image bytes")
            candidate = {
                "source_type": "synthetic_fixture",
                "source_id": "itinerary-candidate-002",
                "source_license": "synthetic-test-only",
                "image_sha256": None,
                "input": {
                    "images": [
                        {"path": "data/eval/images/reference-001.jpg", "sha256": None},
                        {"path": "data/eval/images/reference-002.jpg", "sha256": None},
                    ],
                    "text_constraints": "两天，优先公共交通",
                },
                "coverage_group": "paired_reference_and_text",
            }

            records, _ = stratified_sample(
                [candidate],
                scenario="itinerary_planning",
                dataset_version="week3-eval-v1",
                seed=7,
                stratum_field="coverage_group",
                quotas={"paired_reference_and_text": 1},
                root=root,
            )

        expected_hashes = [
            hashlib.sha256(b"first image bytes").hexdigest(),
            hashlib.sha256(b"second image bytes").hexdigest(),
        ]
        self.assertEqual(
            [image["sha256"] for image in records[0]["input"]["images"]],
            expected_hashes,
        )
        self.assertEqual(records[0]["image_sha256"], expected_hashes[0])

    def test_sampling_log_records_quota_shortfall_without_inventing_candidates(self):
        candidate = {
            "source_type": "synthetic_fixture",
            "source_id": "candidate-001",
            "source_license": "synthetic-test-only",
            "image_sha256": "d" * 64,
            "input": {
                "images": [
                    {
                        "path": "data/eval/images/candidate-001.jpg",
                        "sha256": "d" * 64,
                    }
                ],
                "text_constraints": None,
            },
            "coverage_group": "facility_damage",
        }

        records, sampling_log = stratified_sample(
            [candidate],
            scenario="after_sales",
            dataset_version="week3-eval-v1",
            seed=11,
            stratum_field="coverage_group",
            quotas={"facility_damage": 3},
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            sampling_log["strata"]["facility_damage"],
            {
                "available_count": 1,
                "requested_count": 3,
                "selected_count": 1,
                "shortfall_count": 2,
                "selected_source_ids": ["candidate-001"],
            },
        )

    def test_sampling_rejects_duplicate_source_ids(self):
        candidates = [
            {
                "source_type": "synthetic_fixture",
                "source_id": "duplicate-source",
                "source_license": "synthetic-test-only",
                "image_sha256": character * 64,
                "input": {
                    "images": [
                        {
                            "path": f"data/eval/images/duplicate-{character}.jpg",
                            "sha256": character * 64,
                        }
                    ],
                    "text_constraints": None,
                },
                "coverage_group": "hotel",
            }
            for character in ("a", "b")
        ]

        with self.assertRaisesRegex(ManifestValidationError, "duplicate source_id"):
            stratified_sample(
                candidates,
                scenario="image_product_search",
                dataset_version="week3-eval-v1",
                seed=7,
                stratum_field="coverage_group",
                quotas={"hotel": 2},
            )

    def test_checked_in_config_defines_targets_and_balanced_coverage_quotas(self):
        config = load_evaluation_config(self.PROJECT_ROOT / "configs" / "evaluation_week3.yaml")

        self.assertEqual(config["dataset_version"], "week3_evaluation_v1")
        self.assertEqual(
            {
                scenario: settings["target_count"]
                for scenario, settings in config["scenarios"].items()
            },
            {
                "image_product_search": 200,
                "after_sales": 150,
                "itinerary_planning": 100,
            },
        )
        for settings in config["scenarios"].values():
            self.assertEqual(
                sum(settings["sampling"]["quotas"].values()),
                settings["target_count"],
            )

    def test_config_rejects_paths_that_escape_the_repository(self):
        config_text = (
            self.PROJECT_ROOT.joinpath("configs", "evaluation_week3.yaml")
            .read_text(encoding="utf-8")
            .replace(
                "data/eval/registry/evaluation_exclusion_manifest.jsonl",
                "../outside/evaluation_exclusion_manifest.jsonl",
            )
        )
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "evaluation.yaml"
            config_path.write_text(config_text, encoding="utf-8")

            with self.assertRaisesRegex(ManifestValidationError, "repository-relative"):
                load_evaluation_config(config_path)

    def test_local_evaluation_data_and_run_artifacts_are_git_ignored(self):
        for path in (
            "data/eval/candidates/source_candidates.jsonl",
            "data/eval/images/reference.jpg",
            "data/eval/logs/sampling_log.json",
            "data/eval/manifests/image_product_search_v1.jsonl",
            "data/eval/registry/evaluation_exclusion_manifest.jsonl",
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--quiet", path],
                    cwd=self.PROJECT_ROOT,
                    check=False,
                )

                self.assertEqual(result.returncode, 0, path)

    def test_runtime_manifest_and_registry_files_are_not_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "--", "data/eval/manifests", "data/eval/registry"],
            cwd=self.PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_init_then_validate_reports_targets_without_fake_completion(self):
        config = load_evaluation_config(self.PROJECT_ROOT / "configs" / "evaluation_week3.yaml")
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            initialization = prepare_week3_evaluation.initialize_evaluation_workspace(
                config, root=root
            )
            result = validate_configured_manifests(config, root=root)

        self.assertEqual(initialization["status"], "initialized")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["exclusion_count"], 0)
        for scenario, counts in result["counts"].items():
            with self.subTest(scenario=scenario):
                self.assertGreater(counts["target_count"], 0)
                self.assertEqual(counts["candidate_count"], 0)
                self.assertEqual(counts["annotated_count"], 0)
                self.assertEqual(counts["validated_count"], 0)
                self.assertEqual(counts["tested_count"], 0)

    def test_candidate_sampling_script_writes_pending_manifest_and_audit_log(self):
        config = {
            "dataset_version": "week3_evaluation_v1",
            "scenarios": {
                "image_product_search": {
                    "sampling": {
                        "seed": 5,
                        "stratum_field": "coverage_group",
                        "quotas": {"hotel": 1},
                    }
                }
            },
        }
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "data" / "eval" / "images" / "candidate-001.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"candidate image bytes")
            candidate = {
                "source_type": "synthetic_fixture",
                "source_id": "candidate-001",
                "source_license": "synthetic-test-only",
                "image_sha256": None,
                "input": {
                    "images": [
                        {
                            "path": "data/eval/images/candidate-001.jpg",
                            "sha256": None,
                        }
                    ],
                    "text_constraints": None,
                },
                "coverage_group": "hotel",
            }
            candidates_path = root / "candidates.jsonl"
            output_path = root / "manifest.jsonl"
            log_path = root / "sampling_log.json"
            write_jsonl(candidates_path, [candidate])

            summary = run_candidate_sampling(
                config,
                scenario="image_product_search",
                candidates_path=candidates_path,
                output_path=output_path,
                log_path=log_path,
                root=root,
            )

            records = load_manifest(output_path, root=root)
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["selected_total"], 1)
        self.assertEqual(records[0]["annotation_status"], "pending")
        self.assertEqual(log_payload["seed"], 5)

    def test_week3_cli_init_then_validate_in_new_workspace(self):
        config_path = self.PROJECT_ROOT / "configs" / "evaluation_week3.yaml"
        prepare_script = self.PROJECT_ROOT / "scripts" / "prepare_week3_evaluation.py"
        validate_script = self.PROJECT_ROOT / "scripts" / "validate_week3_evaluation.py"
        with TemporaryDirectory() as tmpdir:
            commands = [
                [
                    sys.executable,
                    str(prepare_script),
                    "--config",
                    str(config_path),
                    "init",
                ],
                [
                    sys.executable,
                    str(validate_script),
                    "--config",
                    str(config_path),
                ],
            ]

            for command in commands:
                with self.subTest(script=command[1]):
                    result = subprocess.run(
                        command,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
