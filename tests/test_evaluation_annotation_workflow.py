import copy
import hashlib
import unittest

from scripts.manage_week3_annotations import export_packet
from src.evaluation.annotation_suggestions import build_deterministic_suggestion
import src.evaluation.annotation_workflow as workflow_module
from src.evaluation.annotation_workflow import apply_annotations
from src.evaluation.manifests import ManifestValidationError


def pending_record():
    digest = hashlib.sha256(b"annotation-workflow").hexdigest()
    return {
        "sample_id": "image-product-1",
        "scenario": "image_product_search",
        "source_type": "public_yelp",
        "source_id": "source-1",
        "source_license": "test",
        "image_sha256": digest,
        "input": {
            "images": [
                {
                    "path": "data/eval/images/sample.jpg",
                    "sha256": digest,
                    "perceptual_hash": "0123456789abcdef",
                }
            ],
            "text_constraints": None,
        },
        "split": "evaluation",
        "dataset_version": "week3_evaluation_v1",
        "annotation_status": "pending",
        "annotator": None,
        "review_status": "pending",
        "reviewer": None,
        "file_status": "valid",
        "annotation": None,
        "notes": None,
        "provenance": {
            "source_uri": "yelp://photo/1",
            "source_version": "source-v1",
            "group_id": "business-1",
            "synthetic_recipe_version": None,
            "constraint_template_id": None,
            "pii_review_status": "not_applicable",
        },
    }


class AnnotationWorkflowTest(unittest.TestCase):
    def test_product_suggestion_uses_stratum_without_guessing_visual_fields(self):
        record = pending_record()
        record["sampling_stratum"] = "hotel"

        suggestion = build_deterministic_suggestion(record)

        category = suggestion["field_suggestions"]["business_category"]
        self.assertEqual(category["value"], "hotel")
        self.assertEqual(category["confidence"], "high")
        self.assertTrue(category["requires_human_confirmation"])
        self.assertEqual(
            suggestion["unsupported_fields"],
            ["price_range", "style_tags", "visible_facilities"],
        )
        self.assertTrue(suggestion["non_gold"])

    def test_synthetic_after_sales_suggestion_reproduces_recipe_text(self):
        record = pending_record()
        record.update(
            scenario="after_sales",
            source_type="business_synthetic",
            source_id="synthetic:transport_delay:0002",
            sampling_stratum="transport_delay",
        )
        record["provenance"].update(
            source_uri="synthetic://week3/transport_delay/0002",
            synthetic_recipe_version="week3_business_evidence_v1",
        )

        suggestion = build_deterministic_suggestion(record)

        fields = suggestion["field_suggestions"]
        self.assertEqual(fields["issue_type"]["value"], "transport_delay")
        self.assertEqual(
            fields["ocr_ground_truth"]["value"],
            [
                "TRANSPORT SERVICE UPDATE",
                "DELAYED",
                "Route R-002 | Scheduled 10:00 | Delay 60 min",
            ],
        )
        self.assertIn("severity", suggestion["unsupported_fields"])

    def test_visual_after_sales_suggestion_does_not_invent_evidence_text(self):
        record = pending_record()
        record.update(
            scenario="after_sales",
            source_type="business_synthetic",
            source_id="synthetic:hygiene_stain:0002",
            sampling_stratum="hygiene_stain",
        )
        record["provenance"].update(
            source_uri="synthetic://week3/hygiene_stain/0002",
            synthetic_recipe_version="week3_after_sales_evidence_v3",
        )

        suggestion = build_deterministic_suggestion(record)

        fields = suggestion["field_suggestions"]
        self.assertEqual(fields["issue_type"]["value"], "hygiene_stain")
        self.assertNotIn("key_information", fields)
        self.assertNotIn("ocr_ground_truth", fields)
        self.assertIn("key_information", suggestion["unsupported_fields"])
        self.assertIn("ocr_ground_truth", suggestion["unsupported_fields"])

    def test_itinerary_suggestion_parses_only_explicit_text_constraints(self):
        record = pending_record()
        record.update(
            scenario="itinerary_planning",
            sampling_stratum="paired_reference_and_text",
        )
        record["input"]["text_constraints"] = (
            "2天行程，预算不超过2000元，慢节奏；公共交通优先；"
            "最后一天17:00前结束，并包含每日用餐与交通安排。"
        )

        suggestion = build_deterministic_suggestion(record)

        fields = suggestion["field_suggestions"]
        self.assertEqual(
            fields["reference_images"]["value"],
            ["data/eval/images/sample.jpg"],
        )
        self.assertEqual(
            fields["hard_constraints"]["value"],
            [
                "2天行程",
                "预算不超过2000元",
                "最后一天17:00前结束",
                "包含每日用餐与交通安排",
            ],
        )
        self.assertEqual(
            fields["soft_constraints"]["value"],
            ["慢节奏", "公共交通优先"],
        )
        self.assertEqual(
            fields["required_itinerary_elements"]["value"],
            [
                "daily_schedule",
                "budget_check",
                "end_time_check",
                "meals",
                "transport",
            ],
        )
        self.assertIn("style_preferences", suggestion["unsupported_fields"])

    def test_export_can_embed_non_gold_suggestion_in_context_only(self):
        record = pending_record()
        record["sampling_stratum"] = "restaurant"

        rows = export_packet(
            [record],
            scenario="image_product_search",
            stage="annotation",
            include_suggestions=True,
        )

        self.assertIsNone(rows[0]["annotation"]["business_category"])
        suggestion = rows[0]["context"]["deterministic_suggestion"]
        self.assertEqual(
            suggestion["field_suggestions"]["business_category"]["value"],
            "restaurant",
        )
        self.assertNotIn("deterministic_suggestion", rows[0]["annotation"])

    def test_review_workflow_is_not_exposed(self):
        self.assertFalse(hasattr(workflow_module, "apply_reviews"))
        with self.assertRaisesRegex(ManifestValidationError, "annotation-only"):
            export_packet(
                [pending_record()],
                scenario="image_product_search",
                stage="review",
            )

    def test_annotation_completion_preserves_legacy_review_compatibility_fields(self):
        annotated = apply_annotations(
            [pending_record()],
            [
                {
                    "sample_id": "image-product-1",
                    "annotator": "annotator-a",
                    "annotation": {
                        "business_category": "hotel",
                        "style_tags": ["modern"],
                        "visible_facilities": ["pool"],
                        "price_range": "unknown",
                    },
                }
            ],
        )
        self.assertEqual(annotated[0]["annotation_status"], "completed")
        self.assertEqual(annotated[0]["review_status"], "pending")
        self.assertIsNone(annotated[0]["reviewer"])

    def test_duplicate_or_unknown_submission_is_rejected(self):
        row = {
            "sample_id": "image-product-1",
            "annotator": "human-a",
            "annotation": {},
        }
        with self.assertRaisesRegex(ManifestValidationError, "duplicate"):
            apply_annotations([pending_record()], [row, row])
        with self.assertRaisesRegex(ManifestValidationError, "unknown sample_id"):
            apply_annotations(
                [pending_record()],
                [{**row, "sample_id": "unknown"}],
            )



if __name__ == "__main__":
    unittest.main()
