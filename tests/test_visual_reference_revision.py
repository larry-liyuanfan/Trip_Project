import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.visual_reference_revision import (
    repair_record, replay_revision, style_payload, validate_config, validate_sources,
)
from src.inference.product_observation import map_observation
from src.inference.product_style_scope import venue_style_evidence
from src.training.week7_data import sha256_file


ROOT = Path(__file__).resolve().parents[1]


class Response:
    status_code = 200

    def __init__(self, content):
        self.content = content

    def json(self):
        return {"model": "qwen3.7-plus", "usage": {"total_tokens": 20},
                "choices": [{"message": {"content": json.dumps(self.content)}}]}


class VisualReferenceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week8/visual_teacher_style_revision_v1.json").read_text(encoding="utf-8"))
        self.observation = json.loads((ROOT / self.config["observation_config"]).read_text(encoding="utf-8"))
        self.original = json.loads((ROOT / "configs/week8/product_observation_v3.json").read_text(encoding="utf-8"))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / "image.jpg"
        image.write_bytes(b"test-image-bytes")
        self.row = {"sample_id": "dev-1", "source_id": "photo-1", "image_sha256": sha256_file(image),
                    "group_id": "venue-1", "constraint_template_id": None, "split": "development", "image_path": "image.jpg"}
        primary = {"subject_kind": "dining_space", "subject_fact": "Tables and chairs",
                   "style_evidence": [{"label": "casual", "fact": "People wearing cotton shirts"}],
                   "facility_evidence": [{"label": "seating", "fact": "Visible chairs"}], "price_text": []}
        self.source = {key: value for key, value in self.row.items() if key not in ("split", "image_path")}
        self.source.update(label_source="model_generated_silver", sample_weight=0.5, error=None,
            response_model="qwen3.7-plus", visual_accuracy_claim_supported=False, observation=primary,
            target=map_observation(primary, self.original), attempts=[{"raw_content": json.dumps(primary), "error": None}])
        self.audit = self.audit_source()

    def audit_source(self):
        return validate_sources([self.row], [self.source], self.original, self.observation)[0]

    def repair(self, content):
        self.payloads = []

        def post(*args, **kwargs):
            self.payloads.append(copy.deepcopy(kwargs["json"]))
            return Response(content)

        return repair_record(self.row, self.source, self.audit, self.config, self.observation,
                             self.root, "https://unused.invalid", "test-key", post=post)

    def test_bounded_nonhuman_development_config(self):
        validate_config(self.config)
        for key, value in (("max_attempts", True), ("max_attempts", 4), ("concurrency", 3),
                           ("max_tokens", 2000), ("final_test_access", True), ("sample_weight", 1),
                           ("development_indices", [0]), ("prior_targets_supplied", True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config({**self.config, key: value})

    def test_seating_is_venue_context_not_a_clothing_only_fact(self):
        primary = copy.deepcopy(self.source["observation"])
        primary["style_evidence"][0]["fact"] = "People in t-shirts and casual seating"
        self.assertEqual(venue_style_evidence(primary, self.observation)[1], [])

    def test_complete_identity_retained_with_explicit_not_applicable_template(self):
        self.assertTrue(self.audit["reobserve_style"])
        for key in ("sample_id", "source_id", "image_sha256", "group_id", "constraint_template_id"):
            changed = {**self.source, key: "different"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_sources([self.row], [changed], self.original, self.observation)

    def test_drop_or_duplicate_samples_and_test_rows_are_rejected(self):
        for rows, refs in (([self.row], []), ([self.row] * 2, [self.source] * 2),
                           ([{**self.row, "split": "test"}], [self.source])):
            with self.assertRaises(ValueError):
                validate_sources(rows, refs, self.original, self.observation)

    def test_original_target_must_replay_from_raw(self):
        self.source["target"]["style_tags"] = []
        with self.assertRaisesRegex(ValueError, "raw replay"):
            self.audit_source()

    def test_prompt_is_blind_to_previous_labels_metadata_and_candidate(self):
        payload = style_payload(self.row, self.config, self.observation, self.root)
        text = json.dumps(payload)
        for forbidden in ("dev-1", "venue-1", "cotton shirts", "Already recorded styles"):
            self.assertNotIn(forbidden, text)
        self.assertIn("data:image/jpeg;base64,", text)

    def test_revision_changes_only_style_and_derived_fields(self):
        original = copy.deepcopy(self.source)
        result = self.repair({"style_evidence": [{"label": "modern", "fact": "Geometric pendant lights"}]})
        self.assertEqual(result["target"]["style_tags"], ["modern"])
        self.assertEqual(self.source, original)
        for key in ("subject_kind", "subject_fact", "facility_evidence", "price_text"):
            self.assertEqual(result["observation"][key], self.source["observation"][key])
        for key in ("business_category", "visible_facilities", "price_range"):
            self.assertEqual(result["target"][key], self.source["target"][key])
        replay_revision(self.source, result, self.audit, self.config, self.observation)

    def test_empty_reobserved_style_is_kept_not_fabricated_for_support(self):
        result = self.repair({"style_evidence": []})
        self.assertEqual(result["target"]["style_tags"], [])
        self.assertIn("style_tags", result["target"]["unknown_fields"])
        replay_revision(self.source, result, self.audit, self.config, self.observation)

    def test_unflagged_reference_exactly_inherited_without_api(self):
        audit = {**self.audit, "reobserve_style": False}
        result = repair_record(self.row, self.source, audit, self.config, self.observation,
                               self.root, "unused", "", post=lambda *args, **kwargs: self.fail("unexpected API"))
        replay_revision(self.source, result, audit, self.config, self.observation)
        result["target"]["style_tags"] = []
        with self.assertRaisesRegex(ValueError, "exactly unchanged"):
            replay_revision(self.source, result, audit, self.config, self.observation)

    def test_invalid_new_scope_fails_with_bounded_attempts_not_silent_filter(self):
        result = self.repair({"style_evidence": [{"label": "casual", "fact": "Cotton shirts"}]})
        self.assertEqual(len(self.payloads), 3)
        self.assertIsNone(result["target"])
        self.assertIn("nonvenue object", result["error"])
        self.assertNotIn("People wearing cotton shirts", json.dumps(self.payloads))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay_revision(self.source, result, self.audit, self.config, self.observation)

    def test_replay_rejects_non_style_or_authority_mutation(self):
        result = self.repair({"style_evidence": []})
        result["target"]["business_category"] = "hotel"
        with self.assertRaisesRegex(ValueError, "style-only replay"):
            replay_revision(self.source, result, self.audit, self.config, self.observation)
        result = self.repair({"style_evidence": []})
        result["label_source"] = "human"
        with self.assertRaisesRegex(ValueError, "authority"):
            replay_revision(self.source, result, self.audit, self.config, self.observation)


if __name__ == "__main__":
    unittest.main()
