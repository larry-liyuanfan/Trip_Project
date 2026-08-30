import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock
import tempfile

from src.inference.product_observation import (
    generate_observation, map_observation, observation_messages,
    observation_schema, observation_correction_messages, load_observation_config,
    canonical_config_sha256,
)
from src.inference.system_runtime import GenerationResult
from src.evaluation.visual_reference_validation import map_teacher_observation
from scripts.review_week8_observation_retry import load_cases, previous_wire
from scripts.verify_week8_observation_retry import replay_records
from src.training.week7_data import sha256_file


ROOT = Path(__file__).resolve().parents[1]


class ObservationHistoryCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.legacy = json.loads((ROOT / "configs/week8/product_observation_v3.json").read_text(encoding="utf-8"))
        self.config = {**self.legacy, "correction_protocol": "bounded_history_v1"}
        self.valid = {"subject_kind": "food_closeup", "subject_fact": "A bowl of noodles",
                      "style_evidence": [], "facility_evidence": [], "price_text": []}
        self.invalid = {**copy.deepcopy(self.valid), "facility_evidence": [{"label": "seating", "fact": "A chair"}]}

    def test_first_prompt_and_all_validation_are_unchanged(self):
        self.assertEqual(observation_messages("image.jpg", self.legacy), observation_messages("image.jpg", self.config))
        self.assertEqual(observation_schema(self.legacy), observation_schema(self.config))
        for config in (self.legacy, self.config):
            with self.assertRaisesRegex(ValueError, "food closeup"):
                map_observation(self.invalid, config)
            with self.assertRaisesRegex(ValueError, "food closeup"):
                map_teacher_observation(self.invalid, config)

    def test_history_contains_only_prior_raw_and_does_not_mutate_prompt(self):
        raw = json.dumps(self.invalid)
        messages = observation_messages("image.jpg", self.config)
        original = copy.deepcopy(messages)
        result = observation_correction_messages(messages, raw, "food closeup cannot establish venue style or facilities", self.config)
        self.assertEqual(messages, original)
        self.assertEqual(result[2], {"role": "assistant", "content": raw})
        self.assertEqual(len(result), 4)
        self.assertIn("complete corrected JSON", result[-1]["content"])
        self.assertIn("arrays must be empty", result[-1]["content"])
        self.assertEqual(len(observation_correction_messages(messages, raw, "error", self.legacy)), 3)

    def test_model_must_actually_correct_and_invalid_raw_is_preserved(self):
        backend = Mock()
        backend.generate_with_usage.side_effect = [GenerationResult(content=json.dumps(value), input_tokens=10, output_tokens=20)
                                                   for value in (self.invalid, self.valid)]
        result = generate_observation(backend, "image.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(json.loads(result["attempts"][0].raw_output), self.invalid)
        self.assertIsNotNone(result["attempts"][0].error)
        self.assertEqual(result["result"], map_observation(self.valid, self.legacy))
        self.assertEqual(backend.generate_with_usage.call_args_list[1].args[0][2]["role"], "assistant")

    def test_repeated_contradiction_still_fails_after_one_correction(self):
        backend = Mock()
        backend.generate_with_usage.return_value = GenerationResult(content=json.dumps(self.invalid), input_tokens=10, output_tokens=20)
        result = generate_observation(backend, "image.jpg", self.config)
        self.assertFalse(result["passed"])
        self.assertEqual(backend.generate_with_usage.call_count, 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertTrue(all(item.error for item in result["attempts"]))

    def test_unbounded_or_unknown_protocol_is_rejected_before_generation(self):
        for changes in ({"correction_protocol": "unknown"}, {"max_attempts": 3}, {"protocol": "product_visual_observation_v4"}):
            backend = Mock()
            with self.assertRaises(ValueError):
                generate_observation(backend, "image.jpg", {**self.config, **changes})
            backend.generate_with_usage.assert_not_called()

    def test_new_version_loads_without_changing_frozen_config(self):
        path = ROOT / "configs/week8/product_observation_retry_v1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        loaded = load_observation_config(path, canonical_config_sha256(config))
        self.assertEqual(loaded["correction_protocol"], "bounded_history_v1")
        frozen = json.loads((ROOT / "configs/week8/product_observation_scope_repair_v2.json").read_text(encoding="utf-8"))
        self.assertNotIn("correction_protocol", frozen)
        self.assertEqual(observation_messages("image.jpg", frozen), observation_messages("image.jpg", config))

    def test_previous_wire_conversion_is_lossless_and_does_not_fill_missing_fields(self):
        value = {"subject_kind": "food_closeup", "style_evidence": {"casual": "Paper bowl"}, "facility_evidence": {"seating": "Chair"}}
        raw = json.dumps(value)
        converted, status = previous_wire(raw, {"protocol": "product_visual_observation_v4"})
        translated = json.loads(converted)
        self.assertEqual(translated["style_evidence"], [{"label": "casual", "fact": "Paper bowl"}])
        self.assertNotIn("price_text", translated)
        self.assertEqual(status, "lossless_evidence_object_to_array")
        self.assertEqual(previous_wire(raw, self.legacy), (raw, "unchanged"))
        self.assertEqual(previous_wire('not JSON', {"protocol": "product_visual_observation_v4"}), ('not JSON', 'unparsed_raw_preserved'))
        self.assertEqual(previous_wire('{"x":1,"x":2}', {"protocol": "product_visual_observation_v4"})[1], 'unparsed_raw_preserved')

    def test_error_corpus_checks_fixed_development_identity_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            (root / "image.jpg").write_bytes(b"development fixture")
            def write(path, value):
                (root / path).write_text(json.dumps(value) + "\n", encoding="utf-8")
            row = {"sample_id": "dev-1", "source_id": "s1", "group_id": "g1", "constraint_template_id": None,
                   "image_path": "image.jpg", "image_sha256": sha256_file(root / "image.jpg")}
            write("manifest.jsonl", row)
            write("observation.json", self.legacy)
            generation = {"run_id": "development_source", "final_test_access": False, "development_indices": "all",
                          "output_root": "source", "observation_config": "observation.json"}
            write("generation.json", generation)
            record = {"sample_id": "dev-1", "attempts": [{"raw_output": json.dumps(self.invalid), "error": "food closeup cannot establish venue style or facilities"}]}
            write("source/base.jsonl", record)
            identity = {"config_sha256": sha256_file(root / "generation.json"), "test_rows_read": False,
                        "development_sha256": sha256_file(root / "manifest.jsonl"), "selected_sample_ids": ["dev-1"],
                        "observation_config_sha256": sha256_file(root / "observation.json")}
            write("source/identity.json", identity)
            write("source/summary.json", {"status": "COMPLETED", "profiles": {"base": {"raw_sha256": sha256_file(root / "source/base.jsonl")}}})
            config = {"final_test_access": False, "human_annotation_count": 0, "development_manifest": "manifest.jsonl",
                      "development_manifest_sha256": sha256_file(root / "manifest.jsonl"), "development_count": 1,
                      "sources": [{"generation_config": "generation.json", "profiles": ["base"]}]}
            cases, audit = load_cases(root, config)
            self.assertEqual(len(cases), 1)
            self.assertEqual(audit["unique_development_images"], 1)
            self.assertEqual(cases[0]["previous_raw"], json.dumps(self.invalid))
            self.assertNotIn("target", cases[0])
            for changed in ({"final_test_access": True}, {"human_annotation_count": 1}, {"development_count": 2}):
                with self.assertRaises(ValueError):
                    load_cases(root, {**config, **changed})
            write("source/base.jsonl", {**record, "attempts": []})
            with self.assertRaisesRegex(ValueError, "raw bytes"):
                load_cases(root, config)

    def test_error_corpus_rejects_final_flag_before_loading_sources(self):
        for flag in (True, None):
            with self.assertRaisesRegex(ValueError, "development-only"):
                load_cases(ROOT, {"final_test_access": flag, "human_annotation_count": 0})

    def test_probe_raw_replay_catches_tampered_output_history_and_coverage(self):
        case = {"case_id": "case-1", "sample_id": "dev-1", "image_path": "image.jpg",
                "previous_raw": json.dumps(self.invalid), "validation_error": "food closeup cannot establish venue style or facilities"}
        messages = observation_correction_messages(observation_messages(str(ROOT / case["image_path"]), self.config),
            case["previous_raw"], case["validation_error"], self.config)
        record = {"case_id": case["case_id"], "sample_id": case["sample_id"],
                  "input_messages_sha256": canonical_config_sha256(messages), "passed": True, "error": None,
                  "raw_output": json.dumps(self.valid), "result": map_observation(self.valid, self.config), "elapsed_ms": 1.0,
                  "input_tokens": 10, "output_tokens": 20}
        result = replay_records(ROOT, [case], [record], self.config)
        self.assertEqual(result["failures"], 0)
        for updates in ({"input_messages_sha256": "bad"}, {"result": {}}, {"raw_output": json.dumps(self.invalid)},
                        {"passed": False, "error": "invented failure"}, {"elapsed_ms": float("nan")}):
            with self.assertRaises(ValueError):
                replay_records(ROOT, [case], [{**record, **updates}], self.config)
        with self.assertRaises(ValueError):
            replay_records(ROOT, [case], [], self.config)
        failed = {**record, "passed": False, "result": None, "raw_output": json.dumps(self.invalid),
                  "error": "food closeup cannot establish venue style or facilities"}
        self.assertEqual(replay_records(ROOT, [case], [failed], self.config)["failures"], 1)
        remote_root = "/example/generation/root"
        remote_messages = observation_correction_messages(observation_messages(remote_root + "/image.jpg", self.config),
            case["previous_raw"], case["validation_error"], self.config)
        remote_record = {**record, "input_messages_sha256": canonical_config_sha256(remote_messages)}
        self.assertEqual(replay_records(ROOT, [case], [remote_record], self.config, remote_root)["failures"], 0)
        with self.assertRaises(ValueError):
            replay_records(ROOT, [case], [remote_record], self.config)


if __name__ == "__main__":
    unittest.main()
