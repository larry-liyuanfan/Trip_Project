import json
import tempfile
import unittest
from pathlib import Path

from src.training.week7_data import (
    _after_sales_identity,
    _dialogue_row,
    _product_target,
    load_week7_config,
)
from src.training.week7_evaluation import (
    Week7EvaluationError,
    compare_schema_decoding,
    enforce_test_once,
    strict_parse_output,
)
from src.training.week7_qlora import decile_evaluation_steps, structure_aware_messages, training_messages


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v1.json"
CONFIG_V2 = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v2.json"


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        length = sum(len(str(message["content"])) for message in messages)
        return {"input_ids": FakeTensor(length)}


class FakeTensor:
    def __init__(self, length):
        self.shape = (1, length)


class Week7ExecutionTests(unittest.TestCase):
    def test_locked_config_has_exact_ratios_and_parameters(self):
        config = load_week7_config(CONFIG_V2)
        self.assertEqual(config["dataset"]["general_regularization_fraction"], 0.09)
        self.assertEqual(config["dataset"]["dialogue_fraction"], 0.15)
        self.assertEqual(config["lora"]["r"], 16)
        self.assertEqual(config["training"]["early_stopping_patience"], 2)

    def test_v2_visual_targets_and_dialogue_ratios_are_evidence_bound(self):
        source = {"caption": "A modern hotel lobby with an accessible front desk"}
        product = _product_target(source)
        self.assertEqual(product["business_category"], "hotel")
        self.assertEqual(product["price_range"], "unknown")
        self.assertEqual(product["inferred_attributes"], [])
        self.assertNotIn("style_tags", product["unknown_fields"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "lock"
            outputs = [_after_sales_identity(root, output, "train", index)[1] for index in range(16)]
        self.assertEqual(len({(item["issue_type"], item["severity"]) for item in outputs}), 16)
        self.assertTrue(all(any(text.startswith("SEVERITY:") for text in item["ocr_text"]) for item in outputs))
        parent = {
            "sample_id": "parent", "source_id": "source", "group_id": "group",
            "constraint_template_id": None, "image_sha256": "a" * 64,
            "image_path": "image.png", "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [{"type": "image", "path": "image.png"}, {"type": "text", "text": "task"}]},
            ],
            "target": product,
        }
        dialogues = [_dialogue_row(parent, "train", index, 0.1, 0.5) for index in range(450)]
        self.assertEqual(sum(row["contains_tool_call"] for row in dialogues), 45)
        self.assertTrue(all(row["target"]["context_state"]["updated_requirement"] == "预算优先" for row in dialogues))

    def test_decile_schedule_includes_exact_final_step(self):
        self.assertEqual(decile_evaluation_steps(376), [38, 76, 113, 151, 188, 226, 264, 301, 339, 376])

    def test_strict_parser_does_not_extract_or_repair_json(self):
        valid = {
            "business_category": "other", "style_tags": [], "visible_facilities": [],
            "price_range": "unknown", "observed_evidence": [], "inferred_attributes": [],
            "unknown_fields": ["price_range"], "confidence": None,
        }
        parsed, json_valid, schema_valid, error = strict_parse_output(
            ROOT, "image_product_search", json.dumps(valid)
        )
        self.assertEqual(parsed, valid)
        self.assertTrue(json_valid)
        self.assertTrue(schema_valid)
        self.assertIsNone(error)
        _, json_valid, schema_valid, _ = strict_parse_output(
            ROOT, "image_product_search", "prefix " + json.dumps(valid)
        )
        self.assertFalse(json_valid)
        self.assertFalse(schema_valid)

    def test_training_target_and_structure_aware_truncation(self):
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "image", "path": "x.jpg"}, {"type": "text", "text": "first"}]},
                {"role": "assistant", "content": "old response"},
                {"role": "user", "content": "new request"},
            ],
            "target": {"result": "final"},
        }
        messages = training_messages(row)
        self.assertEqual(messages[-1]["role"], "assistant")
        truncated = structure_aware_messages(FakeProcessor(), messages, 200)
        self.assertEqual(truncated[0]["role"], "system")
        self.assertEqual(truncated[1]["role"], "user")
        self.assertEqual(truncated[-1]["role"], "assistant")
        self.assertEqual(truncated[-2]["role"], "user")
        image_count = sum(
            item.get("type") == "image"
            for message in truncated
            for item in (message["content"] if isinstance(message["content"], list) else [])
        )
        self.assertEqual(image_count, 1)

    def test_schema_comparison_is_format_only(self):
        config = load_week7_config(CONFIG)
        targets = {
            "image_product_search": {"business_category": "other", "style_tags": [], "visible_facilities": [], "price_range": "unknown", "observed_evidence": [], "inferred_attributes": [], "unknown_fields": [], "confidence": None},
            "after_sales": {"issue_type": "other", "severity": "low", "issue_location": None, "key_information": [], "ocr_text": None, "observed_evidence": [], "unknown_fields": [], "confidence": None},
            "itinerary_planning": {"style_preferences": [], "hard_constraints": [], "soft_constraints": [], "required_itinerary_elements": [], "itinerary": [{"day_index": 1, "date": None, "summary": "day", "activities": [{"start_time": None, "end_time": None, "place_name": None, "activity": "visit", "transport": None, "source_evidence": []}]}], "constraint_check": [], "observed_evidence": [], "unknown_fields": [], "confidence": None},
        }
        rows = [{"sample_id": scenario, "scenario": scenario} for scenario in targets]
        constrained = [{"sample_id": scenario, "raw_output": json.dumps(target), "latency_ms": 12, "failed": False} for scenario, target in targets.items()]
        free = [{"sample_id": scenario, "raw_output": "not-json", "latency_ms": 10, "failed": False} for scenario in targets]
        result = compare_schema_decoding(ROOT, config, rows, free, constrained)
        self.assertEqual(result["scope"], "format_only")
        self.assertNotIn("semantic", result["deltas"])
        self.assertEqual(result["modes"]["constrained"]["schema_coverage"], 1.0)

    def test_schema_fallback_cannot_inflate_constrained_coverage(self):
        config = load_week7_config(CONFIG)
        row = {"sample_id": "schema-primary-failed", "scenario": "after_sales"}
        valid = json.dumps({
            "issue_type": "other", "severity": "low", "issue_location": None,
            "key_information": [], "ocr_text": None, "observed_evidence": [],
            "unknown_fields": [], "confidence": None,
        })
        free = [{"sample_id": row["sample_id"], "raw_output": valid, "latency_ms": 1.0, "failed": False}]
        constrained = [{
            "sample_id": row["sample_id"],
            "raw_output": valid,
            "primary_constrained_raw_output": "",
            "fallback_raw_output": valid,
            "operational_raw_output": valid,
            "constrained_error": "HTTPError: unsupported schema",
            "fallback_used": True,
            "fallback_failed": False,
            "failed": False,
            "latency_ms": 2.0,
        }]
        result = compare_schema_decoding(ROOT, config, [row], free, constrained)
        self.assertEqual(result["modes"]["constrained"]["schema_coverage"], 0.0)
        self.assertEqual(result["modes"]["constrained"]["primary_failure_rate"], 1.0)
        self.assertEqual(result["modes"]["constrained"]["fallback_failure_rate"], 0.0)

    def test_test_gate_is_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.jsonl").write_text("{}\n", encoding="utf-8")
            parameter_lock = root / "parameter_lock.json"
            parameter_lock.write_text(json.dumps({
                "status": "LOCKED", "config_sha256": "a", "dataset_lock_sha256": "b",
                "selected_checkpoint": "checkpoint-1", "selected_checkpoint_sha256": "c",
            }), encoding="utf-8")
            with self.assertRaisesRegex(Week7EvaluationError, "final suite"):
                enforce_test_once(root, parameter_lock, "run-1")


if __name__ == "__main__":
    unittest.main()
