from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.training import week7_data
from src.training.week7_data import (
    CORE_SCENARIOS,
    IDENTITY_FIELDS,
    Week7DataError,
    _dialogue_row,
    _validate_aligned_dialogue,
    build_week7_lock,
    iter_jsonl,
    load_week7_config,
    sha256_file,
)
from src.training.week7_evaluation import (
    score_dialogue_record,
    valid_check_constraints_tool_call,
)
from src.training.week7_dialogue_v4_test import (
    _comparison,
    _sequential_record_generator,
)
from src.training.week7_qlora import (
    Week7TrainingError,
    _generate_record,
    assistant_content_text,
    assistant_span_labels,
    structure_aware_messages,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_V4 = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v4.json"
CONFIG_V4_FIX1 = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v4_fix1.json"


def _parent() -> dict:
    target = {
        "business_category": "hotel",
        "style_tags": ["modern"],
        "visible_facilities": ["front_desk"],
        "price_range": "unknown",
        "observed_evidence": ["modern hotel lobby"],
        "inferred_attributes": [],
        "unknown_fields": ["price_range"],
        "confidence": 0.8,
    }
    return {
        "sample_id": "week7-development-product-0000",
        "scenario": "image_product_search",
        "split": "development",
        "source_id": "fresh-source-0000",
        "image_sha256": "a" * 64,
        "group_id": "fresh-group-0000",
        "constraint_template_id": None,
        "image_path": "outputs/week7/locked_data/v4/images/a.jpg",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": "outputs/week7/locked_data/v4/images/a.jpg",
                    },
                    {"type": "text", "text": "请识别图片。"},
                ],
            },
        ],
        "target": target,
        "label_source": "programmatic_silver",
        "sample_weight": 0.5,
    }


def _perfect_turn_outputs(row: dict) -> list[dict]:
    return [
        {
            "assistant_turn_index": turn_index,
            "message_index": message_index,
            "expected_output": message["content"],
            "raw_output": str(message["content"]),
            "failed": False,
            "latency_ms": 1.0,
        }
        for turn_index, (message_index, message) in enumerate(
            (item for item in enumerate(row["messages"]) if item[1]["role"] == "assistant")
        )
    ]


class _BoolMask:
    def __init__(self, values: list[bool]) -> None:
        self.values = values

    def any(self) -> bool:
        return any(self.values)


class _FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = [list(values)]
        self.shape = (1, len(values))

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(list(self.values[0]))

    def fill_(self, value: int) -> "_FakeTensor":
        self.values[0] = [value] * self.shape[1]
        return self

    def __getitem__(self, key):
        rows, columns = key
        if rows != slice(None):
            raise AssertionError("unit tensor only supports the full row")
        return self.values[0][columns]

    def __setitem__(self, key, value) -> None:
        rows, columns = key
        if rows != slice(None):
            raise AssertionError("unit tensor only supports the full row")
        self.values[0][columns] = value

    def __ne__(self, value: int) -> _BoolMask:
        return _BoolMask([item != value for item in self.values[0]])


class _LengthProcessor:
    def apply_chat_template(
        self, messages: list[dict], *, add_generation_prompt: bool, **_kwargs,
    ) -> dict:
        length = len(messages) * 3 + int(add_generation_prompt)
        return {"input_ids": _FakeTensor(list(range(length)))}


class _FakeModel:
    training = False

    def eval(self) -> None:
        pass


def _aligned_row() -> dict:
    row = _parent()
    row["sample_id"] = "week7-development-dialogue-0000"
    row["scenario"] = "dialogue"
    row["construction_version"] = "aligned_concrete_turns_v4"
    row["messages"] = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "image", "path": row["image_path"]},
                {"type": "text", "text": "请识别图片。"},
            ],
        },
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "请继续。"},
        {"role": "assistant", "content": "最终回答"},
    ]
    return row


def _strict_generated_record(_model, _processor, messages, **kwargs) -> dict:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            raise TypeError("processor requires list content blocks")
        if any(not isinstance(item, dict) or "type" not in item for item in content):
            raise TypeError("processor requires typed content blocks")
    return {
        "sample_id": kwargs["sample_id"],
        "raw_output": "generated reply",
        "failed": False,
        "latency_ms": 1.0,
    }


class Week7V4DialogueTests(unittest.TestCase):
    def test_fix1_config_locks_new_identity_and_support_protocol(self) -> None:
        config = load_week7_config(CONFIG_V4_FIX1)
        self.assertEqual(config["dataset"]["identity_namespace"], "v4fix1")
        self.assertEqual(
            config["dataset"]["train_core_scenario_counts"],
            {
                "image_product_search": 600,
                "after_sales": 840,
                "itinerary_planning": 840,
            },
        )
        self.assertTrue(config["sampling"]["explicit_tool_request"])
        self.assertEqual(
            config["evaluation"]["dialogue_scoring_protocol"],
            "gold_plus_anchor_v1",
        )
        self.assertEqual(
            config["evaluation"]["metric_support_protocol"],
            "week7_evaluation_protocol_v4",
        )

    def test_fix1_tool_prompt_matches_strict_tool_contract(self) -> None:
        row = _dialogue_row(
            _parent(),
            "development",
            0,
            0.1,
            0.5,
            aligned=True,
            identity_version="v4fix1",
            construction_version="aligned_grounded_tool_turns_v4_fix1",
            explicit_tool_request=True,
        )
        self.assertIn("可用工具：check_constraints", row["messages"][0]["content"])
        tool_index = next(
            index for index, message in enumerate(row["messages"])
            if message["role"] == "assistant"
            and "<tool_call>" in str(message["content"])
        )
        self.assertIn("先调用 check_constraints", str(row["messages"][tool_index - 1]))
        self.assertTrue(valid_check_constraints_tool_call(row["messages"][tool_index]["content"]))
        self.assertFalse(valid_check_constraints_tool_call(json.dumps(row["target"])))

    def test_structure_aware_truncation_keeps_complete_user_led_blocks(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "middle"},
            {"role": "assistant", "content": "middle answer"},
            {"role": "user", "content": "final"},
            {"role": "assistant", "content": "final answer"},
        ]
        truncated = structure_aware_messages(_LengthProcessor(), messages, 15)
        self.assertEqual(
            [message["role"] for message in truncated],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(truncated[-2]["content"][0]["text"], "final")

    def test_processor_normalized_assistant_content_round_trips_to_text(self) -> None:
        self.assertEqual(assistant_content_text("plain"), "plain")
        self.assertEqual(
            assistant_content_text([
                {"type": "text", "text": "generated "},
                {"type": "text", "text": "reply"},
            ]),
            "generated reply",
        )
        with self.assertRaisesRegex(Week7TrainingError, "assistant content"):
            assistant_content_text([{"type": "image", "path": "x.jpg"}])

    @patch(
        "src.training.week7_qlora.generate_record",
        side_effect=_strict_generated_record,
    )
    def test_training_development_generation_keeps_normalized_content_blocks(
        self, _generate,
    ) -> None:
        record = _generate_record(
            ROOT, _FakeModel(), _LengthProcessor(), _aligned_row(),
            "unit-run", 8, 9,
        )
        self.assertEqual(len(record["turn_outputs"]), 2)
        self.assertEqual(
            [turn["message_index"] for turn in record["turn_outputs"]],
            [2, 4],
        )
        self.assertEqual(record["raw_output"], "generated reply")

    @patch(
        "src.training.week7_dialogue_v4_test.generate_record",
        side_effect=_strict_generated_record,
    )
    def test_final_test_generation_keeps_normalized_content_blocks(
        self, _generate,
    ) -> None:
        records, warmup = _sequential_record_generator(
            _FakeModel(), _LengthProcessor(), [_aligned_row()],
            "unit-run", "unit-model", 8, runtime_options={}, max_length=9,
        )
        self.assertEqual(len(records[0]["turn_outputs"]), 2)
        self.assertEqual(
            [turn["message_index"] for turn in records[0]["turn_outputs"]],
            [2, 4],
        )
        self.assertEqual(warmup["raw_output"], "generated reply")

    @patch(
        "src.training.week7_dialogue_v4_test.generate_record",
        side_effect=_strict_generated_record,
    )
    def test_invalid_direct_json_tool_turn_stops_before_tool_injection(
        self, _generate,
    ) -> None:
        row = _dialogue_row(
            _parent(),
            "development",
            0,
            0.1,
            0.5,
            aligned=True,
            identity_version="v4fix1",
            construction_version="aligned_grounded_tool_turns_v4_fix1",
            explicit_tool_request=True,
        )
        records, _warmup = _sequential_record_generator(
            _FakeModel(), _LengthProcessor(), [row],
            "unit-run", "unit-model", 8, runtime_options={},
        )
        record = records[0]
        tool_message_index = next(
            index for index, message in enumerate(row["messages"])
            if message["role"] == "assistant"
            and "<tool_call>" in str(message["content"])
        )
        self.assertTrue(record["failed"])
        self.assertEqual(record["turn_outputs"][-1]["message_index"], tool_message_index)
        self.assertFalse(record["turn_outputs"][-1]["protocol_valid"])
        self.assertFalse(any(
            turn["message_index"] > tool_message_index
            for turn in record["turn_outputs"]
        ))

    def test_v4_config_loads_with_locked_automatic_identity(self) -> None:
        config = load_week7_config(CONFIG_V4)
        self.assertEqual(config["schema_version"], "week7_multitask_context_v4")
        self.assertEqual(
            config["sampling"]["dialogue_construction_version"],
            "aligned_concrete_turns_v4",
        )
        gate = config["evaluation"]["dialogue_automatic_gate"]
        self.assertTrue(gate["enabled"])
        self.assertFalse(gate["human_input_required"])
        self.assertEqual(gate["minimum_task_result_value_accuracy"], 0.75)
        self.assertEqual(gate["minimum_sequential_turn_coverage"], 0.75)
        self.assertEqual(gate["maximum_sequential_turn_failure_rate"], 0.02)
        self.assertEqual(config["evaluation"]["human_review_queue_size"], 0)

    def test_v4_config_rejects_declared_construction_mismatch(self) -> None:
        config = copy.deepcopy(load_week7_config(CONFIG_V4))
        config["sampling"]["dialogue_construction_version"] = "legacy_inverted_v3"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v4.json"
            path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(Week7DataError, "construction"):
                load_week7_config(path)

    def test_v4_test_is_disjoint_from_full_historical_v3_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "configs/evaluation", root / "configs/evaluation")
            config = copy.deepcopy(load_week7_config(CONFIG_V4))
            config["dataset"].update({
                "dataset_version": "week7_v4_fresh_test_unit_lock",
                "train_total": 600,
                "train_per_core_scenario": 152,
                "general_regularization_count": 54,
                "dialogue_count": 90,
                "development_per_core_scenario": 3,
                "test_per_core_scenario": 3,
                "development_dialogue_count": 6,
                "test_dialogue_count": 6,
            })
            config["sampling"]["dialogue_parent_scenario_counts"] = {
                "train": {scenario: 30 for scenario in CORE_SCENARIOS},
                "development": {scenario: 2 for scenario in CORE_SCENARIOS},
                "test": {scenario: 2 for scenario in CORE_SCENARIOS},
            }
            historical_relative = "outputs/historical-v3/identity_manifest.jsonl"
            config["dataset"]["source_paths"][
                "historical_v3_identity_manifest"
            ] = historical_relative
            config_path = root / "configs/week7/config-v4.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8",
            )
            load_week7_config(config_path)

            source_root = root / "source"
            source_images = source_root / "images"
            source_images.mkdir(parents=True)
            sources = []
            for index in range(430):
                image = source_images / f"{index:04d}.jpg"
                image.write_bytes(f"fresh-image-{index:04d}".encode())
                sources.append({
                    "photo_id": str(index),
                    "business_id": str(index),
                    "source_id": f"source:{index}",
                    "group_id": f"group:{index}",
                    "source_image": image,
                    "image_sha256": sha256_file(image),
                    "caption": "modern hotel lobby with accessible front desk",
                })

            historical_path = root / historical_relative
            historical_path.parent.mkdir(parents=True)
            historical_rows = [{
                "sample_id": f"week7-v3-row-{index:04d}",
                "source_id": source["source_id"],
                "image_sha256": source["image_sha256"],
                "group_id": source["group_id"],
                "constraint_template_id": None,
                "split": "train" if index < 304 or index >= 316 else (
                    "development" if index < 310 else "test"
                ),
                "scenario": "historical",
                "label_source": "programmatic_silver",
            } for index, source in enumerate(sources[:370])]
            historical_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in historical_rows
                ),
                encoding="utf-8",
            )

            def after_sales_row(
                build_root: Path,
                output: Path,
                split: str,
                ordinal: int,
                weight: float,
                *,
                identity_version: str | None = None,
                identity_ordinal: int | None = None,
            ) -> dict:
                del identity_ordinal
                namespace = identity_version or "legacy"
                image = output / "images/synthetic_after_sales" / (
                    f"{split}-{namespace}-{ordinal}.png"
                )
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(
                    f"after-sales-{split}-{namespace}-{ordinal}".encode()
                )
                identity = {
                    "source_id": f"after-sales:{namespace}:{split}:{ordinal}",
                    "group_id": f"after-sales-group:{namespace}:{split}:{ordinal}",
                    "constraint_template_id": None,
                    "image_sha256": sha256_file(image),
                    "image_path": image.relative_to(build_root).as_posix(),
                }
                target = {
                    "issue_type": "facility_damage",
                    "severity": "high",
                    "issue_location": "evidence card",
                    "key_information": ["visible damage"],
                    "ocr_text": ["SEVERITY: HIGH"],
                    "observed_evidence": ["visible damage"],
                    "unknown_fields": [],
                    "confidence": 1.0,
                }
                return week7_data._row(
                    f"week7-{namespace}-{split}-after_sales-{ordinal:04d}",
                    "after_sales",
                    split,
                    identity,
                    [
                        week7_data._system(),
                        week7_data._user(identity["image_path"], "inspect"),
                    ],
                    target,
                    "programmatic_silver",
                    weight,
                )

            empty_consumed = {field: set() for field in IDENTITY_FIELDS}
            with (
                patch(
                    "src.training.week7_data.load_consumed_identities",
                    return_value=(empty_consumed, {"unit_test": True}),
                ),
                patch(
                    "src.training.week7_data.audit_week5_dialogues",
                    return_value={"disposition": "unit_test_excluded"},
                ),
                patch(
                    "src.training.week7_data._collect_public_sources",
                    return_value=sources,
                ),
                patch(
                    "src.training.week7_data._after_sales_row",
                    side_effect=after_sales_row,
                ),
            ):
                lock_root = build_week7_lock(root, source_root, config_path)

            lock = json.loads(
                (lock_root / "dataset_lock.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                lock["historical_v3_test_exclusion"]["historical_scope"],
                "v3_train_development_test_all_rows",
            )
            self.assertEqual(
                set(lock["historical_v3_test_exclusion"]["overlap_counts"].values()),
                {0},
            )
            v4_test = list(iter_jsonl(lock_root / "test.jsonl"))
            historical_sets = {
                field: {row.get(field) for row in historical_rows if row.get(field)}
                for field in IDENTITY_FIELDS
            }
            for field in IDENTITY_FIELDS:
                current = {row.get(field) for row in v4_test if row.get(field)}
                self.assertFalse(current & historical_sets[field], field)
            public_ids = {
                row["source_id"] for row in v4_test
                if row["source_id"].startswith("source:")
            }
            self.assertEqual(public_ids, {f"source:{index}" for index in range(370, 376)})
            self.assertEqual(
                Counter(row["parent_scenario"] for row in v4_test if row["scenario"] == "dialogue"),
                {scenario: 2 for scenario in CORE_SCENARIOS},
            )

    def test_v4_builds_aligned_concrete_turns_and_preserves_v3_path(self) -> None:
        parent = _parent()
        legacy = _dialogue_row(parent, "development", 0, 0.1, 0.5)
        self.assertEqual(legacy["messages"][2]["role"], "assistant")
        self.assertEqual(legacy["messages"][3]["role"], "user")
        self.assertNotIn("construction_version", legacy)

        aligned = _dialogue_row(
            parent, "development", 0, 0.1, 0.5, aligned=True,
        )
        self.assertEqual(aligned["construction_version"], "aligned_concrete_turns_v4")
        self.assertEqual(
            sum(message["role"] == "user" for message in aligned["messages"]),
            aligned["dialogue_rounds"],
        )
        image_indices = [
            index
            for index, message in enumerate(aligned["messages"])
            if isinstance(message.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") == "image"
                for item in message["content"]
            )
        ]
        self.assertEqual(image_indices, [1])
        self.assertEqual(json.loads(aligned["messages"][-1]["content"]), aligned["target"])
        for index, message in enumerate(aligned["messages"][:-1]):
            if message["role"] == "user":
                self.assertEqual(aligned["messages"][index + 1]["role"], "assistant")

    def test_exact_v4_target_is_automatic_and_perfect(self) -> None:
        row = _dialogue_row(
            _parent(), "development", 1, 0.1, 0.5, aligned=True,
        )
        score = score_dialogue_record(
            row, json.dumps(row["target"], ensure_ascii=False), 1.0, False,
            _perfect_turn_outputs(row),
        )
        self.assertFalse(score["human_required"])
        self.assertTrue(score["automatic_semantic_gate_eligible"])
        self.assertEqual(score["final_target_exact_match"], 1.0)
        self.assertEqual(score["sequential_turn_coverage"], 1.0)
        self.assertEqual(score["sequential_turn_failure_rate"], 0.0)
        self.assertEqual(score["automatic_composite"], 1.0)

    def test_anchor_consistency_cannot_replace_gold_visual_scoring(self) -> None:
        row = _dialogue_row(
            _parent(),
            "development",
            1,
            0.1,
            0.5,
            aligned=True,
            construction_version="aligned_grounded_tool_turns_v4_fix1",
        )
        turns = _perfect_turn_outputs(row)
        initial = json.loads(turns[0]["raw_output"])
        initial["observed_evidence"] = ["hallucinated lobby sign"]
        turns[0]["raw_output"] = json.dumps(initial, ensure_ascii=False)
        final = copy.deepcopy(row["target"])
        final["context_state"]["historical_image_reference"] = [
            "hallucinated lobby sign"
        ]
        final["task_result"]["observed_evidence"] = ["hallucinated lobby sign"]
        turns[-1]["raw_output"] = json.dumps(final, ensure_ascii=False)
        score = score_dialogue_record(
            row,
            turns[-1]["raw_output"],
            1.0,
            False,
            turns,
            scoring_protocol="gold_plus_anchor_v1",
        )
        self.assertEqual(score["anchor_retention"], 1.0)
        self.assertEqual(score["initial_task_stable_value_accuracy"], 1.0)
        self.assertLess(score["context_state_value_accuracy"], 1.0)
        self.assertLess(score["task_result_value_accuracy"], 1.0)
        self.assertLess(score["automatic_composite"], 1.0)

    def test_missing_or_misordered_sequential_turns_reduce_score(self) -> None:
        row = _dialogue_row(
            _parent(), "development", 1, 0.1, 0.5, aligned=True,
        )
        raw = json.dumps(row["target"], ensure_ascii=False)
        perfect_turns = _perfect_turn_outputs(row)
        perfect = score_dialogue_record(row, raw, 1.0, False, perfect_turns)

        missing = score_dialogue_record(row, raw, 1.0, False, perfect_turns[:-1])
        misordered_turns = copy.deepcopy(perfect_turns)
        misordered_turns[0], misordered_turns[1] = (
            misordered_turns[1], misordered_turns[0],
        )
        misordered = score_dialogue_record(
            row, raw, 1.0, False, misordered_turns,
        )
        for degraded in (missing, misordered):
            self.assertLess(
                degraded["sequential_turn_coverage"],
                perfect["sequential_turn_coverage"],
            )
            self.assertGreater(degraded["sequential_turn_failure_rate"], 0.0)
            self.assertLess(degraded["automatic_composite"], perfect["automatic_composite"])

    def test_assistant_span_labels_supervise_each_assistant_only(self) -> None:
        processor = _LengthProcessor()
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer two"},
        ]
        input_ids = processor.apply_chat_template(
            messages, add_generation_prompt=False,
        )["input_ids"]
        labels = assistant_span_labels(processor, messages, input_ids)
        supervised = [
            index for index, value in enumerate(labels.values[0]) if value != -100
        ]
        self.assertEqual(supervised, [7, 8, 13, 14])
        self.assertEqual(
            [labels.values[0][index] for index in supervised], supervised,
        )

    def test_v4_validator_rejects_broken_user_assistant_alignment(self) -> None:
        row = _dialogue_row(
            _parent(), "development", 1, 0.1, 0.5, aligned=True,
        )
        broken = copy.deepcopy(row)
        first_followup = next(
            index
            for index, message in enumerate(broken["messages"])
            if index > 1 and message["role"] == "user"
        )
        broken["messages"][first_followup + 1]["role"] = "tool"
        with self.assertRaisesRegex(Week7DataError, "alignment"):
            _validate_aligned_dialogue(broken)

    def test_wrong_task_values_cannot_receive_perfect_automatic_score(self) -> None:
        row = _dialogue_row(
            _parent(), "development", 1, 0.1, 0.5, aligned=True,
        )
        wrong = copy.deepcopy(row["target"])
        wrong["task_result"] = {
            key: "incorrect" for key in row["target"]["task_result"]
        }
        score = score_dialogue_record(
            row, json.dumps(wrong, ensure_ascii=False), 1.0, False,
        )
        self.assertEqual(score["task_result_key_coverage"], 1.0)
        self.assertEqual(score["task_result_value_accuracy"], 0.0)
        self.assertEqual(score["final_target_exact_match"], 0.0)
        self.assertLess(score["automatic_composite"], 1.0)

    def test_final_gate_enforces_task_result_value_accuracy(self) -> None:
        config = load_week7_config(CONFIG_V4)

        def role(task_value_accuracy: float) -> dict:
            return {
                "sample_count": 24,
                "failure_rate": 0.0,
                "latency_ms_mean": 1.0,
                "dialogue": {
                    "format_compliance": 1.0,
                    "context_recall": 1.0,
                    "context_state_value_accuracy": 1.0,
                    "task_result_key_coverage": 1.0,
                    "task_result_value_accuracy": task_value_accuracy,
                    "sequential_turn_coverage": 1.0,
                    "sequential_turn_failure_rate": 0.0,
                    "automatic_composite": 0.90,
                },
                "automatic_dimensions": {
                    "historical_image_reference": 1.0,
                    "requirement_update": 1.0,
                    "context_carryover": 1.0,
                    "logical_consistency": 1.0,
                },
            }

        comparison = _comparison(config, {
            "multitask": role(0.50),
            "week6_routed": role(0.40),
            "zero_shot": role(0.30),
        })
        self.assertEqual(comparison["status"], "FAIL")
        self.assertFalse(comparison["gate_checks"]["task_result_value_accuracy"])


if __name__ == "__main__":
    unittest.main()
