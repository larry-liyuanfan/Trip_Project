import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data.week5_dataset import (
    SCENARIOS,
    Week5DataError,
    _check_candidate_isolation,
    candidate_payload_sha256,
    initialize_workflow_v2_sidecar,
    load_week5_config,
    qc_cross_review_selected,
    qc_audit_selected,
    validate_dialogue,
    validate_dialogue_v2,
    validate_human_annotation,
    validate_workflow_v2_sidecar,
    write_jsonl_new,
)
from src.data.week5_workflow import (
    _endpoint_allows_anonymous_access,
    _require_model_access,
    _runtime,
    apply_human_corrections,
    apply_quality_records,
    export_audited_pilot_annotation_packet,
    export_quality_packet,
    run_full_preannotation,
    run_itinerary_paired_prompt_pilot,
)


ROOT = Path(__file__).resolve().parents[1]


class Week5DatasetTests(unittest.TestCase):
    def test_only_loopback_model_endpoints_allow_anonymous_access(self) -> None:
        self.assertTrue(_endpoint_allows_anonymous_access("http://127.0.0.1:18001/v1"))
        self.assertTrue(_endpoint_allows_anonymous_access("http://localhost:8000/v1"))
        self.assertTrue(_endpoint_allows_anonymous_access("https://[::1]:8000/v1"))
        self.assertFalse(_endpoint_allows_anonymous_access("https://dashscope.aliyuncs.com/v1"))
        self.assertFalse(_endpoint_allows_anonymous_access("http://example.test/127.0.0.1"))

    def test_external_model_endpoint_still_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _require_model_access({"runtime": {"base_url": "http://127.0.0.1:18001/v1"}})
            with self.assertRaises(Week5DataError):
                _require_model_access(
                    {"runtime": {"base_url": "https://dashscope.aliyuncs.com/v1"}}
                )

    def test_runtime_endpoint_override_preserves_config_and_requires_key_off_loopback(self) -> None:
        config = {
            "runtime": {
                "base_url": "http://127.0.0.1:18001/v1",
                "model_config": "model.yaml",
                "inference_config": "inference.yaml",
                "itinerary_inference_config": "itinerary.yaml",
                "timeout_seconds": 30,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.yaml").write_text(
                "served_model_name: Qwen3-VL-4B-Instruct\n", encoding="utf-8"
            )
            for name in ("inference.yaml", "itinerary.yaml"):
                (root / name).write_text("temperature: 0.1\n", encoding="utf-8")
            original = json.loads(json.dumps(config))
            with patch.dict(
                os.environ,
                {"WEEK5_MODEL_BASE_URL_OVERRIDE": "http://127.0.0.1:8001/v1"},
                clear=True,
            ):
                runtime = _runtime(root, config, "image_product_search")
            self.assertEqual(runtime["live_base_url"], "http://127.0.0.1:8001/v1")
            self.assertEqual(config, original)
            with patch.dict(
                os.environ,
                {"WEEK5_MODEL_BASE_URL_OVERRIDE": "https://model.example/v1"},
                clear=True,
            ):
                with self.assertRaises(Week5DataError):
                    _runtime(root, config, "image_product_search")

    def _workflow_fixture(self, directory: str) -> tuple[Path, dict, str]:
        root = Path(directory)
        (root / "configs/evaluation/schemas").mkdir(parents=True)
        shutil.copy2(
            ROOT / "configs/evaluation/schemas/image_product_search_v1.schema.json",
            root / "configs/evaluation/schemas/image_product_search_v1.schema.json",
        )
        (root / "configs/week5").mkdir(parents=True)
        shutil.copy2(
            ROOT / "configs/week5/annotation_tool.json",
            root / "configs/week5/annotation_tool.json",
        )
        sample_id = "week5-image_product_search-test"
        pool = root / "outputs/week5/pools/image_product_search.jsonl"
        pool.parent.mkdir(parents=True)
        pool.write_text(json.dumps({"sample_id": sample_id, "scenario": "image_product_search"}) + "\n", encoding="utf-8")
        for scenario in ("after_sales", "itinerary_planning"):
            (pool.parent / f"{scenario}.jsonl").write_text("", encoding="utf-8")
        config = {
            "paths": {"output_dir": "outputs/week5"},
            "quality": {
                "mode": "single_operator_minimal_review_v1",
                "core_scenarios": ["after_sales", "itinerary_planning"],
                "core_cross_review_rate": 1.0,
                "general_cross_review_rate": 1.0,
                "core_audit_rate": 1.0,
                "general_audit_rate": 1.0,
            },
        }
        return root, config, sample_id

    def test_config_uses_current_schemas_and_qwen37_prompts(self) -> None:
        config = load_week5_config(ROOT, "configs/week5_dataset.json")
        self.assertEqual(config["prompt_versions"]["image_product_search"], "fewshot_4_v2")
        self.assertEqual(config["prompt_versions"]["after_sales"], "fewshot_4_v2")
        self.assertEqual(config["prompt_versions"]["itinerary_planning"], "standardized_v4")
        self.assertTrue(config["schemas"]["itinerary_planning"].endswith("itinerary_planning_v2.schema.json"))

    def test_qwen3_vl_4b_config_uses_project_control_prompt_mapping(self) -> None:
        config = load_week5_config(ROOT, "configs/week5_dataset_qwen3_vl_4b_gpu.json")
        self.assertEqual(config["prompt_versions"]["image_product_search"], "standardized_v2")
        self.assertEqual(config["prompt_versions"]["after_sales"], "fewshot_4_v2")
        self.assertEqual(config["prompt_versions"]["itinerary_planning"], "standardized_v4")
        self.assertEqual(
            config["pilot"]["itinerary_prompt_versions"],
            ["fewshot_4_v2", "standardized_v4"],
        )
        self.assertLessEqual(config["pilot"]["max_total_requests"], 60)
        self.assertLessEqual(config["pilot"]["max_gpu_hours"], 1.0)
        self.assertLessEqual(config["pilot"]["max_cost_cny"], 20.0)
        self.assertTrue(config["schemas"]["dialogue"].endswith("multimodal_dialogue_v2.schema.json"))

    def test_single_operator_config_reduces_and_nests_qc_samples(self) -> None:
        config = load_week5_config(
            ROOT, "configs/week5_dataset_qwen3_vl_4b_single_operator.json"
        )
        quality = config["quality"]
        self.assertEqual(quality["operator_count"], 1)
        self.assertEqual(quality["general_cross_review_rate"], 0.002)
        self.assertEqual(quality["general_audit_rate"], 0.0005)
        self.assertEqual(quality["core_cross_review_rate"], 0.005)
        self.assertEqual(quality["core_audit_rate"], 0.001)

    def test_isolation_rejects_source_hash_group_and_template(self) -> None:
        candidate = {
            "source_id": "source-a",
            "image_sha256": "a" * 64,
            "provenance": {"group_id": "group-a", "constraint_template_id": "template-a"},
        }
        empty = {name: set() for name in ("source_id", "image_sha256", "group_id", "constraint_template_id")}
        self.assertTrue(_check_candidate_isolation(candidate, empty, set()))
        for name, value in (("source_id", "source-a"), ("image_sha256", "a" * 64), ("group_id", "group-a"), ("constraint_template_id", "template-a")):
            exclusions = {key: set(values) for key, values in empty.items()}
            exclusions[name].add(value)
            self.assertFalse(_check_candidate_isolation(candidate, exclusions, set()))

    def test_jsonl_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            self.assertEqual(write_jsonl_new(path, [{"id": 1}]), 1)
            with self.assertRaises(Week5DataError):
                write_jsonl_new(path, [{"id": 2}])

    def test_human_output_must_match_current_schema(self) -> None:
        valid = {
            "business_category": "hotel", "style_tags": ["modern"],
            "visible_facilities": ["pool"], "price_range": "unknown",
            "observed_evidence": ["可见泳池"], "inferred_attributes": [],
            "unknown_fields": ["price_range"], "confidence": None,
        }
        validate_human_annotation(ROOT, "image_product_search", valid)
        invalid = dict(valid)
        invalid.pop("confidence")
        with self.assertRaises(Week5DataError):
            validate_human_annotation(ROOT, "image_product_search", invalid)

    def test_human_output_rejects_uncontrolled_synonym_labels(self) -> None:
        annotation = {
            "business_category": "hotel", "style_tags": ["contemporary"],
            "visible_facilities": ["swimming_pool"], "price_range": "unknown",
            "observed_evidence": [], "inferred_attributes": [],
            "unknown_fields": ["price_range"], "confidence": None,
        }
        with self.assertRaises(Week5DataError):
            validate_human_annotation(ROOT, "image_product_search", annotation)

    def test_dialogue_requires_turns_and_valid_image_references(self) -> None:
        dialogue = {
            "dialogue_id": "d-1",
            "scenario": "image_search_consultation",
            "images": [{"image_id": "img_1", "path": "data/a.jpg", "sha256": "a" * 64}],
            "messages": [
                {"role": "user", "content": "看看这张图", "image_refs": ["img_1"]},
                {"role": "assistant", "content": "请问更关注风格还是设施？", "image_refs": ["img_1"]},
                {"role": "user", "content": "更关注设施", "image_refs": []},
                {"role": "assistant", "content": "可按可见设施继续筛选。", "image_refs": ["img_1"]},
                {"role": "user", "content": "那上一张适合亲子吗？", "image_refs": ["img_1"]},
                {"role": "assistant", "content": "仅凭图片不能确认亲子服务，需要查看商家信息。", "image_refs": ["img_1"]},
            ],
        }
        validate_dialogue(dialogue)
        dialogue["messages"][2]["image_refs"] = ["missing"]
        with self.assertRaises(Week5DataError):
            validate_dialogue(dialogue)

    def test_dialogue_v2_rejects_v1_aliases_and_checks_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.bin"
            image.write_bytes(b"week5-dialogue-image")
            import hashlib
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            dialogue = {
                "schema_version": "multimodal_dialogue_v2",
                "dialogue_id": "d-v2",
                "scenario": "image_search",
                "image_resources": [{"image_id": "img_1", "path": "image.bin", "sha256": digest}],
                "turns": [
                    {"role": "user", "content": "看看这张图", "image_refs": ["img_1"]},
                    {"role": "assistant", "content": "请问关注什么？", "image_refs": ["img_1"]},
                    {"role": "user", "content": "设施", "image_refs": []},
                    {"role": "assistant", "content": "只按可见设施说明。", "image_refs": ["img_1"]},
                    {"role": "user", "content": "上一张呢？", "image_refs": ["img_1"]},
                    {"role": "assistant", "content": "仍需商家信息确认。", "image_refs": ["img_1"]},
                ],
                "source_sample_ids": ["sample-1"],
                "generation": {"run_id": "run-1", "model_name": "model", "prompt_version": "v2"},
                "human_review": {"status": "awaiting_human_annotation", "reviewer": None, "reviewed_at": None, "checks": {}},
                "qc": {"status": "partial", "reviewer": None, "reviewed_at": None, "issues": []},
            }
            validate_dialogue_v2(root, dialogue)
            invalid = dict(dialogue)
            invalid["images"] = invalid.pop("image_resources")
            with self.assertRaises(Week5DataError):
                validate_dialogue_v2(root, invalid)

    def test_workflow_v2_sidecar_binds_immutable_candidate_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_dir = root / "outputs/week5/pools"
            pool_dir.mkdir(parents=True)
            candidate = {"sample_id": "sample-1", "scenario": "image_product_search", "input": {"images": []}}
            for scenario in ("image_product_search", "after_sales", "itinerary_planning"):
                rows = [candidate] if scenario == "image_product_search" else []
                (pool_dir / f"{scenario}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            config = {"paths": {"output_dir": "outputs/week5"}}
            result = initialize_workflow_v2_sidecar(root, config, "image_product_search")
            self.assertEqual(result["records"], 1)
            row = json.loads((root / result["output"]).read_text(encoding="utf-8"))
            self.assertEqual(row["candidate_sha256"], candidate_payload_sha256(candidate))
            self.assertEqual(row["workflow_status"], "awaiting_human_annotation")
            self.assertEqual(row["model_preannotation"]["status"], "not_started")
            self.assertEqual(validate_workflow_v2_sidecar(root, config, "image_product_search")["status"], "ok")
            with self.assertRaises(Week5DataError):
                initialize_workflow_v2_sidecar(root, config, "image_product_search")

    def test_audited_pilot_refuses_overwrite_and_resumes_identical_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_dir = root / "outputs/week5/pools"
            pool_dir.mkdir(parents=True)
            candidates = [
                {
                    "sample_id": f"sample-{index}", "scenario": "itinerary_planning",
                    "input": {"images": [], "text_constraints": "2天"},
                }
                for index in range(2)
            ]
            for scenario in ("image_product_search", "after_sales", "itinerary_planning"):
                rows = candidates if scenario == "itinerary_planning" else []
                (pool_dir / f"{scenario}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            config = {
                "dataset_version": "week5_instruction_candidates_v1",
                "paths": {"output_dir": "outputs/week5"},
                "runtime": {"base_url": "http://127.0.0.1:18001/v1"},
                "pilot": {
                    "itinerary_prompt_versions": ["fewshot_4_v2", "standardized_v4"],
                    "max_unique_samples": 30, "max_total_requests": 60,
                    "max_gpu_hours": 1.0, "estimated_hourly_cost_cny": 18.3158,
                    "max_cost_cny": 20.0, "early_stop_pairs": 5,
                    "max_failure_rate_after_early_stop": 0.2,
                },
            }
            runtime = {
                "model_name": "Qwen3-VL-4B-Instruct", "served_model_name": "Qwen3-VL-4B-Instruct",
                "live_base_url": "http://127.0.0.1:18001/v1", "timeout_seconds": 30,
                "generation": {}, "model_config": {},
            }
            parsed = {"parsed_output": {"ok": True}, "json_valid": True, "schema_valid": True, "error": None}
            with patch("src.data.week5_workflow._runtime", return_value=runtime), patch(
                "src.data.week5_workflow._fewshot_context", return_value=({}, {})
            ), patch(
                "src.data.week5_workflow._render_preannotation",
                side_effect=lambda *args, prompt_version=None, **kwargs: {"prompt": prompt_version},
            ), patch(
                "src.data.week5_workflow._build_chat_payload", side_effect=lambda root, rendered, runtime: rendered
            ), patch(
                "src.data.week5_workflow.post_chat_completion",
                return_value={"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 10}},
            ), patch(
                "src.data.week5_workflow.parse_and_validate_output", return_value=parsed
            ):
                summary = run_itinerary_paired_prompt_pilot(root, config, "pilot-1", limit=2)
                self.assertEqual(summary["total_requests"], 4)
                run_dir = root / "outputs/week5/runs/pilot-1"
                self.assertTrue((run_dir / "run_manifest.json").is_file())
                self.assertTrue((run_dir / "checkpoint.json").is_file())
                self.assertEqual((run_dir / "failures.jsonl").read_text(encoding="utf-8"), "")
                attempt = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
                for field in ("run_id", "input_sha256", "candidate_sha256", "request_sha256", "raw_output_path", "retry_count"):
                    self.assertIn(field, attempt)
                packet_path = root / "outputs/week5/human_tasks/pilot-1.jsonl"
                exported = export_audited_pilot_annotation_packet(
                    root, config, "pilot-1", packet_path
                )
                self.assertEqual(exported["exported"], 2)
                packet = json.loads(packet_path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(packet["workflow_status"], "awaiting_human_annotation")
                self.assertIsNone(packet["annotator"])
                self.assertEqual(packet["revision_history"], [])
                with self.assertRaises(Week5DataError):
                    run_itinerary_paired_prompt_pilot(root, config, "pilot-1", limit=2)
                resumed = run_itinerary_paired_prompt_pilot(root, config, "pilot-1", limit=2, resume=True)
                self.assertEqual(resumed["total_requests"], 4)
                config["pilot"]["max_cost_cny"] = 19.0
                with self.assertRaises(Week5DataError):
                    run_itinerary_paired_prompt_pilot(root, config, "pilot-1", limit=2, resume=True)

    def test_full_preannotation_is_audited_resumable_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), "white").save(root / "candidate.png")
            pool_dir = root / "outputs/week5/pools"
            pool_dir.mkdir(parents=True)
            for scenario in ("image_product_search", "after_sales", "itinerary_planning"):
                candidate = {
                    "sample_id": f"sample-{scenario}",
                    "scenario": scenario,
                    "input": {
                        "images": [{"path": "candidate.png"}],
                        "text_constraints": "2天",
                    },
                }
                (pool_dir / f"{scenario}.jsonl").write_text(
                    json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            config = {
                "dataset_version": "week5_instruction_candidates_v1",
                "paths": {"output_dir": "outputs/week5"},
                "targets": {scenario: 1 for scenario in SCENARIOS},
                "prompt_versions": {
                    "image_product_search": "standardized_v2",
                    "after_sales": "fewshot_4_v2",
                    "itinerary_planning": "standardized_v4",
                },
                "runtime": {
                    "base_url": "http://127.0.0.1:18001/v1", "concurrency": 2
                },
                "full_preannotation": {
                    "shard_size": 2, "max_retries": 1,
                    "max_consecutive_request_failures": 20,
                },
            }
            runtime = {
                "model_name": "Qwen3-VL-4B-Instruct",
                "served_model_name": "Qwen3-VL-4B-Instruct",
                "live_base_url": "http://127.0.0.1:18001/v1",
                "timeout_seconds": 30, "generation": {}, "model_config": {},
            }
            parsed = {
                "parsed_output": {"ok": True}, "json_valid": True,
                "schema_valid": True, "error": None,
            }
            with patch("src.data.week5_workflow._runtime", return_value=runtime), patch(
                "src.data.week5_workflow._fewshot_context", return_value=({}, {})
            ), patch(
                "src.data.week5_workflow._render_preannotation",
                side_effect=lambda root, config, candidate, *args, **kwargs: {
                    "sample_id": candidate["sample_id"]
                },
            ), patch(
                "src.data.week5_workflow._build_chat_payload",
                side_effect=lambda root, rendered, runtime: rendered,
            ), patch(
                "src.data.week5_workflow.post_chat_completion",
                return_value={
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"total_tokens": 10},
                },
            ), patch(
                "src.data.week5_workflow.parse_and_validate_output", return_value=parsed
            ):
                summary = run_full_preannotation(root, config, "full-1")
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["completed_this_process"], 3)
                run_dir = root / "outputs/week5/runs/full-1"
                self.assertEqual(len(list((run_dir / "raw").rglob("*.txt"))), 3)
                self.assertEqual(len((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()), 3)
                manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["identity"]["run_kind"], "full_preannotation")
                with self.assertRaises(Week5DataError):
                    run_full_preannotation(root, config, "full-1")
                resumed = run_full_preannotation(root, config, "full-1", resume=True)
                self.assertEqual(resumed["skipped_this_process"], 3)
                config["runtime"]["concurrency"] = 1
                with self.assertRaises(Week5DataError):
                    run_full_preannotation(root, config, "full-1", resume=True)

    def test_full_preannotation_stops_on_consecutive_request_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), "white").save(root / "candidate.png")
            pool_dir = root / "outputs/week5/pools"
            pool_dir.mkdir(parents=True)
            for scenario in SCENARIOS:
                candidate = {
                    "sample_id": f"sample-{scenario}", "scenario": scenario,
                    "input": {
                        "images": [{"path": "candidate.png"}],
                        "text_constraints": "unknown",
                    },
                }
                (pool_dir / f"{scenario}.jsonl").write_text(
                    json.dumps(candidate) + "\n", encoding="utf-8"
                )
            config = {
                "dataset_version": "week5_instruction_candidates_v1",
                "paths": {"output_dir": "outputs/week5"},
                "targets": {scenario: 1 for scenario in SCENARIOS},
                "prompt_versions": {scenario: "prompt" for scenario in SCENARIOS},
                "runtime": {"base_url": "http://127.0.0.1:18001/v1", "concurrency": 2},
                "full_preannotation": {
                    "shard_size": 2, "max_retries": 0,
                    "max_consecutive_request_failures": 2,
                },
            }
            runtime = {
                "model_name": "model", "live_base_url": "http://127.0.0.1:18001/v1",
                "timeout_seconds": 1, "generation": {}, "model_config": {},
            }
            with patch("src.data.week5_workflow._runtime", return_value=runtime), patch(
                "src.data.week5_workflow._fewshot_context", return_value=({}, {})
            ), patch(
                "src.data.week5_workflow._render_preannotation", return_value={"prompt": "x"}
            ), patch(
                "src.data.week5_workflow._build_chat_payload", return_value={"prompt": "x"}
            ), patch(
                "src.data.week5_workflow.post_chat_completion",
                side_effect=ConnectionError("tunnel unavailable"),
            ):
                with self.assertRaises(Week5DataError):
                    run_full_preannotation(root, config, "full-fail")
            failures = (
                root / "outputs/week5/runs/full-fail/failures.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(failures), 2)

    def test_full_preannotation_records_unreadable_images_without_model_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate.jpg").write_text("AccessDenied", encoding="utf-8")
            pool_dir = root / "outputs/week5/pools"
            pool_dir.mkdir(parents=True)
            for scenario in SCENARIOS:
                candidate = {
                    "sample_id": f"sample-{scenario}",
                    "scenario": scenario,
                    "input": {
                        "images": [{"path": "candidate.jpg"}],
                        "text_constraints": "unknown",
                    },
                }
                (pool_dir / f"{scenario}.jsonl").write_text(
                    json.dumps(candidate) + "\n", encoding="utf-8"
                )
            config = {
                "dataset_version": "week5_instruction_candidates_v1",
                "paths": {"output_dir": "outputs/week5"},
                "targets": {scenario: 1 for scenario in SCENARIOS},
                "prompt_versions": {scenario: "prompt" for scenario in SCENARIOS},
                "runtime": {"base_url": "http://127.0.0.1:18001/v1", "concurrency": 1},
                "full_preannotation": {
                    "shard_size": 2,
                    "max_retries": 2,
                    "max_consecutive_request_failures": 1,
                },
            }
            runtime = {
                "model_name": "model",
                "served_model_name": "model",
                "live_base_url": "http://127.0.0.1:18001/v1",
                "timeout_seconds": 1,
                "generation": {},
                "model_config": {},
            }
            with patch("src.data.week5_workflow._runtime", return_value=runtime), patch(
                "src.data.week5_workflow._fewshot_context", return_value=({}, {})
            ), patch("src.data.week5_workflow.post_chat_completion") as request:
                summary = run_full_preannotation(root, config, "invalid-images")
            request.assert_not_called()
            self.assertEqual(summary["failed_this_process"], 3)
            attempts = (
                root / "outputs/week5/runs/invalid-images/attempts.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(json.loads(row)["error_type"] == "input_error" for row in attempts))

    def test_single_operator_qc_selection_is_nested_and_rate_bounded(self) -> None:
        config = load_week5_config(ROOT, "configs/week5_dataset.json")
        first = qc_audit_selected("sample-1", "after_sales", config)
        self.assertEqual(first, qc_audit_selected("sample-1", "after_sales", config))
        ids = [f"sample-{index}" for index in range(10000)]
        core_cross = {value for value in ids if qc_cross_review_selected(value, "after_sales", config)}
        core_audit = {value for value in ids if qc_audit_selected(value, "after_sales", config)}
        general_cross = {value for value in ids if qc_cross_review_selected(value, "image_product_search", config)}
        general_audit = {value for value in ids if qc_audit_selected(value, "image_product_search", config)}
        self.assertTrue(core_audit <= core_cross)
        self.assertTrue(general_audit <= general_cross)
        self.assertGreaterEqual(len(core_cross), 30)
        self.assertLessEqual(len(core_cross), 80)
        self.assertGreaterEqual(len(core_audit), 3)
        self.assertLessEqual(len(core_audit), 20)
        self.assertGreaterEqual(len(general_cross), 10)
        self.assertLessEqual(len(general_cross), 35)
        self.assertGreaterEqual(len(general_audit), 1)
        self.assertLessEqual(len(general_audit), 15)

    def test_human_correction_requires_real_preannotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, sample_id = self._workflow_fixture(directory)
            submission = root / "human.jsonl"
            submission.write_text(json.dumps({
                "sample_id": sample_id, "annotator": "annotator-a", "corrected_at": "2026-08-02T00:00:00Z",
                "review_session_id": "session-correction", "self_review_confirmed": True,
                "human_annotation": {
                    "business_category": "unknown", "style_tags": [], "visible_facilities": [],
                    "price_range": "unknown", "observed_evidence": [], "inferred_attributes": [],
                    "unknown_fields": ["business_category", "price_range"], "confidence": None,
                },
            }) + "\n", encoding="utf-8")
            with self.assertRaises(Week5DataError):
                apply_human_corrections(root, config, "image_product_search", submission)
            pre = root / "outputs/week5/preannotations/image_product_search.jsonl"
            pre.parent.mkdir(parents=True)
            pre.write_text(json.dumps({"sample_id": sample_id, "status": "completed", "schema_valid": True}) + "\n", encoding="utf-8")
            self.assertEqual(apply_human_corrections(root, config, "image_product_search", submission)["applied"], 1)
            self_review = json.loads((root / "outputs/week5/quality/image_product_search.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(self_review["stage"], "self_review")
            self.assertEqual(self_review["review_session_id"], "session-correction")

    def test_single_operator_cross_review_requires_selection_and_distinct_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, sample_id = self._workflow_fixture(directory)
            annotations = root / "outputs/week5/annotations/image_product_search.jsonl"
            annotations.parent.mkdir(parents=True)
            annotations.write_text(json.dumps({"sample_id": sample_id, "scenario": "image_product_search", "annotator": "a", "revision": 1, "review_session_id": "session-1"}) + "\n", encoding="utf-8")
            quality_input = root / "quality.jsonl"
            quality_input.write_text(json.dumps({"sample_id": sample_id, "stage": "cross_review", "decision": "pass", "reviewer": "a", "issues": [], "review_session_id": "session-1"}) + "\n", encoding="utf-8")
            with self.assertRaises(Week5DataError):
                apply_quality_records(root, config, "image_product_search", quality_input)
            quality = root / "outputs/week5/quality/image_product_search.jsonl"
            quality.parent.mkdir(parents=True, exist_ok=True)
            quality.write_text(json.dumps({"sample_id": sample_id, "scenario": "image_product_search", "annotation_revision": 1, "stage": "self_review", "decision": "pass", "reviewer": "a", "issues": [], "review_session_id": "session-1"}) + "\n", encoding="utf-8")
            quality_input.write_text("\n".join([
                json.dumps({"sample_id": sample_id, "stage": "cross_review", "decision": "pass", "reviewer": "a", "issues": [], "review_session_id": "session-2"}),
                json.dumps({"sample_id": sample_id, "stage": "core_audit", "decision": "pass", "reviewer": "a", "issues": [], "review_session_id": "session-3"}),
            ]) + "\n", encoding="utf-8")
            self.assertEqual(apply_quality_records(root, config, "image_product_search", quality_input)["applied"], 2)
            with self.assertRaises(Week5DataError):
                apply_quality_records(root, config, "image_product_search", quality_input)

    def test_quality_export_only_emits_ready_unfinished_selected_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, sample_id = self._workflow_fixture(directory)
            pool = root / "outputs/week5/pools/image_product_search.jsonl"
            pool.write_text(json.dumps({
                "sample_id": sample_id,
                "image": {"path": "image.jpg", "sha256": "a" * 64},
                "input": {"query": "visible attributes"},
            }) + "\n", encoding="utf-8")
            annotations = root / "outputs/week5/annotations/image_product_search.jsonl"
            annotations.parent.mkdir(parents=True)
            annotations.write_text(json.dumps({
                "sample_id": sample_id,
                "scenario": "image_product_search",
                "annotator": "operator-a",
                "revision": 1,
                "review_session_id": "correction-session",
                "human_annotation": {"business_category": "unknown"},
            }) + "\n", encoding="utf-8")
            quality = root / "outputs/week5/quality/image_product_search.jsonl"
            quality.parent.mkdir(parents=True)
            quality.write_text(json.dumps({
                "sample_id": sample_id,
                "annotation_revision": 1,
                "stage": "self_review",
                "decision": "pass",
                "reviewer": "operator-a",
                "review_session_id": "correction-session",
            }) + "\n", encoding="utf-8")
            cross_packet = root / "cross.jsonl"
            result = export_quality_packet(
                root, config, "image_product_search", "cross_review", cross_packet
            )
            self.assertEqual(result["exported"], 1)
            task = json.loads(cross_packet.read_text(encoding="utf-8"))
            self.assertIsNone(task["reviewer"])
            self.assertIsNone(task["decision"])
            self.assertEqual(task["annotation_revision"], 1)

            core_packet = root / "core-before-cross.jsonl"
            self.assertEqual(export_quality_packet(
                root, config, "image_product_search", "core_audit", core_packet
            )["exported"], 0)


if __name__ == "__main__":
    unittest.main()
