import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path


class EvaluationRunnerConfigTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def test_checked_in_config_declares_runtime_and_verified_model_sources(self):
        from src.data.yelp_paths import parse_simple_yaml
        from src.evaluation.config import load_evaluation_config

        config = load_evaluation_config(
            self.PROJECT_ROOT / "configs" / "evaluation_week3.yaml"
        )
        self.assertEqual(config["paths"]["runs_dir"], "data/eval/runs")
        self.assertEqual(
            config["runtime"]["model_config_path"],
            "configs/model_qwen2_vl.yaml",
        )
        self.assertEqual(
            config["runtime"]["inference_config_path"],
            "configs/inference_default.yaml",
        )
        self.assertEqual(config["runtime"]["live_base_url"], "http://localhost:8001")
        self.assertEqual(config["runtime"]["timeout_seconds"], 60)

        model_path = self.PROJECT_ROOT / config["runtime"]["model_config_path"]
        model = parse_simple_yaml(model_path.read_text(encoding="utf-8"))
        self.assertEqual(model["model_name"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(model["served_model_name"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(model["backend"], "vLLM")

    def test_config_rejects_missing_or_invalid_runtime_settings(self):
        from src.evaluation.config import load_evaluation_config
        from src.evaluation.manifests import ManifestValidationError

        source = (
            self.PROJECT_ROOT / "configs" / "evaluation_week3.yaml"
        ).read_text(encoding="utf-8")
        invalid_cases = {
            "missing_runtime": source.replace("runtime:\n", "runtime_removed:\n", 1),
            "absolute_model_path": source.replace(
                "configs/model_qwen2_vl.yaml",
                "C:/models/model_qwen2_vl.yaml",
                1,
            ),
            "zero_timeout": source.replace("timeout_seconds: 60", "timeout_seconds: 0", 1),
        }
        for name, content in invalid_cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "evaluation.yaml"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ManifestValidationError):
                    load_evaluation_config(path)

    def test_full_run_readiness_requires_target_and_stratum_coverage(self):
        from src.evaluation.runner import EvaluationRunError, validate_full_run_readiness

        scenarios = (
            "image_product_search",
            "after_sales",
            "itinerary_planning",
        )
        config = {
            "scenarios": {
                scenario: {
                    "target_count": 1,
                    "sampling": {"quotas": {f"{scenario}-stratum": 1}},
                }
                for scenario in scenarios
            }
        }
        records = {
            scenario: [
                {
                    "sample_id": f"{scenario}-1",
                    "annotation_status": "completed",
                    "file_status": "valid",
                    "provenance": {"pii_review_status": "verified"},
                    "sampling_stratum": f"{scenario}-stratum",
                    "annotation": {"gold_values": ["present"]},
                }
            ]
            for scenario in scenarios
        }
        validate_full_run_readiness(config, records)

        records["after_sales"] = []
        with self.assertRaisesRegex(EvaluationRunError, "after_sales.*validated_count"):
            validate_full_run_readiness(config, records)

    def test_full_run_readiness_allows_frozen_unknown_and_empty_semantics(self):
        from src.evaluation.runner import EvaluationRunError, validate_full_run_readiness

        config = {
            "scenarios": {
                "after_sales": {
                    "target_count": 2,
                    "sampling": {"quotas": {"routed": 2}},
                },
                "itinerary_planning": {
                    "target_count": 1,
                    "sampling": {"quotas": {"paired": 1}},
                },
            }
        }
        base = {
            "annotation_status": "completed",
            "file_status": "valid",
            "review_status": "pending",
            "provenance": {"pii_review_status": "verified"},
        }
        records = {
            "after_sales": [
                {**base, "sample_id": "a", "sampling_stratum": "routed", "annotation": {"issue_type": "unknown"}},
                {**base, "sample_id": "b", "sampling_stratum": "routed", "annotation": {"issue_type": "hygiene_stain"}},
            ],
            "itinerary_planning": [
                {**base, "sample_id": "c", "sampling_stratum": "paired", "annotation": {"style_preferences": []}},
            ],
        }

        validate_full_run_readiness(config, records)

    def test_full_run_readiness_requires_configured_source_presence(self):
        from src.evaluation.runner import EvaluationRunError, validate_full_run_readiness

        config = {
            "scenarios": {
                "after_sales": {
                    "target_count": 2,
                    "sampling": {"quotas": {"routed": 2}},
                    "required_source_types": ["public_yelp", "business_synthetic"],
                }
            }
        }
        base = {
            "annotation_status": "completed",
            "file_status": "valid",
            "review_status": "pending",
            "sampling_stratum": "routed",
            "annotation": {"issue_type": "unknown", "severity": "unknown"},
        }
        records = {
            "after_sales": [
                {**base, "sample_id": "public", "source_type": "public_yelp"},
                {**base, "sample_id": "synthetic", "source_type": "public_yelp"},
            ]
        }
        with self.assertRaisesRegex(EvaluationRunError, "source mix.*business_synthetic"):
            validate_full_run_readiness(config, records)
        records["after_sales"][1]["source_type"] = "business_synthetic"
        validate_full_run_readiness(config, records)

    def test_pilot_selection_takes_one_eligible_record_per_approved_stratum(self):
        import src.evaluation.runner as runner_module

        self.assertTrue(hasattr(runner_module, "select_pilot_records"))
        selector = runner_module.select_pilot_records
        config = {
            "scenarios": {
                "image_product_search": {
                    "sampling": {"quotas": {"hotel": 2, "attraction": 1}}
                },
                "after_sales": {
                    "sampling": {"quotas": {"facility_damage": 1}}
                },
            }
        }
        def eligible(sample_id, stratum, scenario):
            annotation = (
                {"business_category": stratum}
                if scenario == "image_product_search"
                else {"issue_type": stratum}
            )
            return {
            "sample_id": sample_id,
            "scenario": scenario,
            "annotation_status": "completed",
            "file_status": "valid",
            "sampling_stratum": stratum,
            "annotation": annotation,
        }
        records = {
            "image_product_search": [
                eligible("hotel-first", "hotel", "image_product_search"),
                eligible("hotel-second", "hotel", "image_product_search"),
                eligible("attraction-first", "attraction", "image_product_search"),
            ],
            "after_sales": [eligible("damage-first", "facility_damage", "after_sales")],
        }

        selected = selector(config, records)

        self.assertEqual(
            [row["sample_id"] for row in selected],
            ["hotel-first", "attraction-first", "damage-first"],
        )


class EvaluationResultTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def _valid_product_output(self):
        return {
            "business_category": "hotel",
            "style_tags": ["现代"],
            "visible_facilities": ["泳池"],
            "price_range": "premium",
            "observed_evidence": ["画面中可见泳池"],
            "inferred_attributes": [],
            "unknown_fields": [],
            "confidence": 0.8,
        }

    def _valid_result_record(self):
        return {
            "run_id": "dry_run_001",
            "sample_id": "sample-001",
            "scenario": "image_product_search",
            "mode": "dry-run",
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "model_config": {
                "backend": "vLLM",
                "generation": {"temperature": 0.1},
            },
            "prompt_version": "baseline_minimal_v1",
            "request_sha256": "b" * 64,
            "input_metadata": {
                "images": [{"path": "data/eval/images/a.jpg", "sha256": "a" * 64}],
                "text_constraints": None,
            },
            "raw_output": None,
            "parsed_output": None,
            "json_valid": False,
            "schema_valid": False,
            "latency_ms": 0.0,
            "error": "dry_run",
            "timestamp": "2026-07-13T12:00:00+00:00",
        }

    def test_parser_separates_json_and_schema_validity(self):
        from src.evaluation.results import parse_and_validate_output

        payload = self._valid_product_output()
        valid = parse_and_validate_output(
            self.PROJECT_ROOT,
            "image_product_search",
            f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```",
        )
        self.assertEqual(valid, {
            "parsed_output": payload,
            "json_valid": True,
            "schema_valid": True,
            "error": None,
        })

        schema_invalid = parse_and_validate_output(
            self.PROJECT_ROOT,
            "image_product_search",
            '{"business_category":"hotel"}',
        )
        self.assertTrue(schema_invalid["json_valid"])
        self.assertFalse(schema_invalid["schema_valid"])
        self.assertEqual(schema_invalid["parsed_output"], {"business_category": "hotel"})
        self.assertIn("schema_validation_error", schema_invalid["error"])

        json_invalid = parse_and_validate_output(
            self.PROJECT_ROOT,
            "image_product_search",
            "not JSON",
        )
        self.assertFalse(json_invalid["json_valid"])
        self.assertFalse(json_invalid["schema_valid"])
        self.assertIsNone(json_invalid["parsed_output"])
        self.assertIn("json_parse_error", json_invalid["error"])

    def test_parser_rejects_non_finite_json_constants(self):
        from src.evaluation.results import parse_and_validate_output

        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                result = parse_and_validate_output(
                    self.PROJECT_ROOT,
                    "image_product_search",
                    f'{{"confidence":{constant}}}',
                )
                self.assertFalse(result["json_valid"])
                self.assertFalse(result["schema_valid"])
                self.assertIsNone(result["parsed_output"])
                self.assertIn("json_parse_error", result["error"])
                self.assertIn("non-finite", result["error"])

    def test_schema_validation_rejects_programmatic_non_finite_numbers(self):
        from src.evaluation.schema_validation import SchemaValidationError, validate_output

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                payload = self._valid_product_output()
                payload["confidence"] = value
                with self.assertRaisesRegex(SchemaValidationError, "finite"):
                    validate_output(
                        self.PROJECT_ROOT,
                        "image_product_search",
                        payload,
                    )

    def test_result_record_requires_all_traceability_fields(self):
        from src.evaluation.results import ResultValidationError, validate_result_record

        record = self._valid_result_record()
        self.assertEqual(validate_result_record(record), record)

        for field in tuple(record):
            with self.subTest(missing=field):
                invalid = dict(record)
                invalid.pop(field)
                with self.assertRaisesRegex(ResultValidationError, "missing"):
                    validate_result_record(invalid)

    def test_configured_runner_persists_provenance_hashes_before_inference(self):
        from src.evaluation.config import load_evaluation_config
        from src.evaluation.runner import run_configured_evaluation

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            EvaluationRunnerCliTest()._write_workspace(root)
            summary = run_configured_evaluation(
                root=root,
                config_path=Path("configs/evaluation_week3.yaml"),
                run_id="provenance_dry_run",
                mode="dry-run",
                prompt_version="baseline_minimal_v1",
            )

            self.assertEqual(summary["dataset_version"], "week3_evaluation_v1")
            self.assertRegex(summary["selected_sample_ids_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(
                "data/eval/manifests/image_product_search_v1.jsonl",
                summary["artifact_hashes"],
            )

    def test_immutable_writer_persists_utf8_and_rejects_existing_run_id(self):
        from src.evaluation.results import ImmutableRunWriter, RunAlreadyExistsError

        with tempfile.TemporaryDirectory() as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            metadata = {
                "run_id": "dry_run_001",
                "mode": "dry-run",
                "prompt_version": "baseline_minimal_v1",
            }
            with ImmutableRunWriter(runs_dir, "dry_run_001", metadata) as writer:
                writer.write(self._valid_result_record())

            run_dir = runs_dir / "dry_run_001"
            rows = [
                json.loads(line)
                for line in (run_dir / "results.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(rows[0]["input_metadata"], self._valid_result_record()["input_metadata"])
            summary = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["status"], "completed")

            with self.assertRaises(RunAlreadyExistsError):
                ImmutableRunWriter(runs_dir, "dry_run_001", metadata)

    def test_immutable_writer_rejects_unsafe_run_ids(self):
        from src.evaluation.results import ResultValidationError, ImmutableRunWriter

        with tempfile.TemporaryDirectory() as temp_dir:
            for run_id in ("../escape", "nested/run", "", ".hidden"):
                with self.subTest(run_id=run_id):
                    with self.assertRaises(ResultValidationError):
                        ImmutableRunWriter(Path(temp_dir), run_id, {"run_id": run_id})

    def test_immutable_jsonl_writer_rejects_nested_non_finite_numbers(self):
        from src.evaluation.results import ImmutableRunWriter, ResultValidationError

        for index, value in enumerate((math.nan, math.inf, -math.inf)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp_dir:
                record = self._valid_result_record()
                record["run_id"] = f"strict_json_{index}"
                record["parsed_output"] = {"nested": [value]}
                metadata = {"run_id": record["run_id"], "mode": "dry-run"}
                with ImmutableRunWriter(
                    Path(temp_dir), record["run_id"], metadata
                ) as writer:
                    with self.assertRaisesRegex(ResultValidationError, "non-finite"):
                        writer.write(record)
                self.assertEqual(
                    (Path(temp_dir) / record["run_id"] / "results.jsonl")
                    .read_text(encoding="utf-8"),
                    "",
                )


class EvaluationModelRunnerTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def _record(self, *, sample_id="sample-001", review_status="pending"):
        return {
            "sample_id": sample_id,
            "scenario": "image_product_search",
            "source_type": "synthetic_fixture",
            "source_id": f"source-{sample_id}",
            "source_license": "synthetic-test-only",
            "image_sha256": "a" * 64,
            "input": {
                "images": [
                    {"path": "temporary_stage3_image.jpg", "sha256": "a" * 64}
                ],
                "text_constraints": None,
            },
            "split": "evaluation",
            "dataset_version": "week3_evaluation_v1",
            "annotation_status": "completed",
            "annotator": "fixture-human",
            "review_status": review_status,
            "reviewer": None,
            "file_status": "valid",
            "annotation": {},
            "provenance": {"pii_review_status": "verified"},
            "notes": None,
        }

    def _product_output(self):
        return {
            "business_category": "hotel",
            "style_tags": ["现代"],
            "visible_facilities": ["泳池"],
            "price_range": "premium",
            "observed_evidence": ["画面中可见泳池"],
            "inferred_attributes": [],
            "unknown_fields": [],
            "confidence": 0.8,
        }

    def _runtime(self):
        return {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "served_model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "model_config": {"backend": "vLLM", "max_model_len": 4096},
            "generation": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 512},
            "live_base_url": "http://localhost:8001",
            "timeout_seconds": 60,
        }

    def test_completed_valid_records_are_selected_without_semantic_or_pii_gates(self):
        from src.evaluation.runner import select_inference_records

        eligible = self._record(sample_id="eligible")
        legacy_validated = self._record(
            sample_id="legacy-validated", review_status="validated"
        )
        legacy_validated["reviewer"] = "legacy-reviewer"
        pending_pii = {
            **self._record(sample_id="pending-pii"),
            "provenance": {"pii_review_status": "pending"},
        }
        records = [
            eligible,
            legacy_validated,
            {**self._record(sample_id="missing-file"), "file_status": "missing"},
            {**self._record(sample_id="pending-label"), "annotation_status": "pending"},
            pending_pii,
        ]
        self.assertEqual(
            select_inference_records(records),
            [eligible, legacy_validated, pending_pii],
        )

    def test_dry_run_renders_without_calling_model_and_persists_traceability(self):
        from src.evaluation.runner import run_records

        def forbidden_transport(url, payload, timeout):
            self.fail("dry-run must not call a model transport")

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_records(
                root=self.PROJECT_ROOT,
                records=[self._record()],
                runs_dir=Path(temp_dir),
                run_id="stage3_dry_001",
                mode="dry-run",
                prompt_version="baseline_minimal_v1",
                runtime=self._runtime(),
                transport=forbidden_transport,
            )
            self.assertEqual(summary["selected_count"], 1)
            row = json.loads(
                (Path(temp_dir) / "stage3_dry_001" / "results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(row["error"], "dry_run")
            self.assertEqual(row["prompt_version"], "baseline_minimal_v1")
            self.assertIsNone(row["raw_output"])
            self.assertFalse(row["json_valid"])
            self.assertFalse(row["schema_valid"])
            self.assertEqual(row["input_metadata"], self._record()["input"])
            self.assertEqual(
                row["model_config"]["live_base_url"],
                "http://localhost:8001",
            )
            self.assertEqual(row["model_config"]["timeout_seconds"], 60)

    def test_mock_mode_uses_supplied_raw_output_and_validates_schema(self):
        from src.evaluation.runner import run_records

        with tempfile.TemporaryDirectory() as temp_dir:
            raw = json.dumps(self._product_output(), ensure_ascii=False)
            run_records(
                root=self.PROJECT_ROOT,
                records=[self._record()],
                runs_dir=Path(temp_dir),
                run_id="stage3_mock_001",
                mode="mock",
                prompt_version="standardized_v1",
                runtime=self._runtime(),
                mock_outputs={"sample-001": raw},
            )
            row = json.loads(
                (Path(temp_dir) / "stage3_mock_001" / "results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(row["raw_output"], raw)
            self.assertEqual(row["parsed_output"], self._product_output())
            self.assertTrue(row["json_valid"])
            self.assertTrue(row["schema_valid"])
            self.assertIsNone(row["error"])

    def test_missing_mock_fixture_has_distinct_error_category(self):
        from src.evaluation.runner import run_records

        with tempfile.TemporaryDirectory() as temp_dir:
            run_records(
                root=self.PROJECT_ROOT,
                records=[self._record()],
                runs_dir=Path(temp_dir),
                run_id="stage3_mock_missing",
                mode="mock",
                prompt_version="baseline_minimal_v1",
                runtime=self._runtime(),
                mock_outputs={},
            )
            row = json.loads(
                (Path(temp_dir) / "stage3_mock_missing" / "results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertIsNone(row["raw_output"])
            self.assertIsNone(row["parsed_output"])
            self.assertFalse(row["json_valid"])
            self.assertFalse(row["schema_valid"])
            self.assertTrue(row["error"].startswith("mock_fixture_missing:"))
            self.assertNotIn("model_request_error", row["error"])

    def test_live_mode_sends_configured_multimodal_payload_and_full_schema(self):
        from src.evaluation.runner import run_records

        captured = {}

        def transport(url, payload, timeout):
            captured.update({"url": url, "payload": payload, "timeout": timeout})
            return json.dumps(self._product_output(), ensure_ascii=False)

        image_path = self.PROJECT_ROOT / "temporary_stage3_image.jpg"
        image_path.write_bytes(b"synthetic image bytes")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                run_records(
                    root=self.PROJECT_ROOT,
                    records=[self._record()],
                    runs_dir=Path(temp_dir),
                    run_id="stage3_live_001",
                    mode="live",
                    prompt_version="standardized_v1",
                    runtime=self._runtime(),
                    transport=transport,
                )
        finally:
            image_path.unlink(missing_ok=True)

        self.assertEqual(captured["url"], "http://localhost:8001/v1/chat/completions")
        self.assertEqual(captured["timeout"], 60)
        self.assertEqual(captured["payload"]["model"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(captured["payload"]["temperature"], 0.1)
        user_content = captured["payload"]["messages"][-1]["content"]
        image_parts = [part for part in user_content if part["type"] == "image_url"]
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        request_text = "\n".join(
            part["text"] for part in user_content if part["type"] == "text"
        )
        self.assertIn('"additionalProperties":false', request_text)
        self.assertIn('"business_category"', request_text)

    def test_live_transport_failure_is_persisted_without_fabricated_output(self):
        from src.evaluation.runner import run_records

        def failing_transport(url, payload, timeout):
            raise RuntimeError("service unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            run_records(
                root=self.PROJECT_ROOT,
                records=[self._record()],
                runs_dir=Path(temp_dir),
                run_id="stage3_live_failure",
                mode="live",
                prompt_version="baseline_minimal_v1",
                runtime=self._runtime(),
                transport=failing_transport,
            )
            row = json.loads(
                (Path(temp_dir) / "stage3_live_failure" / "results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertIsNone(row["raw_output"])
            self.assertIsNone(row["parsed_output"])
            self.assertFalse(row["json_valid"])
            self.assertFalse(row["schema_valid"])
            self.assertIn("model_request_error", row["error"])
            self.assertIn("service unavailable", row["error"])


class EvaluationRunnerCliTest(unittest.TestCase):
    def _write_workspace(self, root):
        (root / "configs").mkdir(parents=True)
        (root / "data" / "eval" / "manifests").mkdir(parents=True)
        (root / "data" / "eval" / "registry").mkdir(parents=True)
        config = """dataset_version: week3_evaluation_v1
paths:
  images_dir: data/eval/images
  exclusion_manifest: data/eval/registry/evaluation_exclusion_manifest.jsonl
  sampling_logs_dir: data/eval/logs
  runs_dir: data/eval/runs
runtime:
  model_config_path: configs/model_qwen2_vl.yaml
  inference_config_path: configs/inference_default.yaml
  live_base_url: http://localhost:8001
  timeout_seconds: 60
scenarios:
  image_product_search:
    manifest_path: data/eval/manifests/image_product_search_v1.jsonl
    target_count: 2
    sampling:
      seed: 1
      stratum_field: coverage_group
      quotas:
        hotel: 2
  after_sales:
    manifest_path: data/eval/manifests/after_sales_v1.jsonl
    target_count: 2
    sampling:
      seed: 1
      stratum_field: coverage_group
      quotas:
        facility_damage: 2
  itinerary_planning:
    manifest_path: data/eval/manifests/itinerary_planning_v1.jsonl
    target_count: 2
    sampling:
      seed: 1
      stratum_field: coverage_group
      quotas:
        paired_reference_and_text: 2
"""
        (root / "configs" / "evaluation_week3.yaml").write_text(
            config, encoding="utf-8"
        )
        (root / "configs" / "model_qwen2_vl.yaml").write_text(
            "model_name: Qwen/Qwen2-VL-2B-Instruct\n"
            "served_model_name: Qwen/Qwen2-VL-2B-Instruct\n"
            "backend: vLLM\n",
            encoding="utf-8",
        )
        (root / "configs" / "inference_default.yaml").write_text(
            "temperature: 0.1\ntop_p: 0.9\nmax_tokens: 512\n",
            encoding="utf-8",
        )
        for scenario in (
            "image_product_search",
            "after_sales",
            "itinerary_planning",
        ):
            prompt = (
                root
                / "configs"
                / "evaluation"
                / "prompts"
                / "baseline_minimal_v1"
                / f"{scenario}.txt"
            )
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text("识别测试输入。\n", encoding="utf-8")
            schema = (
                root
                / "configs"
                / "evaluation"
                / "schemas"
                / f"{scenario}_v1.schema.json"
            )
            schema.parent.mkdir(parents=True, exist_ok=True)
            schema.write_text(
                '{"type":"object","additionalProperties":true}\n',
                encoding="utf-8",
            )
        for scenario in (
            "image_product_search",
            "after_sales",
            "itinerary_planning",
        ):
            (root / "data" / "eval" / "manifests" / f"{scenario}_v1.jsonl").write_text(
                "", encoding="utf-8"
            )
        (root / "data" / "eval" / "registry" / "evaluation_exclusion_manifest.jsonl").write_text(
            "", encoding="utf-8"
        )

    def _pending_record(
        self,
        root,
        *,
        scenario,
        sample_id,
        source_id,
        image_name,
        image_bytes,
    ):
        image_dir = root / "data" / "eval" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / image_name
        image_path.write_bytes(image_bytes)
        relative_path = image_path.relative_to(root).as_posix()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        return {
            "sample_id": sample_id,
            "scenario": scenario,
            "source_type": "synthetic_fixture",
            "source_id": source_id,
            "source_license": "synthetic-test-only",
            "image_sha256": image_hash,
            "input": {
                "images": [{"path": relative_path, "sha256": image_hash}],
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
        }

    def test_cli_dry_run_creates_empty_run_and_rejects_repeat_run_id(self):
        from scripts.run_week3_evaluation import run_cli
        from src.evaluation.results import RunAlreadyExistsError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_workspace(root)
            args = [
                "--config",
                "configs/evaluation_week3.yaml",
                "--run-id",
                "stage3_cli_dry",
                "--mode",
                "dry-run",
                "--prompt-version",
                "baseline_minimal_v1",
            ]
            summary = run_cli(args, root=root)
            self.assertEqual(summary["selected_count"], 0)
            self.assertEqual(summary["record_count"], 0)
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(
                (root / "data" / "eval" / "runs" / "stage3_cli_dry" / "results.jsonl").is_file()
            )

            with self.assertRaises(RunAlreadyExistsError):
                run_cli(args, root=root)

    def test_runtime_settings_are_loaded_from_referenced_config_files(self):
        from src.evaluation.config import load_evaluation_config
        from src.evaluation.runner import load_runtime_settings

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_workspace(root)
            config = load_evaluation_config(root / "configs" / "evaluation_week3.yaml")
            runtime = load_runtime_settings(root, config)

        self.assertEqual(runtime["model_name"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(runtime["served_model_name"], "Qwen/Qwen2-VL-2B-Instruct")
        self.assertEqual(runtime["model_config"]["backend"], "vLLM")
        self.assertEqual(runtime["generation"], {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 512,
        })

    def test_configured_runner_rejects_stale_registry_before_creating_run(self):
        from scripts.run_week3_evaluation import run_cli
        from src.evaluation.manifests import ManifestValidationError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_workspace(root)
            registry = (
                root
                / "data"
                / "eval"
                / "registry"
                / "evaluation_exclusion_manifest.jsonl"
            )
            stale_content = '{"source_id":"stale"}\n'
            registry.write_text(stale_content, encoding="utf-8")
            with self.assertRaisesRegex(ManifestValidationError, "stale"):
                run_cli(
                    [
                        "--config",
                        "configs/evaluation_week3.yaml",
                        "--run-id",
                        "stale_registry_run",
                        "--mode",
                        "dry-run",
                        "--prompt-version",
                        "baseline_minimal_v1",
                    ],
                    root=root,
                )
            self.assertFalse(
                (root / "data" / "eval" / "runs" / "stale_registry_run").exists()
            )
            self.assertEqual(registry.read_text(encoding="utf-8"), stale_content)

    def test_configured_runner_rejects_cross_scenario_source_and_hash_conflicts(self):
        from scripts.run_week3_evaluation import run_cli
        from src.evaluation.manifests import ManifestValidationError

        for conflict in ("source_id", "image_sha256"):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                self._write_workspace(root)
                same_source = "shared-source" if conflict == "source_id" else "source-a"
                first_bytes = b"same" if conflict == "image_sha256" else b"first"
                second_bytes = b"same" if conflict == "image_sha256" else b"second"
                records = [
                    self._pending_record(
                        root,
                        scenario="image_product_search",
                        sample_id="product-1",
                        source_id=same_source,
                        image_name="product.jpg",
                        image_bytes=first_bytes,
                    ),
                    self._pending_record(
                        root,
                        scenario="after_sales",
                        sample_id="sales-1",
                        source_id=(
                            same_source if conflict == "source_id" else "source-b"
                        ),
                        image_name="sales.jpg",
                        image_bytes=second_bytes,
                    ),
                ]
                for record in records:
                    manifest = (
                        root
                        / "data"
                        / "eval"
                        / "manifests"
                        / f"{record['scenario']}_v1.jsonl"
                    )
                    manifest.write_text(
                        json.dumps(record, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                run_id = f"cross_conflict_{conflict}"
                with self.assertRaisesRegex(
                    ManifestValidationError,
                    f"duplicate {conflict}",
                ):
                    run_cli(
                        [
                            "--config",
                            "configs/evaluation_week3.yaml",
                            "--run-id",
                            run_id,
                            "--mode",
                            "dry-run",
                            "--prompt-version",
                            "baseline_minimal_v1",
                        ],
                        root=root,
                    )
                self.assertFalse(
                    (root / "data" / "eval" / "runs" / run_id).exists()
                )

    def test_configured_runner_rejects_wrong_manifest_scenario_before_run_directory(self):
        from scripts.run_week3_evaluation import run_cli
        from src.evaluation.manifests import (
            ManifestValidationError,
            build_exclusion_rows,
            write_jsonl,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_workspace(root)
            wrong_record = self._pending_record(
                root,
                scenario="after_sales",
                sample_id="wrong-scenario-1",
                source_id="wrong-scenario-source",
                image_name="wrong-scenario.jpg",
                image_bytes=b"wrong scenario image",
            )
            wrong_manifest = (
                root
                / "data"
                / "eval"
                / "manifests"
                / "image_product_search_v1.jsonl"
            )
            write_jsonl(wrong_manifest, [wrong_record])
            registry = (
                root
                / "data"
                / "eval"
                / "registry"
                / "evaluation_exclusion_manifest.jsonl"
            )
            write_jsonl(registry, build_exclusion_rows([wrong_record]))

            run_id = "wrong_manifest_scenario"
            with self.assertRaisesRegex(
                ManifestValidationError,
                "contains scenario 'after_sales'.*expected 'image_product_search'",
            ):
                run_cli(
                    [
                        "--config",
                        "configs/evaluation_week3.yaml",
                        "--run-id",
                        run_id,
                        "--mode",
                        "dry-run",
                        "--prompt-version",
                        "baseline_minimal_v1",
                    ],
                    root=root,
                )
            self.assertFalse(
                (root / "data" / "eval" / "runs" / run_id).exists()
            )

    def test_configured_live_runner_requires_release_provenance_before_run_directory(self):
        from scripts.run_week3_evaluation import run_cli
        from src.evaluation.manifests import (
            ManifestValidationError,
            build_exclusion_rows,
            write_jsonl,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_workspace(root)
            record = self._pending_record(
                root,
                scenario="image_product_search",
                sample_id="live-without-provenance",
                source_id="live-source",
                image_name="live.jpg",
                image_bytes=b"live image bytes",
            )
            record.update(
                {
                    "annotation_status": "completed",
                    "annotator": "human-a",
                    "review_status": "validated",
                    "reviewer": "human-b",
                    "annotation": {
                        "business_category": "hotel",
                        "style_tags": [],
                        "visible_facilities": [],
                        "price_range": "unknown",
                    },
                }
            )
            manifest = (
                root
                / "data/eval/manifests/image_product_search_v1.jsonl"
            )
            write_jsonl(manifest, [record])
            registry = (
                root
                / "data/eval/registry/evaluation_exclusion_manifest.jsonl"
            )
            write_jsonl(registry, build_exclusion_rows([record]))

            run_id = "live_missing_release_provenance"
            with self.assertRaisesRegex(ManifestValidationError, "release provenance"):
                run_cli(
                    [
                        "--config",
                        "configs/evaluation_week3.yaml",
                        "--run-id",
                        run_id,
                        "--mode",
                        "live",
                        "--run-scope",
                        "full",
                        "--prompt-version",
                        "baseline_minimal_v1",
                    ],
                    root=root,
                )
            self.assertFalse((root / "data/eval/runs" / run_id).exists())

    def test_mock_response_loader_rejects_duplicate_sample_ids(self):
        from scripts.run_week3_evaluation import load_mock_outputs
        from src.evaluation.runner import EvaluationRunError

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mock.jsonl"
            path.write_text(
                '{"sample_id":"sample-1","raw_output":"{}"}\n'
                '{"sample_id":"sample-1","raw_output":"{}"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationRunError, "duplicate"):
                load_mock_outputs(path)

    def test_mock_response_loader_rejects_non_finite_json_constants(self):
        from scripts.run_week3_evaluation import load_mock_outputs
        from src.evaluation.runner import EvaluationRunError

        with tempfile.TemporaryDirectory() as temp_dir:
            for constant in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(constant=constant):
                    path = Path(temp_dir) / f"mock-{constant}.jsonl"
                    path.write_text(
                        '{"sample_id":"sample-1","raw_output":"{}",'
                        f'"invalid":{constant}}}\n',
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(EvaluationRunError, "non-finite"):
                        load_mock_outputs(path)

    def test_evaluation_run_directory_is_git_ignored(self):
        project_root = Path(__file__).resolve().parents[1]
        ignore_lines = {
            line.strip()
            for line in (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
        self.assertIn("data/eval/runs/", ignore_lines)


if __name__ == "__main__":
    unittest.main()
