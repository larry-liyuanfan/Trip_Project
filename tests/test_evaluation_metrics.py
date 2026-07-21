import csv
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.evaluation.error_analysis import build_error_case, classify_result_error
from src.evaluation.metrics import (
    MetricConfigurationError,
    aggregate_scenario_scores,
    build_annotation_index,
    export_score_artifacts,
    load_metric_aliases,
    load_result_records,
    normalize_text,
    normalize_value,
    score_records,
    score_sample,
    set_metric_counts,
)


ROOT = Path(__file__).resolve().parents[1]
ALIASES_PATH = ROOT / "configs" / "evaluation" / "metric_aliases_v1.json"


def result_record(
    scenario: str,
    parsed_output: dict | None,
    *,
    json_valid: bool = True,
    schema_valid: bool = True,
    error: str | None = None,
    latency_ms: float = 10.0,
    prompt_version: str = "standardized_v1",
) -> dict:
    return {
        "run_id": "run-1",
        "sample_id": f"{scenario}-001",
        "scenario": scenario,
        "mode": "mock",
        "model_name": "fixture",
        "model_config": {},
        "prompt_version": prompt_version,
        "request_sha256": "b" * 64,
        "input_metadata": {},
        "raw_output": None if parsed_output is None else json.dumps(parsed_output),
        "parsed_output": parsed_output,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "latency_ms": latency_ms,
        "error": error,
        "timestamp": "2026-07-13T00:00:00Z",
    }


class MetricPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aliases = load_metric_aliases(ALIASES_PATH)

    def test_normalization_uses_nfkc_casefold_and_collapsed_whitespace(self) -> None:
        self.assertEqual(normalize_text("  ＨＯＴＥＬ\tRoom  "), "hotel room")
        self.assertEqual(
            normalize_value("business_category", "  酒店  ", self.aliases),
            "hotel",
        )

    def test_only_explicit_aliases_match(self) -> None:
        self.assertEqual(
            normalize_value("visible_facilities", "Swimming Pool", self.aliases),
            "pool",
        )
        self.assertEqual(
            normalize_value("visible_facilities", "swimming pools", self.aliases),
            "swimming pools",
        )

    def test_alias_loader_rejects_noncanonical_alias_chains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.json"
            path.write_text(
                json.dumps(
                    {
                        "version": "bad",
                        "fields": {"category": {"hotel": "lodging", "lodging": "stay"}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(MetricConfigurationError):
                load_metric_aliases(path)

    def test_set_metrics_have_auditable_empty_set_rules(self) -> None:
        both_empty = set_metric_counts([], [], "style_tags", self.aliases)
        self.assertEqual(
            {key: both_empty[key] for key in ("precision", "recall", "f1")},
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        missing_prediction = set_metric_counts(
            ["modern"], [], "style_tags", self.aliases
        )
        self.assertEqual(missing_prediction["precision"], 0.0)
        self.assertEqual(missing_prediction["recall"], 0.0)
        extra_prediction = set_metric_counts(
            [], ["modern"], "style_tags", self.aliases
        )
        self.assertEqual(extra_prediction["precision"], 0.0)
        self.assertEqual(extra_prediction["recall"], 1.0)

    def test_checked_in_config_declares_scoring_paths(self) -> None:
        from src.evaluation.config import load_evaluation_config

        config = load_evaluation_config(ROOT / "configs" / "evaluation_week3.yaml")
        self.assertEqual(config["paths"]["scores_dir"], "data/eval/scores")
        self.assertEqual(
            config["metrics"]["aliases_path"],
            "configs/evaluation/metric_aliases_v1.json",
        )


class ScenarioMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aliases = load_metric_aliases(ALIASES_PATH)

    def test_product_metrics_include_accuracy_multilabel_and_completeness(self) -> None:
        annotation = {
            "business_category": "hotel",
            "style_tags": ["modern", "quiet"],
            "visible_facilities": ["pool", "spa"],
            "price_range": "mid_range",
        }
        output = {
            "business_category": "酒店",
            "style_tags": ["Modern", "family"],
            "visible_facilities": ["Swimming Pool"],
            "price_range": "mid-range",
        }
        score = score_sample(
            result_record("image_product_search", output), annotation, self.aliases
        )
        self.assertEqual(score["json_compliance"], 1.0)
        self.assertEqual(score["schema_pass"], 1.0)
        self.assertEqual(score["business_category_accuracy"], 1.0)
        self.assertEqual(score["price_range_accuracy"], 1.0)
        self.assertEqual(score["style_precision"], 0.5)
        self.assertEqual(score["style_recall"], 0.5)
        self.assertAlmostEqual(score["facility_precision"], 1.0)
        self.assertAlmostEqual(score["facility_recall"], 0.5)
        self.assertAlmostEqual(score["label_completeness"], 4 / 6)
        self.assertEqual(score["multilabel_counts"]["style_tags"]["tp"], 1)

    def test_after_sales_metrics_handle_ocr_and_nullable_ground_truth(self) -> None:
        annotation = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": ["broken shower", "room 302"],
            "ocr_ground_truth": ["ROOM 302", "OUT OF ORDER"],
        }
        output = {
            "issue_type": "设施损坏",
            "severity": "高",
            "key_information": ["Broken Shower", "refund requested"],
            "ocr_text": ["room 302"],
        }
        score = score_sample(result_record("after_sales", output), annotation, self.aliases)
        self.assertEqual(score["issue_type_accuracy"], 1.0)
        self.assertEqual(score["severity_accuracy"], 1.0)
        self.assertEqual(score["key_information_f1"], 0.5)
        self.assertEqual(score["ocr_recall"], 0.5)
        self.assertEqual(score["ocr_exact_match"], 0.0)

        annotation["ocr_ground_truth"] = None
        not_applicable = score_sample(
            result_record("after_sales", output), annotation, self.aliases
        )
        self.assertIsNone(not_applicable["ocr_recall"])
        self.assertIsNone(not_applicable["ocr_exact_match"])

    def test_itinerary_metrics_track_typed_constraints_coverage_and_violations(self) -> None:
        annotation = {
            "reference_images": ["image-a.jpg", "image-b.jpg"],
            "text_constraints": "两天，必须乘坐公共交通，偏好安静酒店。",
            "style_preferences": ["quiet"],
            "hard_constraints": ["two days", "public transport"],
            "soft_constraints": ["quiet hotel"],
            "required_itinerary_elements": ["transport", "hotel"],
        }
        output = {
            "style_preferences": ["quiet"],
            "hard_constraints": ["two days", "公共交通"],
            "soft_constraints": [],
            "required_itinerary_elements": ["transport"],
            "constraint_check": [
                {
                    "constraint": "two days",
                    "constraint_type": "hard",
                    "status": "satisfied",
                    "evidence": "two itinerary days",
                },
                {
                    "constraint": "public transport",
                    "constraint_type": "hard",
                    "status": "违反",
                    "evidence": "taxi used",
                },
            ],
        }
        score = score_sample(
            result_record("itinerary_planning", output), annotation, self.aliases
        )
        self.assertAlmostEqual(score["constraint_recognition_accuracy"], 2 / 3)
        self.assertEqual(score["hard_constraint_precision"], 1.0)
        self.assertEqual(score["hard_constraint_recall"], 1.0)
        self.assertEqual(score["hard_constraint_f1"], 1.0)
        self.assertEqual(score["soft_constraint_precision"], 0.0)
        self.assertEqual(score["soft_constraint_recall"], 0.0)
        self.assertEqual(score["soft_constraint_f1"], 0.0)
        self.assertEqual(score["itinerary_element_precision"], 1.0)
        self.assertEqual(score["itinerary_element_completeness"], 0.5)
        self.assertEqual(score["itinerary_element_recall"], 0.5)
        self.assertAlmostEqual(score["itinerary_element_f1"], 2 / 3)
        self.assertAlmostEqual(score["constraint_check_coverage"], 2 / 3)
        self.assertAlmostEqual(score["constraint_violation_rate"], 1 / 3)

    def test_json_or_schema_failure_zeroes_all_structured_metrics(self) -> None:
        annotation = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": ["broken shower"],
            "ocr_ground_truth": None,
        }
        score = score_sample(
            result_record(
                "after_sales",
                None,
                json_valid=False,
                schema_valid=False,
                error="json_parse_error: invalid JSON",
            ),
            annotation,
            self.aliases,
        )
        self.assertEqual(score["json_compliance"], 0.0)
        self.assertEqual(score["schema_pass"], 0.0)
        for name in (
            "issue_type_accuracy",
            "severity_accuracy",
            "key_information_precision",
            "key_information_recall",
            "key_information_f1",
        ):
            self.assertEqual(score[name], 0.0)
        self.assertIsNone(score["ocr_recall"])
        self.assertIsNone(score["ocr_exact_match"])

    def test_unparsed_minimal_baseline_keeps_semantic_metrics_pending(self) -> None:
        annotation = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": ["broken shower"],
            "ocr_ground_truth": None,
        }
        score = score_sample(
            result_record(
                "after_sales",
                None,
                json_valid=False,
                schema_valid=False,
                error="json_parse_error: natural-language baseline",
                prompt_version="baseline_minimal_v1",
            ),
            annotation,
            self.aliases,
        )

        self.assertEqual(score["json_compliance"], 0.0)
        self.assertEqual(score["semantic_metrics_status"], "pending")
        self.assertEqual(score["multilabel_counts"], {})
        for name in (
            "issue_type_accuracy",
            "severity_accuracy",
            "key_information_f1",
            "ocr_recall",
        ):
            self.assertIsNone(score[name])
        aggregate = aggregate_scenario_scores([score])
        self.assertIsNone(aggregate["issue_type_accuracy"])
        self.assertEqual(aggregate["issue_type_accuracy_support_count"], 0)
        self.assertNotIn("key_information_f1_micro", aggregate)

    def test_unknown_gold_scalars_are_excluded_from_metric_support(self) -> None:
        product = score_sample(
            result_record(
                "image_product_search",
                {
                    "business_category": "hotel",
                    "style_tags": [],
                    "visible_facilities": [],
                    "price_range": "premium",
                },
            ),
            {
                "business_category": "unknown",
                "style_tags": [],
                "visible_facilities": [],
                "price_range": "unknown",
            },
            self.aliases,
        )
        self.assertIsNone(product["business_category_accuracy"])
        self.assertIsNone(product["price_range_accuracy"])

        after_sales = score_sample(
            result_record(
                "after_sales",
                None,
                json_valid=False,
                schema_valid=False,
                error="json_parse_error: invalid",
            ),
            {
                "issue_type": "unknown",
                "severity": "unknown",
                "key_information": [],
                "ocr_ground_truth": None,
            },
            self.aliases,
        )
        self.assertIsNone(after_sales["issue_type_accuracy"])
        self.assertIsNone(after_sales["severity_accuracy"])
        aggregate = aggregate_scenario_scores([after_sales])
        self.assertEqual(aggregate["issue_type_accuracy_support_count"], 0)

    def test_invalid_empty_multilabel_sample_does_not_receive_micro_credit(self) -> None:
        annotation = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": [],
            "ocr_ground_truth": None,
        }
        score = score_sample(
            result_record(
                "after_sales",
                None,
                json_valid=False,
                schema_valid=False,
                error="json_parse_error: invalid",
            ),
            annotation,
            self.aliases,
        )
        aggregate = aggregate_scenario_scores([score])
        self.assertFalse(score["structured_valid"])
        self.assertEqual(aggregate["key_information_tp"], 0)
        self.assertEqual(aggregate["key_information_fp"], 0)
        self.assertEqual(aggregate["key_information_fn"], 0)
        self.assertEqual(aggregate["key_information_f1_micro"], 0.0)

    def test_aggregate_reports_macro_micro_format_and_latency(self) -> None:
        scores = [
            {
                "scenario": "image_product_search",
                "json_compliance": 1.0,
                "schema_pass": 1.0,
                "style_precision": 1.0,
                "style_recall": 0.5,
                "style_f1": 2 / 3,
                "latency_ms": 10.0,
                "multilabel_counts": {
                    "style_tags": {"tp": 1, "fp": 0, "fn": 1}
                },
            },
            {
                "scenario": "image_product_search",
                "json_compliance": 0.0,
                "schema_pass": 0.0,
                "style_precision": 0.0,
                "style_recall": 0.0,
                "style_f1": 0.0,
                "latency_ms": 30.0,
                "multilabel_counts": {
                    "style_tags": {"tp": 0, "fp": 1, "fn": 1}
                },
            },
        ]
        aggregate = aggregate_scenario_scores(scores)
        self.assertEqual(aggregate["sample_count"], 2)
        self.assertEqual(aggregate["json_compliance"], 0.5)
        self.assertEqual(aggregate["style_precision_macro"], 0.5)
        self.assertAlmostEqual(aggregate["style_precision_micro"], 0.5)
        self.assertAlmostEqual(aggregate["style_recall_micro"], 1 / 3)
        self.assertEqual(aggregate["latency_mean_ms"], 20.0)
        self.assertEqual(aggregate["latency_p95_ms"], 30.0)

    def test_aggregate_keeps_all_not_applicable_ocr_metrics_as_null(self) -> None:
        annotation = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": [],
            "ocr_ground_truth": None,
        }
        output = {
            "issue_type": "facility_damage",
            "severity": "high",
            "key_information": [],
            "ocr_text": None,
        }
        score = score_sample(result_record("after_sales", output), annotation, self.aliases)
        aggregate = aggregate_scenario_scores([score])
        self.assertIn("ocr_recall", aggregate)
        self.assertIsNone(aggregate["ocr_recall"])
        self.assertIn("ocr_exact_match", aggregate)
        self.assertIsNone(aggregate["ocr_exact_match"])


class ErrorAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aliases = load_metric_aliases(ALIASES_PATH)

    def test_error_taxonomy_distinguishes_runner_and_format_failures(self) -> None:
        cases = {
            "dry_run": result_record(
                "after_sales", None, json_valid=False, schema_valid=False,
                error="dry_run"
            ),
            "mock_fixture_missing": result_record(
                "after_sales", None, json_valid=False, schema_valid=False,
                error="mock_fixture_missing: sample"
            ),
            "model_request_error": result_record(
                "after_sales", None, json_valid=False, schema_valid=False,
                error="model_request_error: timeout"
            ),
            "json_parse_error": result_record(
                "after_sales", None, json_valid=False, schema_valid=False,
                error="json_parse_error: invalid"
            ),
            "schema_validation_error": result_record(
                "after_sales", {}, json_valid=True, schema_valid=False,
                error="schema_validation_error: required"
            ),
            "valid": result_record("after_sales", {}),
        }
        for expected, record in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_result_error(record), expected)

        error_case = build_error_case(cases["json_parse_error"], {"schema_pass": 0.0})
        self.assertEqual(error_case["error_type"], "json_parse_error")
        self.assertEqual(error_case["sample_metrics"], {"schema_pass": 0.0})

    def test_export_writes_strict_artifacts_and_refuses_overwrite(self) -> None:
        sample_scores = [
            {
                "sample_id": "sample-1",
                "scenario": "after_sales",
                "json_compliance": 1.0,
                "schema_pass": 1.0,
                "latency_ms": 12.5,
                "multilabel_counts": {},
            }
        ]
        aggregates = {
            "after_sales": {
                "scenario": "after_sales",
                "sample_count": 1,
                "json_compliance": 1.0,
            }
        }
        error_cases = [{"sample_id": "sample-2", "error_type": "json_parse_error"}]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "scores"
            paths = export_score_artifacts(
                output_dir, sample_scores, aggregates, error_cases
            )
            self.assertEqual(set(paths), {"sample_scores", "aggregate_scores", "error_cases"})
            sample = json.loads(paths["sample_scores"].read_text(encoding="utf-8"))
            self.assertEqual(sample["sample_id"], "sample-1")
            error = json.loads(paths["error_cases"].read_text(encoding="utf-8"))
            self.assertEqual(error["error_type"], "json_parse_error")
            with paths["aggregate_scores"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["scenario"], "after_sales")
            with self.assertRaises(FileExistsError):
                export_score_artifacts(output_dir, sample_scores, aggregates, error_cases)

    def test_score_records_joins_strictly_and_returns_errors(self) -> None:
        record = result_record(
            "after_sales",
            None,
            json_valid=False,
            schema_valid=False,
            error="json_parse_error: invalid",
        )
        annotations = {
            record["sample_id"]: {
                "scenario": "after_sales",
                "annotation": {
                    "issue_type": "facility_damage",
                    "severity": "high",
                    "key_information": ["broken shower"],
                    "ocr_ground_truth": None,
                },
            }
        }
        scores, aggregates, errors = score_records(
            [record], annotations, self.aliases
        )
        self.assertEqual(len(scores), 1)
        self.assertEqual(aggregates["after_sales"]["sample_count"], 1)
        self.assertEqual(errors[0]["error_type"], "json_parse_error")

        with self.assertRaisesRegex(ValueError, "duplicate result sample_id"):
            score_records([record, record], annotations, self.aliases)
        with self.assertRaisesRegex(ValueError, "missing annotation"):
            score_records([record], {}, self.aliases)
        wrong_scenario = {
            record["sample_id"]: {
                **annotations[record["sample_id"]],
                "scenario": "image_product_search",
            }
        }
        with self.assertRaisesRegex(ValueError, "scenario mismatch"):
            score_records([record], wrong_scenario, self.aliases)

    def test_export_rejects_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "scores"
            with self.assertRaises(ValueError):
                export_score_artifacts(
                    output_dir,
                    [{"sample_id": "sample-1", "metric": float("nan")}],
                    {},
                    [],
                )
            self.assertFalse(output_dir.exists())

    def test_result_loader_is_strict_and_annotation_index_rejects_duplicates(self) -> None:
        valid = result_record("after_sales", {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            self.assertEqual(load_result_records(path), [valid])
            path.write_text('{"latency_ms":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                load_result_records(path)

        completed = {
            "sample_id": "sample-1",
            "scenario": "after_sales",
            "annotation_status": "completed",
            "annotation": {"issue_type": "facility_damage"},
        }
        pending = {
            "sample_id": "sample-2",
            "scenario": "after_sales",
            "annotation_status": "pending",
            "annotation": None,
        }
        index = build_annotation_index({"after_sales": [completed, pending]})
        self.assertEqual(set(index), {"sample-1"})
        with self.assertRaisesRegex(ValueError, "duplicate annotation sample_id"):
            build_annotation_index(
                {"after_sales": [completed], "image_product_search": [completed]}
            )

    def test_score_command_rejects_missing_metadata(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifests, _, _ = self._write_score_fixture(root, metadata=None)
            with self._patched_score_dependencies(config, manifests):
                with self.assertRaisesRegex(ValueError, "metadata.json"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            self.assertFalse((root / "scores" / "run-1").exists())

    def test_score_command_rejects_metadata_run_id_mismatch(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self._completed_metadata()
            metadata["run_id"] = "run-2"
            config, manifests, _, _ = self._write_score_fixture(root, metadata=metadata)
            with self._patched_score_dependencies(config, manifests):
                with self.assertRaisesRegex(ValueError, "metadata.run_id"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            self.assertFalse((root / "scores" / "run-1").exists())

    def test_score_command_rejects_metadata_record_count_mismatch(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self._completed_metadata()
            metadata["record_count"] = 2
            metadata["selected_count"] = 2
            config, manifests, _, _ = self._write_score_fixture(root, metadata=metadata)
            with self._patched_score_dependencies(config, manifests):
                with self.assertRaisesRegex(ValueError, "metadata.record_count"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            self.assertFalse((root / "scores" / "run-1").exists())

    def test_score_command_rejects_completed_selected_count_mismatch(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self._completed_metadata()
            metadata["selected_count"] = 2
            config, manifests, _, _ = self._write_score_fixture(root, metadata=metadata)
            with self._patched_score_dependencies(config, manifests):
                with self.assertRaisesRegex(ValueError, "metadata.selected_count"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            self.assertFalse((root / "scores" / "run-1").exists())

    def test_score_command_rejects_failed_run(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self._completed_metadata()
            metadata["status"] = "failed"
            metadata["error"] = "RuntimeError: interrupted"
            config, manifests, _, _ = self._write_score_fixture(root, metadata=metadata)
            with self._patched_score_dependencies(config, manifests):
                with self.assertRaisesRegex(ValueError, "failed run"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            self.assertFalse((root / "scores" / "run-1").exists())

    def test_score_command_rejects_artifact_drift_before_loading_annotations(self) -> None:
        from scripts.score_week3_evaluation import score_run
        from src.evaluation.provenance import ProvenanceValidationError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifests, _, _ = self._write_score_fixture(
                root,
                metadata=self._completed_metadata(),
            )
            with (
                self._patched_score_dependencies(config, manifests),
                patch(
                    "scripts.score_week3_evaluation.verify_artifact_hashes",
                    side_effect=ProvenanceValidationError(
                        "artifact hash mismatch for manifest.jsonl"
                    ),
                ),
                patch(
                    "scripts.score_week3_evaluation.load_configured_manifests"
                ) as load_manifests,
            ):
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            load_manifests.assert_not_called()

    def test_score_command_rejects_selected_sample_hash_mismatch(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self._completed_metadata()
            metadata["selected_sample_ids_sha256"] = "f" * 64
            config, manifests, _, _ = self._write_score_fixture(root, metadata=metadata)
            with self._patched_score_dependencies(config, manifests):
                with self.assertRaisesRegex(ValueError, "selected sample IDs"):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                    )
            self.assertFalse((root / "scores" / "run-1").exists())

    def test_score_command_accepts_consistent_completed_run(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifests, results_path, original_results = self._write_score_fixture(
                root, metadata=self._completed_metadata()
            )
            with self._patched_score_dependencies(config, manifests):
                summary = score_run(
                    root=root,
                    config_path=Path("config.yaml"),
                    run_id="run-1",
                )

            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["run_status"], "completed")
            self.assertTrue((root / "scores" / "run-1" / "sample_scores.jsonl").is_file())
            persisted_summary = json.loads(
                (root / "scores" / "run-1" / "score_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted_summary, summary)
            self.assertEqual(results_path.read_text(encoding="utf-8"), original_results)

    @staticmethod
    def _completed_metadata() -> dict:
        from src.evaluation.provenance import canonical_sha256

        return {
            "run_id": "run-1",
            "mode": "mock",
            "prompt_version": "standardized_v1",
            "model_name": "fixture",
            "model_config": {},
            "dataset_version": "fixture_v1",
            "artifact_hashes": {"fixture.txt": "a" * 64},
            "selected_sample_ids_sha256": canonical_sha256(["after_sales-001"]),
            "selected_count": 1,
            "status": "completed",
            "record_count": 1,
            "error": None,
        }

    def _write_score_fixture(
        self,
        root: Path,
        *,
        metadata: dict | None,
    ) -> tuple[dict, dict, Path, str]:
        run_dir = root / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        record = result_record(
            "after_sales",
            {
                "issue_type": "facility_damage",
                "severity": "high",
                "key_information": ["broken shower"],
                "ocr_text": None,
            },
        )
        results_path = run_dir / "results.jsonl"
        original_results = json.dumps(record) + "\n"
        results_path.write_text(original_results, encoding="utf-8")
        if metadata is not None:
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        (root / "aliases.json").write_text(
            json.dumps({"version": "test", "fields": {}}), encoding="utf-8"
        )
        config = {
            "paths": {"runs_dir": "runs", "scores_dir": "scores"},
            "metrics": {"aliases_path": "aliases.json"},
        }
        manifests = {
            "after_sales": [
                {
                    "sample_id": record["sample_id"],
                    "scenario": "after_sales",
                    "annotation_status": "completed",
                    "annotation": {
                        "issue_type": "facility_damage",
                        "severity": "high",
                        "key_information": ["broken shower"],
                        "ocr_ground_truth": None,
                    },
                }
            ]
        }
        return config, manifests, results_path, original_results

    @staticmethod
    @contextmanager
    def _patched_score_dependencies(config: dict, manifests: dict):
        with (
            patch(
                "scripts.score_week3_evaluation.load_evaluation_config",
                return_value=config,
            ),
            patch(
                "scripts.score_week3_evaluation.load_configured_manifests",
                return_value=manifests,
            ),
            patch(
                "scripts.score_week3_evaluation.verify_artifact_hashes",
                return_value=None,
            ),
        ):
            yield


if __name__ == "__main__":
    unittest.main()
