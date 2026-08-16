import json
import hashlib
import tempfile
import unittest
from pathlib import Path


class Week4DeliveryValidationTest(unittest.TestCase):
    def test_unified_validator_checks_runs_scores_and_comparisons(self):
        from src.evaluation.provenance import canonical_sha256
        from src.evaluation.week4_validation import validate_week4_delivery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/week4"
            artifact = root / "artifact.txt"
            artifact.write_text("固定契约\n", encoding="utf-8", newline="\n")
            artifact_hash = __import__("hashlib").sha256(
                artifact.read_bytes()
            ).hexdigest()
            pilot_ids = ["pilot-standard", "pilot-four", "pilot-seven"]
            full_id = "winner-full"
            scenarios = (
                ("after_sales", 150),
                ("image_product_search", 200),
                ("itinerary_planning", 100),
            )

            pilot_sample_ids = []
            for scenario, _ in scenarios:
                pilot_sample_ids.extend(f"{scenario}-pilot-{index}" for index in range(5))
            pilot_versions = ("standardized_v2", "fewshot_4_v2", "fewshot_7_v2")
            for run_id, version in zip(pilot_ids, pilot_versions):
                rows = [
                    self._record(run_id, sample_id, sample_id.rsplit("-pilot-", 1)[0], version)
                    for sample_id in pilot_sample_ids
                ]
                self._write_run(
                    output,
                    run_id,
                    rows,
                    artifact_hash,
                    "pilot",
                )

            full_rows = []
            for scenario, count in scenarios:
                full_rows.extend(
                    self._record(
                        full_id,
                        f"{scenario}-full-{index}",
                        scenario,
                        "standardized_v2",
                    )
                    for index in range(count)
                )
            self._write_run(
                output,
                full_id,
                full_rows,
                artifact_hash,
                "full",
            )
            full_hash = canonical_sha256(
                [row["sample_id"] for row in full_rows]
            )
            winners = {
                "after_sales": "standardized_v2",
                "image_product_search": "standardized_v2",
                "itinerary_planning": "standardized_v2",
            }
            comparisons = output / "comparisons"
            comparisons.mkdir(parents=True)
            self._write_json(
                comparisons / "pilot_comparison_v3.json",
                {
                    "selection_scope": "best_among_tested_candidates",
                    "evidence_status": (
                        "descriptive_only_test_gold_demo_contamination"
                    ),
                    "effect_claim_allowed": False,
                    "pilot_run_ids": pilot_ids,
                    "winners": winners,
                    "candidate_summaries": [
                        {"model_request_error_count": 0}
                        for _ in range(9)
                    ],
                },
            )
            self._write_json(
                comparisons / "selected_prompts_v3.json",
                winners,
            )
            self._write_json(
                comparisons / "full_baseline_comparison_v3.json",
                {
                    "full_run_id": full_id,
                    "common_semantic_comparison_id": "common-test",
                    "fewshot_evidence_status": (
                        "descriptive_only_test_gold_demo_contamination"
                    ),
                    "optimized_summaries": [
                        {"scenario": scenario, "sample_count": count}
                        for scenario, count in scenarios
                    ],
                    "bad_case_counts": {"classification_error": 1},
                    "baseline_comparison": [
                        {
                            "scenario": scenario,
                            "business_metrics_comparable": False,
                            "business_comparison_status": (
                                "not_comparable_different_prediction_encodings"
                            ),
                            "baseline_mean_total_tokens": None,
                            "baseline_token_status": "PENDING_not_recorded",
                        }
                        for scenario, _ in scenarios
                    ],
                },
            )
            self._write_jsonl(
                output / "scores" / full_id / "sample_scores.jsonl",
                [{"sample_id": row["sample_id"]} for row in full_rows],
            )
            self._write_jsonl(
                output / "bad_cases/week4_bad_cases_v3.jsonl",
                [
                    {
                        "sample_id": full_rows[0]["sample_id"],
                        "categories": ["classification_error"],
                    }
                ],
            )
            self._write_common_comparison(
                root,
                output,
                full_id,
                full_rows,
                scenarios,
            )
            config = {
                "paths": {"output_dir": "outputs/week4"},
                "validation": {
                    "artifact_version": "v3",
                    "fewshot_evidence_status": (
                        "descriptive_only_test_gold_demo_contamination"
                    ),
                    "common_semantic_comparison_id": "common-test",
                    "pilot_run_ids": pilot_ids,
                    "full_run_id": full_id,
                    "expected_full_sample_sha256": full_hash,
                    "expected_winners": winners,
                },
            }

            summary = validate_week4_delivery(root, config)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["full_record_count"], 450)
            self.assertEqual(summary["score_record_count"], 450)
            self.assertEqual(summary["bad_case_record_count"], 1)
            self.assertEqual(summary["model_request_error_count"], 0)
            self.assertEqual(
                summary["business_comparison_status"],
                "comparable_on_common_semantic_track",
            )
            self.assertEqual(summary["common_semantic_paired_count"], 450)

            from src.evaluation.week4_validation import Week4ValidationError

            failed_rows = [
                self._record(
                    pilot_ids[1],
                    sample_id,
                    sample_id.rsplit("-pilot-", 1)[0],
                    "fewshot_4_v2",
                )
                for sample_id in pilot_sample_ids
            ]
            failed_rows[0]["error"] = "model_request_error: HTTP 400"
            failed_rows[0]["raw_output"] = None
            failed_rows[0]["parsed_output"] = None
            failed_rows[0]["json_valid"] = False
            failed_rows[0]["schema_valid"] = False
            self._write_jsonl(
                output / "runs" / pilot_ids[1] / "results.jsonl",
                failed_rows,
            )
            with self.assertRaisesRegex(
                Week4ValidationError,
                "model request errors",
            ):
                validate_week4_delivery(root, config)

    @staticmethod
    def _record(run_id, sample_id, scenario, prompt_version):
        from src.evaluation.provenance import canonical_sha256

        input_metadata = {"sample_id": sample_id}
        return {
            "run_id": run_id,
            "sample_id": sample_id,
            "scenario": scenario,
            "mode": "live",
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "model_config": {},
            "prompt_version": prompt_version,
            "request_sha256": "a" * 64,
            "prompt_artifact_sha256": "b" * 64,
            "input_sha256": canonical_sha256(input_metadata),
            "input_metadata": input_metadata,
            "raw_output": "{}",
            "parsed_output": {},
            "json_valid": True,
            "schema_valid": True,
            "latency_ms": 1.0,
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "error": None,
            "timestamp": "2026-07-25T00:00:00+00:00",
        }

    @classmethod
    def _write_run(cls, output, run_id, rows, artifact_hash, run_scope):
        from src.evaluation.provenance import canonical_sha256

        run_dir = output / "runs" / run_id
        cls._write_jsonl(run_dir / "results.jsonl", rows)
        prompts = {}
        for row in rows:
            prompts[row["scenario"]] = row["prompt_version"]
        cls._write_json(
            run_dir / "metadata.json",
            {
                "run_id": run_id,
                "mode": "live",
                "prompt_version": rows[0]["prompt_version"],
                "prompt_versions_by_scenario": prompts,
                "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                "model_config": {},
                "dataset_version": "week3_evaluation_v2",
                "artifact_hashes": {"artifact.txt": artifact_hash},
                "selected_sample_ids_sha256": canonical_sha256(
                    [row["sample_id"] for row in rows]
                ),
                "selected_count": len(rows),
                "run_scope": run_scope,
                "status": "completed",
                "record_count": len(rows),
                "error": None,
            },
        )

    @classmethod
    def _write_common_comparison(
        cls,
        root,
        output,
        full_id,
        full_rows,
        scenarios,
    ):
        from src.evaluation.baseline_semantics import BaselineSemanticCoder
        from src.evaluation.provenance import canonical_sha256

        baseline_id = "week3_v2_baseline_full_20260724_001"
        baseline_rows = [
            {**row, "run_id": baseline_id, "prompt_version": "baseline_minimal_v1"}
            for row in full_rows
        ]
        baseline_path = (
            root
            / "data/eval/runs"
            / baseline_id
            / "results.jsonl"
        )
        cls._write_jsonl(baseline_path, baseline_rows)
        winner_path = output / "runs" / full_id / "results.jsonl"
        codebook_target = (
            root / "configs/evaluation/baseline_semantic_coding_v1.json"
        )
        codebook_target.parent.mkdir(parents=True, exist_ok=True)
        codebook_target.write_bytes(
            (
                Path(__file__).resolve().parents[1]
                / "configs/evaluation/baseline_semantic_coding_v1.json"
            ).read_bytes()
        )
        coder = BaselineSemanticCoder.from_path(codebook_target)
        common = output / "common_semantic/common-test"
        for name, run_id, source_rows in (
            ("baseline", baseline_id, baseline_rows),
            ("winner", full_id, full_rows),
        ):
            predictions = [
                coder.encode(
                    scenario=row["scenario"],
                    raw_output=row["raw_output"],
                )
                for row in source_rows
            ]
            cls._write_jsonl(
                common / f"{name}_canonical_predictions.jsonl",
                [
                    {
                        "run_id": run_id,
                        "sample_id": row["sample_id"],
                        "scenario": row["scenario"],
                        "prediction": prediction,
                    }
                    for row, prediction in zip(source_rows, predictions)
                ],
            )
            cls._write_jsonl(
                common / f"{name}_score/sample_scores.jsonl",
                [
                    {
                        "sample_id": row["sample_id"],
                        "scoring_track": "week4_common_semantic_coding_v1",
                        "deterministic_prediction": prediction,
                    }
                    for row, prediction in zip(source_rows, predictions)
                ],
            )
        metadata = {
            "comparison_id": "common-test",
            "scoring_track": "week4_common_semantic_coding_v1",
            "coding_version": "baseline_semantic_coding_v1",
            "baseline_run_id": baseline_id,
            "winner_run_id": full_id,
            "paired_sample_count": 450,
            "bootstrap_iterations": 2000,
            "gold_joined_after_prediction": True,
            "prediction_input_fields": ["scenario", "raw_output"],
            "selected_sample_ids_sha256": canonical_sha256(
                [row["sample_id"] for row in full_rows]
            ),
            "codebook_sha256": coder.codebook_sha256,
        }
        summary = {
            **metadata,
            "scenario_counts": {
                scenario: count for scenario, count in scenarios
            },
            "baseline_results_sha256": hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest(),
            "winner_results_sha256": hashlib.sha256(
                winner_path.read_bytes()
            ).hexdigest(),
        }
        cls._write_json(common / "summary.json", summary)
        cls._write_json(
            common / "paired_comparison/metadata.json",
            metadata,
        )
        aggregate = common / "paired_comparison/aggregate_deltas.csv"
        aggregate.parent.mkdir(parents=True, exist_ok=True)
        required = [
            ("image_product_search", "business_category_accuracy"),
            ("image_product_search", "price_range_accuracy"),
            ("after_sales", "issue_type_accuracy"),
            ("after_sales", "severity_accuracy"),
            ("after_sales", "ocr_recall"),
            ("itinerary_planning", "constraint_recognition_accuracy"),
            ("itinerary_planning", "itinerary_element_completeness"),
        ]
        aggregate.write_text(
            "scenario,metric\n"
            + "".join(f"{scenario},{metric}\n" for scenario, metric in required),
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _write_jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    unittest.main()
