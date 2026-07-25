import json
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
            pilot_versions = ("standardized_v2", "fewshot_4_v1", "fewshot_7_v1")
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
                comparisons / "pilot_comparison_v1.json",
                {
                    "selection_scope": "best_among_tested_candidates",
                    "pilot_run_ids": pilot_ids,
                    "winners": winners,
                    "candidate_summaries": [{} for _ in range(9)],
                },
            )
            self._write_json(
                comparisons / "selected_prompts_v1.json",
                winners,
            )
            self._write_json(
                comparisons / "full_baseline_comparison_v1.json",
                {
                    "full_run_id": full_id,
                    "optimized_summaries": [
                        {"scenario": scenario, "sample_count": count}
                        for scenario, count in scenarios
                    ],
                    "bad_case_counts": {"classification_error": 1},
                },
            )
            self._write_jsonl(
                output / "scores" / full_id / "sample_scores.jsonl",
                [{"sample_id": row["sample_id"]} for row in full_rows],
            )
            self._write_jsonl(
                output / "bad_cases/week4_bad_cases_v1.jsonl",
                [
                    {
                        "sample_id": full_rows[0]["sample_id"],
                        "categories": ["classification_error"],
                    }
                ],
            )
            config = {
                "paths": {"output_dir": "outputs/week4"},
                "validation": {
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
