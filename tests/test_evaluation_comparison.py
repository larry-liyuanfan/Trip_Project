import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class EvaluationComparisonTest(unittest.TestCase):
    def _score(self, sample_id: str, accuracy: float, *, latency: float) -> dict:
        return {
            "run_id": "baseline-run",
            "sample_id": sample_id,
            "scenario": "image_product_search",
            "model_name": "fixture",
            "prompt_version": "baseline_minimal_v1",
            "json_compliance": accuracy,
            "schema_pass": accuracy,
            "business_category_accuracy": accuracy,
            "latency_ms": latency,
            "structured_valid": bool(accuracy),
            "multilabel_counts": {},
        }

    def _metadata(self, prompt_version: str) -> dict:
        return {
            "run_id": "baseline-run" if prompt_version == "baseline_minimal_v1" else "standard-run",
            "mode": "live",
            "run_scope": "full",
            "prompt_version": prompt_version,
            "model_name": "fixture",
            "model_config": {"generation": {"temperature": 0}},
            "dataset_version": "week3-v1",
            "artifact_hashes": {
                "data/eval/manifests/product.jsonl": "a" * 64,
                "configs/evaluation/schemas/product.json": "b" * 64,
                f"configs/evaluation/prompts/{prompt_version}/product.txt": "c" * 64,
            },
            "selected_sample_ids_sha256": "d" * 64,
            "selected_count": 2,
            "record_count": 2,
            "status": "completed",
            "error": None,
        }

    def test_metadata_requires_same_live_dataset_samples_model_and_non_prompt_assets(self) -> None:
        from src.evaluation.comparison import (
            ComparisonValidationError,
            validate_comparable_runs,
        )

        baseline = self._metadata("baseline_minimal_v1")
        standardized = self._metadata("standardized_v1")
        validate_comparable_runs(baseline, standardized)

        standardized["selected_sample_ids_sha256"] = "e" * 64
        with self.assertRaisesRegex(ComparisonValidationError, "sample set"):
            validate_comparable_runs(baseline, standardized)

    def test_metadata_rejects_pilot_runs_for_final_comparison(self) -> None:
        from src.evaluation.comparison import ComparisonValidationError, validate_comparable_runs

        baseline = self._metadata("baseline_minimal_v1")
        standardized = self._metadata("standardized_v1")
        baseline["run_scope"] = "pilot"
        with self.assertRaisesRegex(ComparisonValidationError, "full-scope"):
            validate_comparable_runs(baseline, standardized)

    def test_paired_comparison_computes_delta_direction_and_win_tie_loss(self) -> None:
        from src.evaluation.comparison import compare_score_records

        baseline = [
            self._score("sample-a", 0.0, latency=100.0),
            self._score("sample-b", 1.0, latency=120.0),
        ]
        standardized = [
            {**self._score("sample-a", 1.0, latency=90.0), "run_id": "standard-run"},
            {**self._score("sample-b", 1.0, latency=130.0), "run_id": "standard-run"},
        ]

        sample_rows, aggregate_rows = compare_score_records(
            baseline,
            standardized,
            bootstrap_iterations=100,
            bootstrap_seed=7,
        )

        self.assertEqual(len(sample_rows), 2)
        category = next(
            row
            for row in aggregate_rows
            if row["metric"] == "business_category_accuracy"
        )
        self.assertEqual(category["baseline_mean"], 0.5)
        self.assertEqual(category["standardized_mean"], 1.0)
        self.assertEqual(category["absolute_delta"], 0.5)
        self.assertEqual(category["standardized_wins"], 1)
        self.assertEqual(category["ties"], 1)
        latency = next(row for row in aggregate_rows if row["metric"] == "latency_ms")
        self.assertEqual(latency["standardized_wins"], 1)
        self.assertEqual(latency["baseline_wins"], 1)

    def test_comparison_rejects_different_sample_sets(self) -> None:
        from src.evaluation.comparison import (
            ComparisonValidationError,
            compare_score_records,
        )

        with self.assertRaisesRegex(ComparisonValidationError, "sample IDs"):
            compare_score_records(
                [self._score("sample-a", 1.0, latency=1.0)],
                [self._score("sample-b", 1.0, latency=1.0)],
            )

    def test_comparison_export_is_immutable_and_strict(self) -> None:
        from src.evaluation.comparison import (
            ComparisonValidationError,
            export_comparison_artifacts,
        )

        sample_rows = [
            {
                "sample_id": "sample-a",
                "scenario": "image_product_search",
                "metrics": {
                    "business_category_accuracy": {
                        "baseline": 0.0,
                        "standardized": 1.0,
                        "delta": 1.0,
                    }
                },
            }
        ]
        aggregate_rows = [
            {
                "scenario": "image_product_search",
                "metric": "business_category_accuracy",
                "paired_count": 1,
                "baseline_mean": 0.0,
                "standardized_mean": 1.0,
                "absolute_delta": 1.0,
                "relative_delta": None,
                "baseline_wins": 0,
                "standardized_wins": 1,
                "ties": 0,
                "delta_ci95_low": 1.0,
                "delta_ci95_high": 1.0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison-1"
            paths = export_comparison_artifacts(
                output,
                {"comparison_id": "comparison-1"},
                sample_rows,
                aggregate_rows,
            )
            self.assertEqual(json.loads(paths["metadata"].read_text())["comparison_id"], "comparison-1")
            with self.assertRaisesRegex(ComparisonValidationError, "already exists"):
                export_comparison_artifacts(
                    output,
                    {"comparison_id": "comparison-1"},
                    sample_rows,
                    aggregate_rows,
                )

    def test_report_and_representative_cases_are_generated_from_comparison_rows(self) -> None:
        from src.evaluation.comparison import (
            generate_comparison_report,
            select_representative_cases,
        )

        sample_rows = [
            {
                "sample_id": "sample-a",
                "scenario": "image_product_search",
                "metrics": {
                    "business_category_accuracy": {
                        "baseline": 0.0,
                        "standardized": 1.0,
                        "delta": 1.0,
                    }
                },
            },
            {
                "sample_id": "sample-b",
                "scenario": "image_product_search",
                "metrics": {
                    "business_category_accuracy": {
                        "baseline": 1.0,
                        "standardized": 0.0,
                        "delta": -1.0,
                    }
                },
            },
        ]
        aggregate_rows = [
            {
                "scenario": "image_product_search",
                "metric": "business_category_accuracy",
                "paired_count": 2,
                "baseline_mean": 0.5,
                "standardized_mean": 0.5,
                "absolute_delta": 0.0,
                "relative_delta": 0.0,
                "baseline_wins": 1,
                "standardized_wins": 1,
                "ties": 0,
                "delta_ci95_low": -1.0,
                "delta_ci95_high": 1.0,
            }
        ]
        cases = select_representative_cases(sample_rows, per_direction=1)
        self.assertEqual(
            {(case["direction"], case["sample_id"]) for case in cases},
            {("improvement", "sample-a"), ("regression", "sample-b")},
        )
        report = generate_comparison_report(
            {
                "comparison_id": "comparison-1",
                "baseline_run_id": "baseline-run",
                "standardized_run_id": "standard-run",
                "dataset_version": "week3-v1",
                "paired_sample_count": 2,
            },
            aggregate_rows,
            cases,
        )
        self.assertIn("baseline-run", report)
        self.assertIn("standard-run", report)
        self.assertIn("business_category_accuracy", report)
        self.assertIn("sample-a", report)

    def test_compare_command_writes_immutable_artifacts_and_generated_report(self) -> None:
        from scripts.compare_week3_evaluation import compare_runs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_metadata = self._metadata("baseline_minimal_v1")
            standardized_metadata = self._metadata("standardized_v1")
            for metadata in (baseline_metadata, standardized_metadata):
                run_dir = root / "runs" / metadata["run_id"]
                run_dir.mkdir(parents=True)
                (run_dir / "metadata.json").write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
            baseline_scores = root / "scores/baseline-run"
            standardized_scores = root / "scores/standard-run"
            baseline_scores.mkdir(parents=True)
            standardized_scores.mkdir(parents=True)
            baseline_rows = [
                self._score("sample-a", 0.0, latency=100.0),
                self._score("sample-b", 1.0, latency=100.0),
            ]
            standardized_rows = [
                {**self._score("sample-a", 1.0, latency=90.0), "run_id": "standard-run"},
                {**self._score("sample-b", 1.0, latency=90.0), "run_id": "standard-run"},
            ]
            (baseline_scores / "sample_scores.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_rows),
                encoding="utf-8",
            )
            (standardized_scores / "sample_scores.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in standardized_rows),
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "runs_dir": "runs",
                    "scores_dir": "scores",
                    "comparisons_dir": "comparisons",
                    "generated_reports_dir": "generated_reports",
                }
            }
            with (
                patch(
                    "scripts.compare_week3_evaluation.load_evaluation_config",
                    return_value=config,
                ),
                patch(
                    "scripts.compare_week3_evaluation.verify_artifact_hashes",
                    return_value=None,
                ),
            ):
                summary = compare_runs(
                    root=root,
                    config_path=Path("config.yaml"),
                    comparison_id="comparison-1",
                    baseline_run_id="baseline-run",
                    standardized_run_id="standard-run",
                    bootstrap_iterations=100,
                )

            self.assertEqual(summary["paired_sample_count"], 2)
            self.assertTrue((root / "comparisons/comparison-1/metadata.json").is_file())
            self.assertTrue((root / "generated_reports/comparison-1/report.md").is_file())


if __name__ == "__main__":
    unittest.main()
