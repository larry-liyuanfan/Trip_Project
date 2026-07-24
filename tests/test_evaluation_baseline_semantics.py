import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.evaluation.baseline_semantics import BaselineSemanticCoder
from src.evaluation.metrics import score_semantic_prediction
from src.evaluation.provenance import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEBOOK_PATH = (
    PROJECT_ROOT / "configs/evaluation/baseline_semantic_coding_v1.json"
)


class BaselineSemanticCoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coder = BaselineSemanticCoder.from_path(CODEBOOK_PATH)

    def test_encoder_signature_rejects_annotation_and_gold(self) -> None:
        for forbidden in (
            {"annotation": {"issue_type": "facility_damage"}},
            {"sampling_stratum": "facility_damage"},
            {"source_metadata": {"category": "facility_damage"}},
        ):
            with self.assertRaises(TypeError):
                self.coder.encode(
                    scenario="after_sales",
                    raw_output="broken chair",
                    **forbidden,
                )

    def test_changing_gold_cannot_change_prediction(self) -> None:
        prediction = self.coder.encode(
            scenario="after_sales",
            raw_output="The chair is broken.",
        )
        for unused_gold in (
            {"issue_type": "facility_damage"},
            {"issue_type": "hygiene_stain"},
        ):
            self.assertEqual(
                self.coder.encode(
                    scenario="after_sales",
                    raw_output="The chair is broken.",
                ),
                prediction,
            )
            self.assertIn("issue_type", unused_gold)

    def test_fixed_input_is_deterministic(self) -> None:
        first = self.coder.encode(
            scenario="image_product_search",
            raw_output="A quiet hotel lobby with a pool.",
        )
        second = self.coder.encode(
            scenario="image_product_search",
            raw_output="A quiet hotel lobby with a pool.",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["business_category"], "hotel")
        self.assertEqual(first["style_tags"], ["quiet"])
        self.assertEqual(first["visible_facilities"], ["lobby", "pool"])

    def test_ambiguous_scalar_returns_unknown(self) -> None:
        prediction = self.coder.encode(
            scenario="image_product_search",
            raw_output="The image may show a hotel restaurant.",
        )
        self.assertEqual(prediction["business_category"], "unknown")

    def test_unmatched_text_does_not_guess(self) -> None:
        prediction = self.coder.encode(
            scenario="after_sales",
            raw_output="No classifiable lexical evidence.",
        )
        self.assertEqual(prediction["issue_type"], "unknown")
        self.assertEqual(prediction["severity"], "unknown")
        self.assertEqual(prediction["key_information"], [])

    def test_ocr_negation_returns_null(self) -> None:
        prediction = self.coder.encode(
            scenario="after_sales",
            raw_output="The image has no visible text.",
        )
        self.assertIsNone(prediction["ocr_text"])

    def test_semantic_track_preserves_format_failures(self) -> None:
        result = _result_record(raw_output="The chair is broken.")
        prediction = self.coder.encode(
            scenario="after_sales",
            raw_output=result["raw_output"],
        )
        score = score_semantic_prediction(
            result,
            {
                "issue_type": "facility_damage",
                "severity": "unknown",
                "key_information": ["damaged facility"],
                "ocr_ground_truth": None,
            },
            prediction,
            {},
            coding_version=self.coder.version,
            codebook_sha256=self.coder.codebook_sha256,
        )
        self.assertEqual(score["json_compliance"], 0.0)
        self.assertEqual(score["schema_pass"], 0.0)
        self.assertFalse(score["structured_valid"])
        self.assertEqual(score["issue_type_accuracy"], 1.0)
        self.assertEqual(score["scoring_track"], "baseline_semantic_coding_v1")

    def test_semantic_track_rejects_non_baseline_prompt(self) -> None:
        result = {**_result_record(raw_output="broken"), "prompt_version": "standardized_v2"}
        with self.assertRaisesRegex(ValueError, "baseline_minimal_v1"):
            score_semantic_prediction(
                result,
                {},
                {"issue_type": "facility_damage"},
                {},
                coding_version=self.coder.version,
                codebook_sha256=self.coder.codebook_sha256,
            )

    def test_score_command_is_immutable(self) -> None:
        from scripts.score_week3_evaluation import score_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs/run-1"
            run_dir.mkdir(parents=True)
            result = _result_record(raw_output="The chair is broken.")
            (run_dir / "results.jsonl").write_text(
                json.dumps(result) + "\n", encoding="utf-8"
            )
            metadata = {
                "run_id": "run-1",
                "mode": "live",
                "prompt_version": "baseline_minimal_v1",
                "model_name": "fixture",
                "model_config": {},
                "dataset_version": "fixture_v1",
                "artifact_hashes": {"fixture.txt": "a" * 64},
                "selected_sample_ids_sha256": canonical_sha256(["after-sales-1"]),
                "selected_count": 1,
                "status": "completed",
                "record_count": 1,
                "error": None,
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (root / "aliases.json").write_text(
                json.dumps({"version": "test", "fields": {}}), encoding="utf-8"
            )
            codebook = root / "codebook.json"
            codebook.write_bytes(CODEBOOK_PATH.read_bytes())
            config = {
                "paths": {"runs_dir": "runs", "scores_dir": "scores"},
                "metrics": {"aliases_path": "aliases.json"},
            }
            manifests = {
                "after_sales": [
                    {
                        "sample_id": "after-sales-1",
                        "scenario": "after_sales",
                        "annotation_status": "completed",
                        "annotation": {
                            "issue_type": "facility_damage",
                            "severity": "unknown",
                            "key_information": ["damaged facility"],
                            "ocr_ground_truth": None,
                        },
                    }
                ]
            }
            with _patched_score_dependencies(config, manifests):
                summary = score_run(
                    root=root,
                    config_path=Path("config.yaml"),
                    run_id="run-1",
                    semantic_coding_config=Path("codebook.json"),
                    score_id="run-1__baseline_semantic_coding_v1",
                )
                with self.assertRaises(FileExistsError):
                    score_run(
                        root=root,
                        config_path=Path("config.yaml"),
                        run_id="run-1",
                        semantic_coding_config=Path("codebook.json"),
                        score_id="run-1__baseline_semantic_coding_v1",
                    )
            self.assertEqual(
                summary["scoring_track"], "baseline_semantic_coding_v1"
            )
            aggregate = (
                root
                / "scores/run-1__baseline_semantic_coding_v1/aggregate_scores.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("issue_type_accuracy_support_count", aggregate)


def _result_record(*, raw_output: str) -> dict:
    return {
        "run_id": "run-1",
        "sample_id": "after-sales-1",
        "scenario": "after_sales",
        "mode": "live",
        "model_name": "fixture",
        "model_config": {},
        "prompt_version": "baseline_minimal_v1",
        "input_metadata": {"images": [{"path": "fixture.jpg", "sha256": "a" * 64}]},
        "request_sha256": "b" * 64,
        "raw_output": raw_output,
        "parsed_output": None,
        "json_valid": False,
        "schema_valid": False,
        "latency_ms": 1.0,
        "error": "json_parse_error",
        "timestamp": "2026-07-25T00:00:00+00:00",
    }


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
