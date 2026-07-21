import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class EvaluationProvenanceTest(unittest.TestCase):
    def test_artifact_hashes_detect_changed_files(self) -> None:
        from src.evaluation.provenance import (
            ProvenanceValidationError,
            build_artifact_hashes,
            verify_artifact_hashes,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "manifest.jsonl"
            second = root / "prompt.txt"
            first.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            second.write_text("识别图片。\n", encoding="utf-8")

            hashes = build_artifact_hashes(root, [first, second])
            self.assertEqual(
                hashes,
                {
                    "manifest.jsonl": hashlib.sha256(first.read_bytes()).hexdigest(),
                    "prompt.txt": hashlib.sha256(second.read_bytes()).hexdigest(),
                },
            )
            verify_artifact_hashes(root, hashes)

            first.write_text('{"sample_id":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ProvenanceValidationError,
                "artifact hash mismatch.*manifest.jsonl",
            ):
                verify_artifact_hashes(root, hashes)

    def test_canonical_hash_is_stable_and_order_sensitive_for_lists(self) -> None:
        from src.evaluation.provenance import canonical_sha256

        first = canonical_sha256({"b": 2, "a": ["x", "y"]})
        second = canonical_sha256({"a": ["x", "y"], "b": 2})
        reversed_list = canonical_sha256({"a": ["y", "x"], "b": 2})
        self.assertEqual(first, second)
        self.assertNotEqual(first, reversed_list)

    def test_build_run_artifact_hashes_covers_manifests_prompts_and_schemas(self) -> None:
        from src.evaluation.provenance import build_run_artifact_hashes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "paths": {"exclusion_manifest": "data/eval/registry/exclusion.jsonl"},
                "scenarios": {
                    scenario: {
                        "manifest_path": f"data/eval/manifests/{scenario}.jsonl"
                    }
                    for scenario in (
                        "image_product_search",
                        "after_sales",
                        "itinerary_planning",
                    )
                },
            }
            required = [
                root / "data/eval/registry/exclusion.jsonl",
                *[
                    root / f"data/eval/manifests/{scenario}.jsonl"
                    for scenario in config["scenarios"]
                ],
                *[
                    root
                    / f"configs/evaluation/prompts/baseline_minimal_v1/{scenario}.txt"
                    for scenario in config["scenarios"]
                ],
                *[
                    root / f"configs/evaluation/schemas/{scenario}_v1.schema.json"
                    for scenario in config["scenarios"]
                ],
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")

            hashes = build_run_artifact_hashes(
                root,
                config,
                "baseline_minimal_v1",
            )

            self.assertEqual(set(hashes), {path.relative_to(root).as_posix() for path in required})
            json.dumps(hashes)


if __name__ == "__main__":
    unittest.main()
