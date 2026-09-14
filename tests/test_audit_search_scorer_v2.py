import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.audit_search_scorer_v2 import checked_file, verify_bindings
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256
from src.evaluation.search_scorer_v2 import require_versioned_inference_protocol
from test_search_scorer_v2 import fixture


class AuditTests(unittest.TestCase):
    def test_changed_prediction_identity_cannot_be_recomputed(self):
        queries, annotations, results, _ = fixture()
        old = {"query_manifest_sha256": canonical_json_sha256(queries),
               "annotation_sha256": canonical_json_sha256(annotations),
               "result_sha256": canonical_json_sha256(results)}
        verify_bindings(queries, annotations, results, old)
        results[0]["methods"]["m"]["hits"].clear()
        with self.assertRaisesRegex(ValueError, "result_sha256 mismatch"):
            verify_bindings(queries, annotations, results, old)

    def test_file_sha_failure_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}", encoding="utf-8")
            checked_file(path, file_sha256(path))
            with self.assertRaisesRegex(ValueError, "historical file SHA mismatch"):
                checked_file(path, "a" * 64)

    def test_old_protocol_cannot_reconsume_test_with_new_scorer(self):
        with self.assertRaisesRegex(ValueError, "historical inference protocol is frozen"):
            require_versioned_inference_protocol({"search": {}})
        require_versioned_inference_protocol({"search": {"scorer_version": "search_scorer_v2"}})

    def test_cli_valid_qrels_missing_qrels_and_exclusive_output(self):
        queries, annotations, results, qrels = fixture(human=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (("queries", queries), ("annotations", annotations), ("results", results)):
                (root / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (root / "qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
            config = {"search": {"query_manifest": str(root / "queries.jsonl"), "weak_annotations": str(root / "annotations.jsonl")}}
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            command = [sys.executable, "scripts/run_relevance_evidence.py", "--config", str(root / "config.json"),
                       "score-search", "--results", str(root / "results.jsonl"), "--output", str(root / "scored.json")]
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("full qrels", missing.stderr)
            command += ["--qrels", str(root / "qrels.json")]
            valid = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            report = json.loads((root / "scored.json").read_text(encoding="utf-8"))
            self.assertEqual(report["evidence_class"], "human_declared_qrels_identity_unverified")
            self.assertEqual(report["schema_version"], "search_scorer_v2")
            self.assertIn("git_sha", report["provenance"])
            original_sha = file_sha256(root / "scored.json")
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(file_sha256(root / "scored.json"), original_sha)


if __name__ == "__main__":
    unittest.main()
