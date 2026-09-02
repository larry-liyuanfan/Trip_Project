import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_relevance_source_snapshot import canonical_sha256, validate_snapshot


class RelevanceSourceSnapshotTests(unittest.TestCase):
    def test_validation_is_byte_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "runner.py"
            source.write_text("print('ok')\n", encoding="utf-8")
            manifest = {
                "schema_version": "relevance_source_snapshot_v1",
                "git_base_sha": "base",
                "implementation_commit_sha": "implementation",
                "files": [{
                    "path": "runner.py",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }],
            }
            manifest_path = root / "source_snapshot.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            expected = canonical_sha256(manifest)
            report = validate_snapshot(root, manifest_path, expected)
            self.assertEqual(report["status"], "PASS")
            source.write_text("print('changed')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
                validate_snapshot(root, manifest_path, expected)


if __name__ == "__main__":
    unittest.main()
