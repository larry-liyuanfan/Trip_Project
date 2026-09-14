"""Tests for immutable relevance source snapshot manifest creation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_relevance_source_snapshot import build_snapshot_manifest
from scripts.verify_relevance_source_snapshot import canonical_sha256, validate_snapshot


class BuildRelevanceSourceSnapshotTests(unittest.TestCase):
    def test_manifest_round_trips_through_fail_closed_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "snapshot"
            root.mkdir()
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            output = Path(temp_dir) / "manifest.json"
            manifest = build_snapshot_manifest(
                project_root=root,
                output=output,
                git_base_sha="a" * 40,
                implementation_commit="b" * 40,
            )
            verified = validate_snapshot(
                root,
                output,
                canonical_sha256(manifest),
                "b" * 40,
            )
            self.assertEqual(verified["status"], "PASS")
            with self.assertRaises(FileExistsError):
                build_snapshot_manifest(
                    project_root=root,
                    output=output,
                    git_base_sha="a" * 40,
                    implementation_commit="b" * 40,
                )


if __name__ == "__main__":
    unittest.main()
