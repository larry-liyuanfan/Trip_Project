from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.week6_image_manifest import audit_shard, build_manifest, merge_audits


class Week6ImageManifestTests(unittest.TestCase):
    def test_build_and_audit_use_deterministic_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "a.bin").write_bytes(b"a")
            (image_dir / "b.bin").write_bytes(b"bb")
            data = root / "train.jsonl"
            rows = [
                {
                    "messages": [
                        {
                            "content": [
                                {"type": "image", "path": f"images/{name}"}
                            ]
                        }
                    ]
                }
                for name in ("b.bin", "a.bin", "a.bin")
            ]
            data.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            payload = build_manifest(
                inputs=[data], project_root=root, output=manifest, summary=summary
            )
            self.assertEqual(payload["records"], 2)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual([record["path"] for record in records], ["images/a.bin", "images/b.bin"])
            shard = audit_shard(
                manifest=manifest,
                project_root=root,
                shard_index=0,
                shard_count=2,
                output=root / "audit.json",
            )
            self.assertEqual(shard["status"], "ok")
            self.assertEqual(shard["checked"], 1)

    def test_audit_reports_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"path": "missing.bin", "size_bytes": 1, "sha256": "0" * 64})
                + "\n",
                encoding="utf-8",
            )
            payload = audit_shard(
                manifest=manifest,
                project_root=root,
                shard_index=0,
                shard_count=1,
                output=root / "audit.json",
            )
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["failures"][0]["reason"], "missing")

    def test_merge_requires_every_shard_and_no_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                (root / f"audit-{index}.json").write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "shard_index": index,
                            "checked": 2,
                            "failures": [],
                        }
                    ),
                    encoding="utf-8",
                )
            payload = merge_audits(
                input_dir=root, shard_count=2, output=root / "merged.json"
            )
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["checked"], 4)


if __name__ == "__main__":
    unittest.main()
