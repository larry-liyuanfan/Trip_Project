from __future__ import annotations

import json
import gzip
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.week6_image_manifest import (
    audit_shard,
    build_manifest,
    merge_audits,
    pack_failures,
    split_file,
)


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

    def test_pack_failures_preserves_project_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images" / "sample.bin"
            source.parent.mkdir()
            source.write_bytes(b"image")
            failure_list = root / "failures.txt.gz"
            with gzip.open(failure_list, "wt", encoding="utf-8") as handle:
                handle.write("images/sample.bin\n")
            archive = root / "recovery.tar.gz"
            payload = pack_failures(
                failure_list=failure_list,
                project_root=root,
                output=archive,
                summary=root / "summary.json",
            )
            self.assertEqual(payload["records"], 1)
            with tarfile.open(archive, "r:gz") as handle:
                self.assertEqual(handle.getnames(), ["images/sample.bin"])

    def test_split_file_records_ordered_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"abcdefg")
            payload = split_file(
                source=source,
                output_dir=root / "parts",
                part_size_bytes=3,
                summary=root / "split.json",
            )
            self.assertEqual(
                [part["name"] for part in payload["parts"]],
                ["part-0000", "part-0001", "part-0002"],
            )
            self.assertEqual(
                b"".join((root / "parts" / part["name"]).read_bytes() for part in payload["parts"]),
                b"abcdefg",
            )


if __name__ == "__main__":
    unittest.main()
