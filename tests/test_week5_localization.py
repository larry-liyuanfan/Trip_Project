from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.data.week5_localization import export_localized_annotations, localized_value


class Week5LocalizationTests(unittest.TestCase):
    def test_localizes_keys_and_enums_but_preserves_free_text(self) -> None:
        self.assertEqual(
            localized_value(
                {"issue_type": "hygiene_stain", "observed_evidence": ["wooden table"]}
            ),
            {"问题类型": "卫生污渍（hygiene_stain）", "可观察证据": ["wooden table"]},
        )

    def test_export_keeps_canonical_hash_and_latest_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_dir = root / "outputs/week5/annotations"
            annotation_dir.mkdir(parents=True)
            canonical = {"issue_type": "unknown", "observed_evidence": ["证据"]}
            rows = [
                {"sample_id": "s1", "revision": 1, "human_annotation": {}},
                {"sample_id": "s1", "revision": 2, "human_annotation": canonical},
            ]
            (annotation_dir / "after_sales.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            for scenario in ("image_product_search", "itinerary_planning"):
                (annotation_dir / f"{scenario}.jsonl").write_text("", encoding="utf-8")
            counts = export_localized_annotations(
                root, {"paths": {"output_dir": "outputs/week5"}}
            )
            self.assertEqual(counts["after_sales"], 1)
            exported = json.loads(
                (root / "outputs/week5/localized_annotations/after_sales.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
            self.assertEqual(
                exported["canonical_annotation_sha256"],
                hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(exported["annotation_revision"], 2)


if __name__ == "__main__":
    unittest.main()
