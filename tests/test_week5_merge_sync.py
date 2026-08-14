import json
import tempfile
import unittest
from pathlib import Path

from src.data.week5_dataset import Week5DataError
from src.data.week5_merge_sync import sync_merged_preannotations


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class Week5MergeSyncTest(unittest.TestCase):
    def test_preserves_reviewed_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            merge = root / "merge"
            merge.mkdir()
            (merge / "summary.json").write_text(
                json.dumps({"unique_success": 3}), encoding="utf-8"
            )
            rows = []
            for scenario in (
                "image_product_search", "after_sales", "itinerary_planning"
            ):
                rows.append({
                    "sample_id": f"sample-{scenario}",
                    "scenario": scenario,
                    "status": "completed",
                    "schema_valid": True,
                    "parsed_output": {"source": "merge"},
                })
            _write(merge / "results.jsonl", rows)
            preserved = root / "preserved"
            reviewed = {**rows[0], "parsed_output": {"source": "reviewed"}}
            _write(preserved / "image_product_search.jsonl", [reviewed])

            result = sync_merged_preannotations(
                merge_dir=merge,
                preserved_dir=preserved,
                output_dir=root / "output",
            )

            self.assertEqual(result["unique_success"], 3)
            self.assertEqual(result["preserved_human_review_rows"], 1)
            written = json.loads(
                (root / "output/image_product_search.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(written["parsed_output"], {"source": "reviewed"})

    def test_rejects_preserved_id_absent_from_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            merge = root / "merge"
            merge.mkdir()
            (merge / "summary.json").write_text(
                json.dumps({"unique_success": 0}), encoding="utf-8"
            )
            _write(merge / "results.jsonl", [])
            _write(root / "preserved/image_product_search.jsonl", [{
                "sample_id": "missing",
                "scenario": "image_product_search",
                "status": "completed",
                "schema_valid": True,
            }])

            with self.assertRaises(Week5DataError):
                sync_merged_preannotations(
                    merge_dir=merge,
                    preserved_dir=root / "preserved",
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
