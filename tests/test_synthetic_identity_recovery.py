from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.recover_synthetic_train_dev_identities import (
    VLM_QUERY_SCHEMA,
    _require_allowed_split,
    _search_identity,
    _vlm_identity,
    _cross_scope_overlap_check,
    _write_jsonl,
)
from src.evaluation.relevance_evidence import canonical_json_sha256


class SyntheticIdentityRecoveryTests(unittest.TestCase):
    def test_rejects_final_split_before_recovery(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbids non-train/dev"):
            _require_allowed_split("final")

    def test_search_export_contains_only_flat_identity_metadata(self) -> None:
        source = {"source_id": "synthetic:v4:training:q1"}
        row = {
            "query_id": "q1",
            "split": "training",
            "image": {"sha256": "1" * 64},
            "source": source,
            "source_record_sha256": canonical_json_sha256(source),
            "query_sha256": "2" * 64,
            "requested_filters": {"city": "Austin"},
        }
        identity = _search_identity(row, "v4")
        self.assertEqual(identity["content_sha256"], "1" * 64)
        self.assertFalse(identity["contains_image_bytes"])
        self.assertNotIn("requested_filters", identity)
        self.assertTrue(all(not isinstance(value, (dict, list)) for value in identity.values()))

    def test_vlm_query_hash_binds_prompt_and_content_without_exporting_them(self) -> None:
        source_sha = "3" * 64
        row = {
            "sample_id": "d1",
            "source_id": "synthetic:v5:development:d1",
            "source_record_sha256": source_sha,
            "split": "development",
            "scenario": "dialogue",
            "prompt": "Return JSON.",
            "dialogue": "Newest city is Austin.",
            "gold": {"city": "Austin"},
        }
        identity = _vlm_identity(row, "v5")
        content_sha = canonical_json_sha256({"dialogue": row["dialogue"]})
        expected_query = canonical_json_sha256(
            {
                "schema_version": VLM_QUERY_SCHEMA,
                "scenario": "dialogue",
                "prompt": row["prompt"],
                "content_sha256": content_sha,
            }
        )
        self.assertEqual(identity["dialogue_text_sha256"], content_sha)
        self.assertEqual(identity["content_sha256"], content_sha)
        self.assertEqual(identity["query_sha256"], expected_query)
        self.assertNotIn("prompt", identity)
        self.assertNotIn("dialogue", identity)
        self.assertNotIn("gold", identity)

    def test_cross_scope_overlap_check_fails_closed(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = [{"sample_id": "same", "source_id": "source-a"}]
            second = [{"sample_id": "same", "source_id": "source-b"}]
            _write_jsonl(root / "first.jsonl", first)
            _write_jsonl(root / "second.jsonl", second)
            scopes = [
                {"scope_id": "first", "identity_path": "first.jsonl"},
                {"scope_id": "second", "identity_path": "second.jsonl"},
            ]
            with self.assertRaisesRegex(ValueError, "cross-scope identity collision"):
                _cross_scope_overlap_check(scopes, root)


if __name__ == "__main__":
    unittest.main()
