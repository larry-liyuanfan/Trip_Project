from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.protected_identity_inventory import audit_inventory
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


class ProtectedIdentityInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = {
            "sample_id": "sample-one", "source_id": "source-one", "split": "test",
            "image_sha256": "a" * 64, "query_sha256": "b" * 64,
            "source_record_sha256": "c" * 64,
        }

    def config(self, rows=None, *, json_array=False):
        rows = [self.row] if rows is None else rows
        path = self.root / ("identity.json" if json_array else "identity.jsonl")
        content = json.dumps(rows) if json_array else "\n".join(json.dumps(row) for row in rows)
        path.write_text(content + "\n", encoding="utf-8")
        return {"protected_identity_inventory": [{
            "id": "protected", "scope": "unit_test_fixture_not_real_test",
            "file": path.name, "file_sha256": file_sha256(path), "expected_rows": len(rows),
            "required_fields": ["sample_id", "source_id", "image_sha256", "query_sha256",
                                "source_record_sha256"],
        }]}

    def test_ready_inventory_never_authorizes_model_or_claims_zero_overlap(self):
        report = audit_inventory(self.config(), self.root)
        self.assertEqual(report["status"], "INVENTORY_READY_ONLY")
        self.assertFalse(report["model_execution_authorized"])
        self.assertIsNone(report["overlap_count"])
        self.assertEqual(report["registries"][0]["fields"]["query_sha256"]["nonempty_rows"], 1)

    def test_sample_only_week6_like_registry_is_blocked(self):
        report = audit_inventory(self.config([{"sample_id": "one", "split": "train"}]), self.root)
        self.assertEqual(report["status"], "BLOCKED_PROTECTED_IDENTITY_COVERAGE")
        self.assertEqual(report["registries"][0]["fields"]["source_id"]["nonempty_rows"], 0)

    def test_group_or_template_id_is_not_query_hash(self):
        row = {**self.row, "query_sha256": None, "group_id": "g", "constraint_template_id": "t"}
        report = audit_inventory(self.config([row]), self.root)
        self.assertEqual(report["registries"][0]["missing_or_partial_dimensions"], ["query_sha256"])

    def test_partial_dimension_not_promoted_to_complete(self):
        row = {**self.row, "sample_id": "two", "query_sha256": ""}
        report = audit_inventory(self.config([self.row, row]), self.root)
        self.assertEqual(report["registries"][0]["fields"]["query_sha256"], {
            "nonempty_rows": 1, "unique_values": 1, "row_denominator": 2,
        })
        self.assertTrue(report["blocking_registry_ids"])

    def test_no_identity_values_are_returned(self):
        report = json.dumps(audit_inventory(self.config(), self.root))
        self.assertNotIn("sample-one", report)
        self.assertNotIn("source-one", report)
        self.assertNotIn("a" * 64, report)

    def test_missing_registry_blocks_without_fallback(self):
        config = self.config()
        config["protected_identity_inventory"][0]["file"] = "not_present.jsonl"
        self.assertTrue(audit_inventory(config, self.root)["blocking_registry_ids"])

    def test_aggregate_lock_or_unlocated_registry_not_enough(self):
        config = {"protected_identity_inventory": [{"id": "v5", "scope": "historical", "file": None}]}
        self.assertEqual(audit_inventory(config, self.root)["registries"][0]["status"], "UNAVAILABLE")

    def test_file_digest_mismatch_rejected(self):
        config = self.config()
        config["protected_identity_inventory"][0]["file_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            audit_inventory(config, self.root)

    def test_canonical_digest_binding_for_asset_registry(self):
        row = {"record_id": "one", "source_id": "source", "sha256": "a" * 64,
               "query_sha256": "b" * 64, "source_record_sha256": "c" * 64,
               "contains_image_bytes": False}
        config = self.config([row], json_array=True)
        spec = config["protected_identity_inventory"][0]
        spec.pop("file_sha256")
        spec["canonical_sha256"] = canonical_json_sha256([row])
        spec["field_aliases"] = {"sample_id": "record_id", "image_sha256": "sha256"}
        self.assertFalse(audit_inventory(config, self.root)["blocking_registry_ids"])
        spec["canonical_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
            audit_inventory(config, self.root)

    def test_hash_binding_required(self):
        config = self.config()
        config["protected_identity_inventory"][0].pop("file_sha256")
        with self.assertRaisesRegex(ValueError, "pre-existing SHA"):
            audit_inventory(config, self.root)

    def test_row_count_and_split_support_enforced(self):
        config = self.config()
        bad = copy.deepcopy(config)
        bad["protected_identity_inventory"][0]["expected_rows"] = 2
        with self.assertRaisesRegex(ValueError, "row support"):
            audit_inventory(bad, self.root)
        config["protected_identity_inventory"][0]["expected_splits"] = {"train": 1}
        with self.assertRaisesRegex(ValueError, "split support"):
            audit_inventory(config, self.root)

    def test_non_identity_and_nested_metadata_rejected(self):
        for field, value in [("gold", {}), ("prompt", "answer"), ("prediction", "answer"),
                             ("source_uri", {"raw": "nested"})]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    audit_inventory(self.config([{**self.row, field: value}]), self.root)

    def test_image_bytes_claim_and_bad_digest_rejected(self):
        for update in [{"contains_image_bytes": True}, {"query_sha256": "not-a-sha"}]:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    audit_inventory(self.config([{**self.row, **update}]), self.root)

    def test_query_cannot_alias_image_or_id(self):
        config = self.config()
        config["protected_identity_inventory"][0]["field_aliases"] = {"query_sha256": "image_sha256"}
        with self.assertRaisesRegex(ValueError, "aliases"):
            audit_inventory(config, self.root)

    def test_directory_escape_rejected(self):
        config = self.config()
        config["protected_identity_inventory"][0]["file"] = "../elsewhere.jsonl"
        with self.assertRaisesRegex(ValueError, "escapes"):
            audit_inventory(config, self.root)

    def test_empty_or_duplicate_inventory_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            audit_inventory({"protected_identity_inventory": []}, self.root)
        config = self.config()
        config["protected_identity_inventory"] *= 2
        with self.assertRaisesRegex(ValueError, "unique"):
            audit_inventory(config, self.root)

    def test_cli_nonzero_on_block_and_no_overwrite(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config([{"sample_id": "one"}])), encoding="utf-8")
        output = self.root / "report.json"
        command = [sys.executable, "scripts/audit_price_conflict_prerequisites.py", "--config",
                   str(config_path), "--identity-root", str(self.root), "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 2, first.stderr)
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_preregistration_keeps_training_hyperparameters_and_discloses_confound(self):
        root = Path(__file__).resolve().parents[1]
        old = json.loads((root / "configs/evaluation/automated_evidence_v9.json").read_text(encoding="utf-8"))
        new = json.loads((root / "configs/evaluation/price_conflict_abstention_20260914.json").read_text(encoding="utf-8"))
        for key in ("optimizer", "learning_rate", "lr_scheduler_type", "warmup_ratio", "weight_decay",
                    "max_grad_norm", "gradient_accumulation_steps", "epochs", "max_length",
                    "logging_steps", "attn_implementation", "seed"):
            self.assertEqual(old["training"][key], new["training"][key], key)
        for split in ("training", "development"):
            counts = new["planned_data_not_yet_generated"][split]
            self.assertEqual(sum(v for k, v in counts.items() if k != "total"), counts["total"])
        self.assertIn("confounded", new["causal_claim"])
        self.assertEqual(new["training"]["fixed_baseline_additional_optimizer_steps"], 0)
        self.assertEqual(new["resource_limits"]["max_gpu_jobs"], 1)
        self.assertEqual(new["final_policy"], "not_defined_not_generated_not_consumed")


if __name__ == "__main__":
    unittest.main()
