from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.protected_identity_inventory import (
    IMPORT_CANONICALIZATION,
    audit_inventory,
    validate_identity_only_import,
)
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


class IdentityOnlyImportTests(unittest.TestCase):
    """All bundle contents are tiny synthetic fixtures, never project identities."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "synthetic_bundle"
        self.bundle.mkdir()

    def build_fixture(self, scope_count=12, *, text_scope=None):
        text_scope = scope_count - 1 if text_scope is None else text_scope
        sources = []
        scopes = []
        scope_ids = []
        for index in range(scope_count):
            scope_id = f"synthetic_scope_{index:02d}"
            scope_ids.append(scope_id)
            text_only = index == text_scope
            required = [
                "sample_id", "source_id", "source_record_sha256", "query_sha256",
                "content_sha256", "dialogue_text_sha256" if text_only else "image_sha256",
            ]
            source_manifest_sha = f"{index + 1:064x}"
            data_lock_sha = f"{index + 101:064x}"
            sources.append({
                "scope_id": scope_id,
                "source_manifest_file_sha256": source_manifest_sha,
                "data_lock_file_sha256": data_lock_sha,
                "required_fields": required,
                "image_identity_policy": "not_applicable_text_only" if text_only else "required",
            })
            row = {
                "sample_id": f"synthetic-sample-{index}",
                "source_id": f"synthetic-source-{index}",
                "source_record_sha256": f"{index + 201:064x}",
                "query_sha256": f"{index + 301:064x}",
                "content_sha256": f"{index + 401:064x}",
                "dialogue_text_sha256" if text_only else "image_sha256": f"{index + 501:064x}",
                "contains_image_bytes": False,
            }
            identity = self.bundle / f"scope-{index:02d}.jsonl"
            identity.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            scopes.append({
                "scope_id": scope_id,
                "path": identity.name,
                "file_sha256": file_sha256(identity),
                "canonical_rows_sha256": canonical_json_sha256([row]),
                "row_count": 1,
                "source_manifest_file_sha256": source_manifest_sha,
                "data_lock_file_sha256": data_lock_sha,
            })
        approval = {
            "schema_version": "identity_import_approved_sources_v1",
            "canonicalization_version": IMPORT_CANONICALIZATION,
            "approval_id": "synthetic-unit-test-approval-not-custodian-evidence",
            "required_scope_ids": scope_ids,
            "sources": sources,
        }
        approved_path = self.root / "synthetic_approved_sources.json"
        approved_path.write_text(json.dumps(approval, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "identity_only_export_manifest_v1",
            "canonicalization_version": IMPORT_CANONICALIZATION,
            "export_id": "synthetic-unit-test-export-not-real-data",
            "approval_id": approval["approval_id"],
            "approved_sources_file_sha256": file_sha256(approved_path),
            "scopes": scopes,
        }
        manifest_path = self.bundle / "export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return approval, approved_path, manifest, manifest_path

    def validate(self, approved_path, manifest_path, scope_count=12):
        return validate_identity_only_import(
            bundle_root=self.bundle,
            export_manifest_sha256=file_sha256(manifest_path),
            approved_sources_path=approved_path,
            approved_sources_sha256=file_sha256(approved_path),
            expected_scope_count=scope_count,
        )

    def rewrite_json(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def test_complete_twelve_scope_fixture_validates_but_never_authorizes(self):
        _, approved, _, manifest = self.build_fixture()
        report = self.validate(approved, manifest)
        self.assertEqual(report["status"], "VALIDATED_IMPORT_NOT_TRAINING_AUTHORIZATION")
        self.assertEqual((report["scope_support"], report["scope_denominator"]), (12, 12))
        self.assertEqual(report["row_support"], 12)
        self.assertFalse(report["model_execution_authorized"])
        self.assertTrue(report["coordinator_release_required"])
        self.assertFalse(report["custodian_attestation_verified"])
        self.assertIsNone(report["human_annotation_support"])
        self.assertEqual(report["leakage_check_status"], "NOT_RUN")
        self.assertIsNone(report["overlap_count"])

    def test_unbound_approval_registry_is_rejected(self):
        _, approved, _, manifest = self.build_fixture()
        with self.assertRaisesRegex(ValueError, "approved source registry file SHA-256 mismatch"):
            validate_identity_only_import(
                bundle_root=self.bundle,
                export_manifest_sha256=file_sha256(manifest),
                approved_sources_path=approved,
                approved_sources_sha256="f" * 64,
            )

    def test_unknown_canonicalization_is_rejected_in_both_trust_layers(self):
        approval, approved, export, manifest = self.build_fixture()
        approval["canonicalization_version"] = "invented-normalization"
        self.rewrite_json(approved, approval)
        with self.assertRaisesRegex(ValueError, "approved canonicalization"):
            self.validate(approved, manifest)
        approval["canonicalization_version"] = IMPORT_CANONICALIZATION
        export["canonicalization_version"] = "invented-normalization"
        self.rewrite_json(approved, approval)
        export["approved_sources_file_sha256"] = file_sha256(approved)
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "export canonicalization"):
            self.validate(approved, manifest)

    def test_export_manifest_must_bind_the_approved_registry(self):
        _, approved, export, manifest = self.build_fixture()
        export["approved_sources_file_sha256"] = "f" * 64
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "not bound to the approved source registry"):
            self.validate(approved, manifest)

    def test_missing_scope_is_rejected(self):
        _, approved, export, manifest = self.build_fixture()
        export["scopes"].pop()
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "cover every approved scope"):
            self.validate(approved, manifest)

    def test_self_reported_source_hash_cannot_replace_approved_identity(self):
        _, approved, export, manifest = self.build_fixture()
        export["scopes"][0]["source_manifest_file_sha256"] = "f" * 64
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "is not approved"):
            self.validate(approved, manifest)

    def test_file_and_canonical_hashes_are_recomputed(self):
        _, approved, export, manifest = self.build_fixture()
        identity = self.bundle / export["scopes"][0]["path"]
        identity.write_text(identity.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch"):
            self.validate(approved, manifest)
        export["scopes"][0]["file_sha256"] = file_sha256(identity)
        export["scopes"][0]["canonical_rows_sha256"] = "f" * 64
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "canonical rows SHA-256 mismatch"):
            self.validate(approved, manifest)

    def test_missing_required_field_and_raw_payload_are_rejected(self):
        for mutation, message in [
            (lambda row: row.pop("query_sha256"), "missing an approved required field"),
            (lambda row: row.update({"prompt": "forbidden raw prompt"}), "non-identity fields"),
            (lambda row: row.update({"contains_image_bytes": True}), "claims image bytes"),
        ]:
            with self.subTest(message=message):
                _, approved, export, manifest = self.build_fixture()
                identity = self.bundle / export["scopes"][0]["path"]
                row = json.loads(identity.read_text(encoding="utf-8"))
                mutation(row)
                identity.write_text(json.dumps(row) + "\n", encoding="utf-8")
                export["scopes"][0]["file_sha256"] = file_sha256(identity)
                export["scopes"][0]["canonical_rows_sha256"] = canonical_json_sha256([row])
                self.rewrite_json(manifest, export)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(approved, manifest)

    def test_duplicate_and_conflicting_sample_ids_are_rejected(self):
        _, approved, export, manifest = self.build_fixture()
        first_path = self.bundle / export["scopes"][0]["path"]
        first = json.loads(first_path.read_text(encoding="utf-8"))
        first_path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n", encoding="utf-8")
        export["scopes"][0].update(
            file_sha256=file_sha256(first_path), canonical_rows_sha256=canonical_json_sha256([first, first]),
            row_count=2,
        )
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "duplicate sample_id within"):
            self.validate(approved, manifest)

        _, approved, export, manifest = self.build_fixture()
        second_path = self.bundle / export["scopes"][1]["path"]
        second = json.loads(second_path.read_text(encoding="utf-8"))
        second["sample_id"] = "synthetic-sample-0"
        second_path.write_text(json.dumps(second) + "\n", encoding="utf-8")
        export["scopes"][1].update(
            file_sha256=file_sha256(second_path), canonical_rows_sha256=canonical_json_sha256([second]),
        )
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate sample_id"):
            self.validate(approved, manifest)

    def test_duplicate_json_keys_are_rejected(self):
        _, approved, export, manifest = self.build_fixture()
        identity = self.bundle / export["scopes"][0]["path"]
        identity.write_text('{"sample_id":"a","sample_id":"b"}\n', encoding="utf-8")
        export["scopes"][0]["file_sha256"] = file_sha256(identity)
        export["scopes"][0]["canonical_rows_sha256"] = canonical_json_sha256([{"sample_id": "b"}])
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            self.validate(approved, manifest)

    def test_text_only_policy_requires_dialogue_digest_and_forbids_image(self):
        approval, approved, export, manifest = self.build_fixture(scope_count=2, text_scope=1)
        approval["sources"][1]["required_fields"].remove("dialogue_text_sha256")
        self.rewrite_json(approved, approval)
        with self.assertRaisesRegex(ValueError, "lacks its required digest"):
            self.validate(approved, manifest, scope_count=2)
        approval, approved, export, manifest = self.build_fixture(scope_count=2, text_scope=1)
        identity = self.bundle / export["scopes"][1]["path"]
        row = json.loads(identity.read_text(encoding="utf-8"))
        row["image_sha256"] = "f" * 64
        identity.write_text(json.dumps(row) + "\n", encoding="utf-8")
        export["scopes"][1].update(
            file_sha256=file_sha256(identity), canonical_rows_sha256=canonical_json_sha256([row]),
        )
        self.rewrite_json(manifest, export)
        with self.assertRaisesRegex(ValueError, "unexpectedly contains image"):
            self.validate(approved, manifest, scope_count=2)

    def test_cli_exit_codes_and_exclusive_output(self):
        _, approved, _, manifest = self.build_fixture()
        output = self.root / "validation.json"
        command = [
            sys.executable, "scripts/validate_identity_only_import.py",
            "--bundle-root", str(self.bundle),
            "--export-manifest-sha256", file_sha256(manifest),
            "--approved-sources", str(approved),
            "--approved-sources-sha256", file_sha256(approved),
            "--output", str(output),
        ]
        success = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["model_execution_authorized"])
        original = output.read_bytes()
        overwrite = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(overwrite.returncode, 0)
        self.assertEqual(output.read_bytes(), original)
        failed_output = self.root / "failed.json"
        command[command.index("--approved-sources-sha256") + 1] = "f" * 64
        command[command.index("--output") + 1] = str(failed_output)
        failed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(failed.returncode, 2)
        self.assertFalse(failed_output.exists())
        self.assertFalse(json.loads(failed.stderr)["model_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
