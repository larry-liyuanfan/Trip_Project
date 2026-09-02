import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_system_repair_final_test import audit_final_test
from scripts.audit_system_repair_development import _file_sha


class SystemRepairFinalTestAuditTests(unittest.TestCase):
    def test_reconciles_recorded_composite_and_recomputes_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "test.jsonl"
            raw = root / "raw.jsonl"
            metrics = root / "metrics.json"
            gate = root / "gate.json"
            consumption = root / "consumption.json"
            dataset.write_text(json.dumps({
                "sample_id": "p1",
                "scenario": "image_product_search",
                "target": {
                    "business_category": "hotel",
                    "style_tags": ["modern"],
                    "visible_facilities": ["pool"],
                    "price_range": "unknown",
                    "unknown_fields": ["price_range"],
                },
            }) + "\n", encoding="utf-8")
            raw.write_text(json.dumps({
                "sample_id": "p1",
                "run_id": "final-once",
                "failed": False,
                "raw_output": json.dumps({
                    "business_category": "hotel",
                    "style_tags": ["modern"],
                    "visible_facilities": [],
                    "price_range": "budget",
                }),
            }) + "\n", encoding="utf-8")
            metrics.write_text(json.dumps({
                "status": "COMPLETED",
                "split": "test",
                "sample_count": 1,
                "run_id": "final-once",
                "raw_outputs": {"count": 1, "sha256": _file_sha(raw)},
                "scenarios": {"image_product_search": {"composite": 0.7806388888888889}},
                "dialogue": {},
                "adapter_hashes": {"adapter_model.safetensors": "adapter"},
            }), encoding="utf-8")
            gate.write_text('{"status":"PASS"}', encoding="utf-8")
            consumption.write_text('{"status":"CONSUMED"}', encoding="utf-8")
            report = audit_final_test(
                dataset, raw, metrics, gate, consumption,
                implementation_commit_sha="implementation",
                run_source_snapshot_sha256="snapshot",
            )
        self.assertEqual(
            report["historical_image_product_composite"]["reported_six_decimals"],
            0.780639,
        )
        self.assertEqual(
            report["error_slices"]["field_mismatch"]["visible_facilities"]["count"],
            1,
        )
        self.assertEqual(report["error_slices"]["unknown_field_hallucination"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
