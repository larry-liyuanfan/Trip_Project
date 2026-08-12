from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.spartan_migration import (
    merge_spartan_migration,
    prepare_spartan_migration,
    spartan_migration_status,
)
from src.data.week5_dataset import SCENARIOS, Week5DataError, candidate_payload_sha256


class SpartanMigrationTests(unittest.TestCase):
    def _fixture(self, directory: str) -> tuple[Path, dict, Path]:
        root = Path(directory)
        output = root / "outputs/week5"
        pools = output / "pools"
        pools.mkdir(parents=True)
        config = {
            "dataset_version": "week5_instruction_candidates_v1",
            "paths": {"output_dir": "outputs/week5"},
            "targets": {
                "image_product_search": 3,
                "after_sales": 2,
                "itinerary_planning": 1,
                "dialogues": 0,
            },
            "final_minimums": {
                "image_product_search": 0,
                "after_sales": 0,
                "itinerary_planning": 0,
                "dialogues": 0,
            },
            "runtime": {"base_url": "http://old/v1", "concurrency": 1},
        }
        candidates = []
        for scenario in SCENARIOS:
            rows = []
            for index in range(config["targets"][scenario]):
                row = {
                    "sample_id": f"{scenario}-{index}",
                    "scenario": scenario,
                    "input": {"images": [], "text_constraints": None},
                }
                rows.append(row)
                candidates.append(row)
            (pools / f"{scenario}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        source = output / "runs/source-run"
        source.mkdir(parents=True)
        source_manifest = {
            "identity": {
                "candidate_manifests": {
                    scenario: {"path": f"outputs/week5/pools/{scenario}.jsonl"}
                    for scenario in SCENARIOS
                }
            }
        }
        (source / "run_manifest.json").write_text(
            json.dumps(source_manifest), encoding="utf-8"
        )
        first = candidates[0]
        result = {
            "sample_id": first["sample_id"],
            "scenario": first["scenario"],
            "status": "completed",
            "schema_valid": True,
            "candidate_sha256": candidate_payload_sha256(first),
        }
        (source / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
        (source / "failures.jsonl").write_text("", encoding="utf-8")
        return root, config, source

    def test_prepare_creates_disjoint_complete_inputs_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, source = self._fixture(directory)
            destination = root / "outputs/migration"
            manifest = prepare_spartan_migration(
                root,
                config,
                source_run_dir=source,
                output_dir=destination,
                migration_id="migration-1",
                shard_count=2,
                benchmark_count=2,
            )
            self.assertEqual(manifest["coverage"], {
                "base_success": 1, "benchmark": 2, "shards": 3, "total": 6
            })
            ids = set()
            for unit in [manifest["benchmark"], *manifest["shards"]]:
                for scenario in SCENARIOS:
                    path = root / unit["path"] / "pools" / f"{scenario}.jsonl"
                    for line in path.read_text(encoding="utf-8").splitlines():
                        sample_id = json.loads(line)["sample_id"]
                        self.assertNotIn(sample_id, ids)
                        ids.add(sample_id)
            self.assertEqual(len(ids), 5)
            with self.assertRaises(Week5DataError):
                prepare_spartan_migration(
                    root,
                    config,
                    source_run_dir=source,
                    output_dir=destination,
                    migration_id="migration-1",
                )

    def test_status_and_merge_report_only_actual_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, source = self._fixture(directory)
            migration = root / "outputs/migration"
            manifest = prepare_spartan_migration(
                root,
                config,
                source_run_dir=source,
                output_dir=migration,
                migration_id="migration-2",
                shard_count=1,
                benchmark_count=1,
            )
            benchmark_root = root / manifest["benchmark"]["path"]
            benchmark_run = benchmark_root / "runs" / manifest["run_ids"]["benchmark"]
            benchmark_run.mkdir(parents=True)
            benchmark_candidate = next(
                json.loads(line)
                for scenario in SCENARIOS
                for line in (benchmark_root / "pools" / f"{scenario}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
            benchmark_result = {
                "sample_id": benchmark_candidate["sample_id"],
                "scenario": benchmark_candidate["scenario"],
                "status": "completed",
                "schema_valid": True,
                "candidate_sha256": candidate_payload_sha256(benchmark_candidate),
            }
            (benchmark_run / "results.jsonl").write_text(
                json.dumps(benchmark_result) + "\n", encoding="utf-8"
            )
            (benchmark_run / "failures.jsonl").write_text("", encoding="utf-8")
            status = spartan_migration_status(root, migration)
            self.assertEqual(status["unique_success"], 2)
            self.assertEqual(status["remaining"], 4)
            merged = merge_spartan_migration(root, migration, root / "outputs/merged")
            self.assertEqual(merged["status"], "partial")
            self.assertEqual(merged["unique_success"], 2)
            self.assertFalse(merged["human_completion"])


if __name__ == "__main__":
    unittest.main()
