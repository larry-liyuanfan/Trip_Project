from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.week7_selection import select_development_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week7/qwen3_vl_8b_multitask_context_v2.json"
CONFIG_SHA = "f08f31066a35f216f3294782b1dea23d321bff321193ecfef6d241a4467e07f6"
LOCK_SHA = "daabf6d225f52408f38097c379077a5b09fe94090e6ed897ba631c2f3e85e014"


def _scenario(composite: float, latency: float = 100.0) -> dict:
    return {
        "composite": composite,
        "aggregate": {"json_compliance": 1.0, "schema_pass": 1.0, "latency_mean_ms": latency},
        "metric_support": {"metric": 30},
    }


class Week7SelectionTests(unittest.TestCase):
    def test_higher_composite_with_task_regression_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = {
                "status": "COMPLETED", "model_role": "week6_single_task_adapters",
                "split": "development", "config_sha256": CONFIG_SHA,
                "dataset_lock_sha256": LOCK_SHA, "failure_rate": 0.0,
                "scenarios": {name: _scenario(0.5) for name in (
                    "image_product_search", "after_sales", "itinerary_planning",
                )},
            }
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            training = root / "training"
            weights = {"image_product_search": 0.30, "after_sales": 0.35, "itinerary_planning": 0.35}
            for step, values in ((38, (0.55, 0.55, 0.50)), (76, (0.9, 0.9, 0.2))):
                metrics_dir = training / "development_evaluations" / f"step-{step:06d}"
                checkpoint = training / f"checkpoint-{step}"
                metrics_dir.mkdir(parents=True)
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
                scenarios = dict(zip(weights, (_scenario(value) for value in values)))
                metrics = {
                    "sample_count": 114, "dialogue": {"sample_count": 24},
                    "scenarios": scenarios, "failure_rate": 0.0,
                    "weighted_composite": sum(weights[name] * scenarios[name]["composite"] for name in weights),
                }
                (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            output = root / "selection.json"
            result = select_development_checkpoint(CONFIG, training, baseline_path, output)
            self.assertEqual(result["status"], "SELECTED")
            self.assertEqual(result["selected"]["step"], 38)
            self.assertFalse(next(item for item in result["candidates"] if item["step"] == 76)["eligible"])


if __name__ == "__main__":
    unittest.main()
