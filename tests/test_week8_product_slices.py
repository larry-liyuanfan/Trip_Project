import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.summarize_week8_product_slices import run, summarize


class ProductSliceTests(unittest.TestCase):
    def test_empty_style_references_still_count_false_positive(self):
        target = {"business_category": "unknown", "style_tags": [], "visible_facilities": [], "price_range": "unknown"}
        references = [{"sample_id": "s", "label_source": "model_generated_silver", "target": target,
                       "observation": {"subject_kind": "food_closeup"}}]
        baseline = {**target, "style_tags": ["modern"], "visible_facilities": ["parking"], "price_range": "budget"}
        result = summarize(references, {"baseline": {"s": baseline}, "candidate": {"s": target}})
        for key in ("food_closeup", "style_missing_or_extra", "facility_missing_or_extra", "price_without_evidence", "should_use_unknown"):
            self.assertEqual(result["error_slices"][key]["baseline"], {"support": 1, "errors": 1})
            self.assertEqual(result["error_slices"][key]["candidate"], {"support": 1, "errors": 0})
        self.assertFalse(result["human_accuracy_claim"])

    def test_cli_replays_incumbent_with_its_own_observation_identity(self):
        target = {"business_category": "unknown", "style_tags": [], "visible_facilities": [], "price_range": "unknown"}
        reference = {"sample_id": "s", "label_source": "model_generated_silver", "target": target}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = {"references": reference, "baseline": {"sample_id": "s"}, "candidate": {"sample_id": "s"},
                      "observation": {"identity": "candidate"}, "baseline_observation": {"identity": "incumbent"}}
            for name, value in values.items():
                (directory / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
            for explicit in (False, True):
                args = argparse.Namespace(**{name: directory / name for name in values if name != "baseline_observation"},
                    baseline_observation=directory / "baseline_observation" if explicit else None,
                    output=directory / f"result-{explicit}.json")
                with patch("scripts.summarize_week8_product_slices.replay_record", return_value=target) as replay, patch("builtins.print"):
                    run(args)
                self.assertEqual(replay.call_args_list[0].args[2], values["baseline_observation"] if explicit else None)
                self.assertEqual(replay.call_args_list[1].args[2], values["observation"])
                result = json.loads(args.output.read_text(encoding="utf-8"))
                self.assertEqual("baseline_observation" in result["source_sha256"], explicit)
                self.assertEqual(result["error_slices"]["all_samples"]["candidate"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
