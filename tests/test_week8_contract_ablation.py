import json
from pathlib import Path
import unittest

from scripts.review_week8_contracts import build_requests
from src.evaluation.prompting import render_standard_prompt

ROOT = Path(__file__).resolve().parents[1]


class ContractAblationTests(unittest.TestCase):
    def test_prompt_has_real_schema_and_no_null_evidence_instruction(self):
        rendered = render_standard_prompt(ROOT, "itinerary_planning", {
            "images": [{"path": "fixture.jpg"}], "text_constraints": "上海两日行程"},
            "week8_itinerary_actionable_v1")
        self.assertIn('"constraint_check"', rendered["layers"]["output_constraint"])
        self.assertNotIn("evidence 使用 null", rendered["layers"]["task_instruction"])
        self.assertIn("城市来自用户文字", rendered["layers"]["task_instruction"])
        self.assertIn("不能伪造 satisfied", rendered["layers"]["output_constraint"])

    def test_product_prompt_requires_visual_not_merchant_evidence(self):
        rendered = render_standard_prompt(ROOT, "image_product_search", {
            "images": [{"path": "fixture.jpg"}], "text_constraints": None},
            "week8_product_visual_facts_v3")
        self.assertIn("不使用商家属性", rendered["layers"]["system_role"])
        self.assertIn("装修档次不是价格证据", rendered["layers"]["task_instruction"])
        self.assertIn('"visible_facilities"', rendered["layers"]["output_constraint"])

    def test_requests_use_locked_image_path_without_target_metadata(self):
        rows = [{"sample_id": "dev01", "image_path": "images/a.jpg",
                 "target": {"parking": True}, "business_description": "secret-reference"}]
        requests = build_requests(ROOT, rows, ["上海两日行程"])
        self.assertEqual(requests[0][2].image_urls, [str(ROOT / "images/a.jpg")])
        self.assertEqual(requests[1][2].text_context, "上海两日行程")
        self.assertNotIn("secret-reference", str(requests))
        self.assertEqual(len(requests), 2)

    def test_pilot_is_diagnostic_development_not_a_final_run(self):
        config = json.loads((ROOT / "configs/week8/contract_ablation_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(config["final_test_access"])
        self.assertEqual(config["human_annotation_count"], 0)
        self.assertEqual(config["selection_policy"], "development_diagnostic_only")
        self.assertEqual(len(config["development_indices"]), len(set(config["development_indices"])))


if __name__ == "__main__":
    unittest.main()
