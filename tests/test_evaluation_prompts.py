import json
import unittest
from pathlib import Path


class EvaluationPromptTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SCENARIOS = (
        "image_product_search",
        "after_sales",
        "itinerary_planning",
    )
    BASELINE_TEXT = {
        "image_product_search": "识别图片所展示的旅游商品特征。",
        "after_sales": "识别图片中与旅游售后相关的问题。",
        "itinerary_planning": "根据参考图片和文字约束识别用户的行程需求。",
    }

    def _multi_image_context(self, *, text_constraints=None):
        return {
            "images": [
                {"path": "data/eval/images/first.jpg", "sha256": "a" * 64},
                {"path": "data/eval/images/second.jpg", "sha256": "b" * 64},
            ],
            "text_constraints": text_constraints,
        }

    def test_minimal_baselines_remain_exact_separate_single_sentence_tasks(self):
        from src.evaluation.prompting import load_baseline_prompt

        forbidden_fragments = (
            "你是",
            "角色",
            "JSON",
            "字段",
            "格式",
            "reasoning",
            "思维链",
            "首先",
            "然后",
            "示例",
        )
        for scenario, expected in self.BASELINE_TEXT.items():
            with self.subTest(scenario=scenario):
                prompt = load_baseline_prompt(self.PROJECT_ROOT, scenario)
                self.assertEqual(prompt, expected)
                for fragment in forbidden_fragments:
                    self.assertNotIn(fragment, prompt)

    def test_baseline_request_uses_ordered_multimodal_parts_without_changing_prompt(self):
        from src.evaluation.prompting import render_baseline_request

        rendered = render_baseline_request(
            self.PROJECT_ROOT,
            "itinerary_planning",
            self._multi_image_context(text_constraints="两天行程；第二天18:00前结束"),
        )

        self.assertEqual(rendered["prompt_version"], "baseline_minimal_v1")
        self.assertEqual(rendered["messages"][0]["role"], "user")
        content = rendered["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], [
            "text", "text", "image_url", "text", "image_url", "text"
        ])
        self.assertEqual(content[0]["text"], self.BASELINE_TEXT["itinerary_planning"])
        self.assertEqual(content[1]["text"], "参考图片占位符 <image_1>")
        self.assertEqual(content[2]["image_url"]["url"], "file://data/eval/images/first.jpg")
        self.assertEqual(content[3]["text"], "参考图片占位符 <image_2>")
        self.assertEqual(content[4]["image_url"]["url"], "file://data/eval/images/second.jpg")
        self.assertEqual(content[5]["text"], "原始文字约束：两天行程；第二天18:00前结束")

    def test_standard_request_uses_same_image_and_text_constraint_order(self):
        from src.evaluation.prompting import render_standard_prompt

        rendered = render_standard_prompt(
            self.PROJECT_ROOT,
            "itinerary_planning",
            self._multi_image_context(text_constraints="两天行程"),
        )
        content = rendered["messages"][1]["content"]
        image_parts = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(
            [part["image_url"]["url"] for part in image_parts],
            ["file://data/eval/images/first.jpg", "file://data/eval/images/second.jpg"],
        )
        placeholders = [
            part["text"] for part in content
            if part["type"] == "text" and part["text"].startswith("参考图片占位符")
        ]
        self.assertEqual(placeholders, [
            "参考图片占位符 <image_1>",
            "参考图片占位符 <image_2>",
        ])
        self.assertLess(
            next(index for index, part in enumerate(content) if part.get("text") == placeholders[-1]),
            next(index for index, part in enumerate(content) if part.get("text", "").startswith("原始文字约束：")),
        )
        self.assertEqual(
            [part["text"] for part in content if part.get("text", "").startswith("原始文字约束：")],
            ["原始文字约束：两天行程"],
        )

    def test_text_constraints_follow_scenario_contract(self):
        from src.evaluation.prompting import PromptConfigurationError, render_baseline_request

        invalid_itinerary = self._multi_image_context(text_constraints=None)
        with self.assertRaisesRegex(PromptConfigurationError, "itinerary_planning"):
            render_baseline_request(
                self.PROJECT_ROOT,
                "itinerary_planning",
                invalid_itinerary,
            )

        invalid_image_only = self._multi_image_context(text_constraints="不应出现")
        with self.assertRaisesRegex(PromptConfigurationError, "image_product_search"):
            render_baseline_request(
                self.PROJECT_ROOT,
                "image_product_search",
                invalid_image_only,
            )

    def test_image_count_is_validated_by_scenario(self):
        from src.evaluation.prompting import PromptConfigurationError, render_baseline_request

        for scenario in ("image_product_search", "after_sales"):
            for count in (0, 2):
                with self.subTest(scenario=scenario, image_count=count):
                    context = self._multi_image_context(text_constraints=None)
                    context["images"] = context["images"][:count]
                    with self.assertRaisesRegex(PromptConfigurationError, "exactly one"):
                        render_baseline_request(self.PROJECT_ROOT, scenario, context)

        itinerary = self._multi_image_context(text_constraints="两天行程")
        itinerary["images"] = []
        with self.assertRaisesRegex(PromptConfigurationError, "at least one"):
            render_baseline_request(
                self.PROJECT_ROOT,
                "itinerary_planning",
                itinerary,
            )

    def test_standard_prompt_renders_four_layers_and_exposes_complete_schema(self):
        from src.evaluation.prompting import render_standard_prompt

        input_context = self._multi_image_context()
        input_context["images"] = input_context["images"][:1]
        rendered = render_standard_prompt(
            self.PROJECT_ROOT,
            "image_product_search",
            input_context,
        )

        self.assertEqual(rendered["prompt_version"], "standardized_v1")
        self.assertEqual(
            set(rendered["layers"]),
            {"system_role", "task_instruction", "input_context", "output_constraint"},
        )
        self.assertEqual(json.loads(rendered["layers"]["input_context"]), input_context)
        self.assertEqual([message["role"] for message in rendered["messages"]], ["system", "user"])
        self.assertEqual(rendered["messages"][0]["content"], rendered["layers"]["system_role"])
        self.assertEqual(rendered["output_schema"]["$id"], "image_product_search_v1.schema.json")
        self.assertIn('"business_category"', rendered["layers"]["output_constraint"])
        self.assertIn('"additionalProperties":false', rendered["layers"]["output_constraint"])
        request_text = "\n".join(
            part["text"] for part in rendered["messages"][1]["content"]
            if part["type"] == "text"
        )
        self.assertIn(rendered["layers"]["task_instruction"], request_text)
        self.assertIn(rendered["layers"]["output_constraint"], request_text)

    def test_standard_prompts_have_common_rules_without_language_enum_conflict(self):
        from src.evaluation.prompting import render_standard_prompt

        required_fragments = (
            "中文",
            "英文枚举",
            "不得编造",
            "未知",
            "观察",
            "推断",
            "隐私",
            "安全",
            "仅输出",
            "observed_evidence",
            "简短",
        )
        one_image = [
            {"path": "data/eval/images/example.jpg", "sha256": "a" * 64}
        ]
        contexts = {
            "image_product_search": {"images": one_image, "text_constraints": None},
            "after_sales": {"images": one_image, "text_constraints": None},
            "itinerary_planning": {
                "images": one_image,
                "text_constraints": "两天行程",
            },
        }
        for scenario in self.SCENARIOS:
            with self.subTest(scenario=scenario):
                rendered = render_standard_prompt(
                    self.PROJECT_ROOT,
                    scenario,
                    contexts[scenario],
                )
                combined = "\n".join(rendered["layers"].values())
                for fragment in required_fragments:
                    self.assertIn(fragment, combined)
                self.assertNotIn("reasoning", combined.lower())
                self.assertNotIn("首先", combined)
                self.assertNotIn("然后", combined)
                self.assertIn(f"{scenario}_v1.schema.json", rendered["layers"]["output_constraint"])


if __name__ == "__main__":
    unittest.main()
