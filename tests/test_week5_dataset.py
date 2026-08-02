import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.data.week5_dataset import (
    Week5DataError,
    _check_candidate_isolation,
    load_week5_config,
    qc_audit_selected,
    validate_dialogue,
    validate_human_annotation,
    write_jsonl_new,
)
from src.data.week5_workflow import apply_human_corrections, apply_quality_records


ROOT = Path(__file__).resolve().parents[1]


class Week5DatasetTests(unittest.TestCase):
    def _workflow_fixture(self, directory: str) -> tuple[Path, dict, str]:
        root = Path(directory)
        (root / "configs/evaluation/schemas").mkdir(parents=True)
        shutil.copy2(
            ROOT / "configs/evaluation/schemas/image_product_search_v1.schema.json",
            root / "configs/evaluation/schemas/image_product_search_v1.schema.json",
        )
        (root / "configs/week5").mkdir(parents=True)
        shutil.copy2(
            ROOT / "configs/week5/annotation_tool.json",
            root / "configs/week5/annotation_tool.json",
        )
        sample_id = "week5-image_product_search-test"
        pool = root / "outputs/week5/pools/image_product_search.jsonl"
        pool.parent.mkdir(parents=True)
        pool.write_text(json.dumps({"sample_id": sample_id, "scenario": "image_product_search"}) + "\n", encoding="utf-8")
        for scenario in ("after_sales", "itinerary_planning"):
            (pool.parent / f"{scenario}.jsonl").write_text("", encoding="utf-8")
        config = {
            "paths": {"output_dir": "outputs/week5"},
            "quality": {"core_scenarios": ["after_sales", "itinerary_planning"], "core_audit_rate": 0.10, "general_audit_rate": 0.05},
        }
        return root, config, sample_id

    def test_config_uses_current_schemas_and_qwen37_prompts(self) -> None:
        config = load_week5_config(ROOT, "configs/week5_dataset.json")
        self.assertEqual(config["prompt_versions"]["image_product_search"], "fewshot_4_v2")
        self.assertEqual(config["prompt_versions"]["after_sales"], "fewshot_4_v2")
        self.assertEqual(config["prompt_versions"]["itinerary_planning"], "standardized_v4")
        self.assertTrue(config["schemas"]["itinerary_planning"].endswith("itinerary_planning_v2.schema.json"))

    def test_isolation_rejects_source_hash_group_and_template(self) -> None:
        candidate = {
            "source_id": "source-a",
            "image_sha256": "a" * 64,
            "provenance": {"group_id": "group-a", "constraint_template_id": "template-a"},
        }
        empty = {name: set() for name in ("source_id", "image_sha256", "group_id", "constraint_template_id")}
        self.assertTrue(_check_candidate_isolation(candidate, empty, set()))
        for name, value in (("source_id", "source-a"), ("image_sha256", "a" * 64), ("group_id", "group-a"), ("constraint_template_id", "template-a")):
            exclusions = {key: set(values) for key, values in empty.items()}
            exclusions[name].add(value)
            self.assertFalse(_check_candidate_isolation(candidate, exclusions, set()))

    def test_jsonl_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            self.assertEqual(write_jsonl_new(path, [{"id": 1}]), 1)
            with self.assertRaises(Week5DataError):
                write_jsonl_new(path, [{"id": 2}])

    def test_human_output_must_match_current_schema(self) -> None:
        valid = {
            "business_category": "hotel", "style_tags": ["modern"],
            "visible_facilities": ["pool"], "price_range": "unknown",
            "observed_evidence": ["可见泳池"], "inferred_attributes": [],
            "unknown_fields": ["price_range"], "confidence": None,
        }
        validate_human_annotation(ROOT, "image_product_search", valid)
        invalid = dict(valid)
        invalid.pop("confidence")
        with self.assertRaises(Week5DataError):
            validate_human_annotation(ROOT, "image_product_search", invalid)

    def test_dialogue_requires_turns_and_valid_image_references(self) -> None:
        dialogue = {
            "dialogue_id": "d-1",
            "scenario": "image_search_consultation",
            "images": [{"image_id": "img_1", "path": "data/a.jpg", "sha256": "a" * 64}],
            "messages": [
                {"role": "user", "content": "看看这张图", "image_refs": ["img_1"]},
                {"role": "assistant", "content": "请问更关注风格还是设施？", "image_refs": ["img_1"]},
                {"role": "user", "content": "更关注设施", "image_refs": []},
                {"role": "assistant", "content": "可按可见设施继续筛选。", "image_refs": ["img_1"]},
                {"role": "user", "content": "那上一张适合亲子吗？", "image_refs": ["img_1"]},
                {"role": "assistant", "content": "仅凭图片不能确认亲子服务，需要查看商家信息。", "image_refs": ["img_1"]},
            ],
        }
        validate_dialogue(dialogue)
        dialogue["messages"][2]["image_refs"] = ["missing"]
        with self.assertRaises(Week5DataError):
            validate_dialogue(dialogue)

    def test_audit_selection_is_deterministic_and_rate_bounded(self) -> None:
        config = load_week5_config(ROOT, "configs/week5_dataset.json")
        first = qc_audit_selected("sample-1", "after_sales", config)
        self.assertEqual(first, qc_audit_selected("sample-1", "after_sales", config))
        core = sum(qc_audit_selected(f"sample-{index}", "after_sales", config) for index in range(10000))
        general = sum(qc_audit_selected(f"sample-{index}", "image_product_search", config) for index in range(10000))
        self.assertGreaterEqual(core, 900)
        self.assertGreaterEqual(general, 400)
        self.assertGreater(core, general)

    def test_human_correction_requires_real_preannotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, sample_id = self._workflow_fixture(directory)
            submission = root / "human.jsonl"
            submission.write_text(json.dumps({
                "sample_id": sample_id, "annotator": "annotator-a", "corrected_at": "2026-08-02T00:00:00Z",
                "human_annotation": {
                    "business_category": "unknown", "style_tags": [], "visible_facilities": [],
                    "price_range": "unknown", "observed_evidence": [], "inferred_attributes": [],
                    "unknown_fields": ["business_category", "price_range"], "confidence": None,
                },
            }) + "\n", encoding="utf-8")
            with self.assertRaises(Week5DataError):
                apply_human_corrections(root, config, "image_product_search", submission)
            pre = root / "outputs/week5/preannotations/image_product_search.jsonl"
            pre.parent.mkdir(parents=True)
            pre.write_text(json.dumps({"sample_id": sample_id, "status": "completed", "schema_valid": True}) + "\n", encoding="utf-8")
            self.assertEqual(apply_human_corrections(root, config, "image_product_search", submission)["applied"], 1)

    def test_cross_review_requires_self_review_and_distinct_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, config, sample_id = self._workflow_fixture(directory)
            annotations = root / "outputs/week5/annotations/image_product_search.jsonl"
            annotations.parent.mkdir(parents=True)
            annotations.write_text(json.dumps({"sample_id": sample_id, "scenario": "image_product_search", "annotator": "a", "revision": 1}) + "\n", encoding="utf-8")
            quality_input = root / "quality.jsonl"
            quality_input.write_text(json.dumps({"sample_id": sample_id, "stage": "cross_review", "decision": "pass", "reviewer": "b", "issues": []}) + "\n", encoding="utf-8")
            with self.assertRaises(Week5DataError):
                apply_quality_records(root, config, "image_product_search", quality_input)
            quality_input.write_text("\n".join([
                json.dumps({"sample_id": sample_id, "stage": "self_review", "decision": "pass", "reviewer": "a", "issues": []}),
                json.dumps({"sample_id": sample_id, "stage": "cross_review", "decision": "pass", "reviewer": "b", "issues": []}),
            ]) + "\n", encoding="utf-8")
            self.assertEqual(apply_quality_records(root, config, "image_product_search", quality_input)["applied"], 2)


if __name__ == "__main__":
    unittest.main()
