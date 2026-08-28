import copy
import json
from pathlib import Path
import unittest
from src.inference.product_observation import map_observation, generate_observation
from src.inference.product_observation import canonical_config_sha256, load_observation_config
from src.inference.system_runtime import GenerationResult, ReleaseSettings, ScenarioService, TransformersPeftBackend, RuntimeConfigurationError
from src.inference.schemas import TaskRequest, DialogueRequest
from dataclasses import replace
from contextlib import contextmanager
from unittest.mock import Mock, patch
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/week8/product_observation_v1.json").read_text(encoding="utf-8"))
        self.value = {"subject_kind": "dining_space", "subject_fact": "Tables and chairs in a dining room",
                      "style_evidence": [{"label": "modern", "fact": "Geometric furniture with plain walls"}],
                      "facility_evidence": [{"label": "seating", "fact": "Chairs at tables"}], "price_text": []}

    def test_mapping_preserves_positive_labels_and_unknown_price(self):
        target = map_observation(self.value, self.config)
        self.assertEqual(target["business_category"], "restaurant")
        self.assertEqual(target["style_tags"], ["modern"])
        self.assertEqual(target["visible_facilities"], ["seating"])
        self.assertEqual(target["unknown_fields"], ["price_range"])
        self.assertEqual(target["inferred_attributes"], [])

    def test_food_closeup_does_not_guess_venue_or_discard_positive_labels_silently(self):
        self.value["subject_kind"] = "food_closeup"
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)
        self.value.update(style_evidence=[], facility_evidence=[])
        result = map_observation(self.value, self.config)
        self.assertEqual(result["business_category"], "unknown")
        self.assertEqual(len(result["unknown_fields"]), 4)

    def test_negated_facility_fact_is_rejected(self):
        self.value["facility_evidence"] = [{"label": "parking", "fact": "Parking is not visible"}]
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)

    def test_duplicate_labels_are_not_silently_repaired(self):
        self.value["style_evidence"].append(copy.deepcopy(self.value["style_evidence"][0]))
        with self.assertRaises(ValueError):
            map_observation(self.value, self.config)

    def test_price_numbers_do_not_invent_price_tiers(self):
        self.value["price_text"] = ["$100"]
        self.assertEqual(map_observation(self.value, self.config)["price_range"], "unknown")

    def test_single_retry_preserves_invalid_raw_output(self):
        class Backend:
            def __init__(self, value):
                self.values = iter(["not-json", json.dumps(value)])
            def generate_with_usage(self, *args, **kwargs):
                return GenerationResult(content=next(self.values), input_tokens=10, output_tokens=20)
        result = generate_observation(Backend(self.value), "image.jpg", self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["attempts"][0].raw_output, "not-json")
        self.assertIsNotNone(result["attempts"][0].error)
        self.assertIsNone(result["attempts"][1].error)

    def test_release_binds_observation_hash_and_scenario_adapter_policy(self):
        settings = ReleaseSettings.load(ROOT, ROOT / "configs/releases/qwen3_vl_week8_observation_probe_v1.json")
        self.assertEqual(settings.product_observation["protocol"], "product_visual_observation_v2")
        self.assertIn("image_product_search", settings.adapter_disabled_scenarios)
        self.assertNotIn("after_sales", settings.adapter_disabled_scenarios)
        with self.assertRaises(ValueError):
            load_observation_config(ROOT / "configs/week8/product_observation_v1.json", "0" * 64)

    def test_product_dialogue_uses_actual_observation_and_reports_base(self):
        class Backend:
            def __init__(self, value):
                self.value = value
                self.modes = []
            @contextmanager
            def adapter_mode(self, enabled):
                self.modes.append(enabled)
                yield
            def generate_with_usage(self, *args, **kwargs):
                return GenerationResult(content=json.dumps(self.value), input_tokens=10, output_tokens=20)
        settings = ReleaseSettings.load(ROOT, ROOT / "configs/releases/qwen3_vl_week8_observation_probe_v1.json")
        backend = Backend(self.value)
        result = ScenarioService(settings, backend).run_dialogue(DialogueRequest(messages=[{"role": "user", "content": "识别这张图片"}], image_urls=["image.jpg"]))
        self.assertEqual(result.task_status, "COMPLETED")
        self.assertEqual(result.task_result["result"], map_observation(self.value, self.config))
        self.assertEqual(result.adapter, "none")
        self.assertEqual(backend.modes, [False])
        self.assertEqual(len(result.attempts), 1)

    def test_adapter_mode_restores_state_after_exception(self):
        class Model:
            disabled = False
            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False
        model = Model()
        backend = TransformersPeftBackend.from_loaded(model, Mock(), Mock())
        with self.assertRaisesRegex(ValueError, "generation failed"):
            with backend.adapter_mode(False):
                self.assertTrue(model.disabled)
                raise ValueError("generation failed")
        self.assertFalse(model.disabled)
        with backend.adapter_mode(True):
            self.assertFalse(model.disabled)

    def test_backend_without_adapter_scope_cannot_silently_use_wrong_weights(self):
        settings = ReleaseSettings.load(ROOT, ROOT / "configs/releases/qwen3_vl_week8_observation_probe_v1.json")
        with self.assertRaises(RuntimeConfigurationError):
            ScenarioService(settings, object()).run_task("image_product_search", TaskRequest(image_urls=["image.jpg"]))

    def test_canonical_config_hash_is_not_changed_by_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_bytes(json.dumps(self.config, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(load_observation_config(path, canonical_config_sha256(self.config)), self.config)


if __name__ == "__main__":
    unittest.main()
