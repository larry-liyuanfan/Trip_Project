import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.evaluation.visual_teacher_retry import collect_with_history

ROOT = Path(__file__).resolve().parents[1]


class TeacherRetryTests(unittest.TestCase):
    def test_exhausted_corrections_never_create_a_target(self):
        config = json.loads((ROOT / "configs/week8/visual_teacher_v4.json").read_text(encoding="utf-8"))
        observation = json.loads((ROOT / config["observation_config"]).read_text(encoding="utf-8"))
        invalid = {"subject_kind": "food_closeup", "subject_fact": "Bowl of soup", "style_evidence": [],
                   "facility_evidence": [{"label": "parking", "fact": "Parking lot"}], "price_text": []}
        response = Mock(status_code=200)
        response.json.return_value = {"model": config["model"], "choices": [{"message": {"content": json.dumps(invalid)}}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image.jpg").write_bytes(b"image")
            with patch("src.evaluation.visual_teacher_retry.requests.post", return_value=response) as request:
                result = collect_with_history({"sample_id": "dev", "image_path": "image.jpg"}, config, observation, root, "https://example.invalid", "test-key")
        self.assertEqual(request.call_count, 4)
        self.assertIsNone(result["target"])
        self.assertTrue(result["error"])
        self.assertTrue(all(attempt["error"] for attempt in result["attempts"]))

    def test_retry_budget_must_be_explicit_and_bounded(self):
        with self.assertRaisesRegex(ValueError, "explicitly versioned and bounded"):
            collect_with_history({}, {"retry_protocol": "bounded_history_correction_v1", "max_attempts": 100}, {}, ROOT, "https://example.invalid", "test-key")

    def test_failed_teacher_output_is_context_not_an_accepted_label(self):
        config = json.loads((ROOT / "configs/week8/visual_teacher_v4.json").read_text(encoding="utf-8"))
        observation = json.loads((ROOT / config["observation_config"]).read_text(encoding="utf-8"))
        valid = {"subject_kind": "food_closeup", "subject_fact": "Bowl of soup", "style_evidence": [], "facility_evidence": [], "price_text": []}
        invalid = {**valid, "facility_evidence": [{"label": "parking", "fact": "Parking is not visible"}]}
        requests_seen = []
        def respond(*args, **kwargs):
            requests_seen.append(copy.deepcopy(kwargs["json"]))
            value = invalid if len(requests_seen) < 4 else valid
            response = Mock(status_code=200)
            response.json.return_value = {"model": config["model"], "choices": [{"message": {"content": json.dumps(value)}}]}
            return response
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "image.jpg").write_bytes(b"image")
            schema = Path("configs/evaluation/schemas/image_product_search_v1.schema.json")
            (root / schema).parent.mkdir(parents=True)
            (root / schema).write_bytes((ROOT / schema).read_bytes())
            with patch("src.evaluation.visual_teacher_retry.requests.post", side_effect=respond):
                result = collect_with_history({"sample_id": "dev", "image_path": "image.jpg"}, config, observation, root, "https://example.invalid", "test-key")
        self.assertEqual(len(result["attempts"]), 4)
        self.assertIsNone(result["error"])
        self.assertEqual(result["target"]["visible_facilities"], [])
        self.assertEqual(requests_seen[-1]["messages"][2]["role"], "assistant")
        self.assertIn("Parking is not visible", requests_seen[-1]["messages"][2]["content"])
        self.assertEqual(result["label_source"], "model_generated_silver")
        self.assertFalse(result["visual_accuracy_claim_supported"])
        self.assertNotIn("test-key", json.dumps(result))
