import unittest

from src.evaluation.readiness import ReadinessValidationError, parse_models_payload


class EvaluationReadinessTest(unittest.TestCase):
    def test_models_payload_requires_nonempty_model_ids(self):
        self.assertEqual(
            parse_models_payload({"object": "list", "data": [{"id": "model-a"}]}),
            ["model-a"],
        )
        with self.assertRaises(ReadinessValidationError):
            parse_models_payload({"object": "list", "data": []})
        with self.assertRaises(ReadinessValidationError):
            parse_models_payload({"data": [{"id": ""}]})


if __name__ == "__main__":
    unittest.main()
