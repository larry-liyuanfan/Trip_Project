from pathlib import Path
import tempfile
import unittest

from scripts.verify_week8_candidate_acceptance import validate_test_log, validate_retrieval


class CandidateAcceptanceTests(unittest.TestCase):
    def test_a_failed_or_empty_suite_cannot_be_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.log"
            path.write_text("Ran 740 tests in 20.001s\nOK\n", encoding="utf-8")
            self.assertEqual(validate_test_log(path)["count"], 740)
            for value in ("Ran 0 tests in 0.01s\nOK\n", "Ran 740 tests in 20s\nFAILED (failures=1)\n", "OK\n"):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_test_log(path)

    def test_retrieval_requires_actual_execution_and_exact_release(self):
        value = {"status": "PASS", "production_route_executed": True, "reference_query_metadata_used": False,
                 "query_change_changes_results": True, "queries": [{"filter_correct": True}],
                 "dialogue_routing": [{"passed": True, "expected_status": "COMPLETED", "response": {
                     "release_id": "new", "task_status": "COMPLETED", "tool_calls": [{"function": "visual_search"}]}}]}
        validate_retrieval(value, "new")
        with self.assertRaises(ValueError):
            validate_retrieval(value, "old")
        value["dialogue_routing"][0]["response"]["tool_calls"] = []
        with self.assertRaises(ValueError):
            validate_retrieval(value, "new")
