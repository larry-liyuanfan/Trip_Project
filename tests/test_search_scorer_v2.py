import copy
import math
import random
import unittest

from src.evaluation.relevance_evidence import canonical_json_sha256 as sha
from src.evaluation.relevance_evidence import score_search_results, validate_annotation_protocol
from src.evaluation.search_scorer_v2 import build_weak_qrels, ndcg


def fixture(grades=(3, 2, 1, 0), human=False, slices=None):
    source = {"source_id": "unit-test-source"}
    query = {
        "query_id": "q", "source": source, "source_record_sha256": sha(source),
        "image": {"sha256": "a" * 64}, "requested_filters": {"city": "x"},
        "unsupported_constraints": [], "slices": slices or ["image_similar"],
    }
    query["query_sha256"] = sha(query)
    annotation = {
        "query_id": "q", "label_provenance": "synthetic",
        "annotators": ["programmatic_fixture"], "conflict_resolution": "not_applicable_single_programmatic",
        "grade_rules": {"target_business_category": "hotel", "grades": {str(i): str(i) for i in range(4)}},
    }
    corpus = [{"image_id": f"d{i}", "city": "x", "business_category": "hotel"} for i in range(len(grades))]
    qrels = build_weak_qrels([query], [annotation], corpus or [{"image_id": "placeholder"}])
    qrels["documents"] = corpus
    qrels["queries"][0]["judgments"] = [{"image_id": f"d{i}", "grade": g} for i, g in enumerate(grades)]
    if human:
        annotation.update(label_provenance="human", annotators=["alice", "bob"], conflict_resolution="none")
        annotation.pop("grade_rules")
        qrels["document_universe_kind"] = "judged_pool"
        qrels["queries"][0]["label_provenance"] = "human"
        for doc in corpus:
            doc.update(image_sha256="b" * 64, source=source, source_record_sha256=sha(source))
        for judgment in qrels["queries"][0]["judgments"]:
            judgment.update(conflict_resolution="none", ratings=[
                {"annotator_id": name, "grade": judgment["grade"]} for name in ("alice", "bob")
            ])
    qrels["corpus_sha256"] = sha(corpus)
    qrels["annotation_sha256"] = sha([annotation])
    results = [{"query_id": "q", "methods": {"m": {"hits": [{"image_id": "d0"}], "no_result": False}}}]
    return [query], [annotation], results, qrels


def run_fixture(data, **kwargs):
    queries, annotations, results, qrels = data
    return score_search_results(queries, annotations, results, methods=("m",), qrels=qrels, **kwargs)


class FullQrelsTests(unittest.TestCase):
    def test_missing_ideal_document_counterexample(self):
        data = fixture((3, 1))
        data[2][0]["methods"]["m"]["hits"] = [{"image_id": "d1"}]
        metrics = run_fixture(data, ks=(1, 10))["methods"]["m"]
        self.assertAlmostEqual(metrics["ndcg_at_1"], 1 / 7)
        self.assertEqual(metrics["recall_at_1"], 0)
        self.assertEqual(metrics["per_query"][0]["idcg_at_1"], 7)

    def test_full_recall_denominator_ignores_claim_in_predictions(self):
        data = fixture((3, 3, 2))
        data[2][0]["methods"]["m"]["relevant_total"] = 1
        metrics = run_fixture(data)["methods"]["m"]
        self.assertAlmostEqual(metrics["recall_at_10"], 1 / 3)
        self.assertLess(metrics["ndcg_at_10"], 1)
        self.assertEqual(metrics["per_query"][0]["relevant_total"], 3)

    def test_perfect_ranking_and_short_corpus(self):
        data = fixture((1, 3, 2))
        data[2][0]["methods"]["m"]["hits"] = [{"image_id": f"d{i}"} for i in (1, 2, 0)]
        metrics = run_fixture(data)["methods"]["m"]
        self.assertEqual(metrics["recall_at_10"], 1)
        self.assertEqual(metrics["ndcg_at_10"], 1)

    def test_binary_threshold_is_separate_from_graded_gain(self):
        metrics = run_fixture(fixture((1,)), ks=(1,), binary_threshold=2)["methods"]["m"]
        self.assertIsNone(metrics["recall_at_1"])
        self.assertIsNone(metrics["mrr_at_1"])
        self.assertEqual(metrics["ndcg_at_1"], 1)
        self.assertEqual(metrics["denominators"]["recall_at_1"]["denominator"], 0)
        self.assertEqual(run_fixture(fixture((1,)), ks=(1,), binary_threshold=1)["methods"]["m"]["recall_at_1"], 1)

    def test_empty_prediction_not_dataset_slice_is_no_result(self):
        data = fixture((3,), slices=["no_result"])
        metrics = run_fixture(data)["methods"]["m"]
        self.assertEqual(metrics["no_result_rate"], 0)
        self.assertEqual(metrics["dataset_no_result_slice_rate"], 1)
        self.assertEqual(metrics["no_relevant_judgments_rate"], 0)
        data[2][0]["methods"]["m"] = {"hits": [], "no_result": True}
        metrics = run_fixture(data)["methods"]["m"]
        self.assertEqual(metrics["no_result_rate"], 1)
        self.assertEqual(metrics["no_result_accuracy"], 0)
        self.assertEqual(metrics["dataset_no_result_accuracy"], 1)
        self.assertEqual(metrics["ndcg_at_10"], 0)

    def test_no_positive_or_no_judged_documents(self):
        for grades in ((0, 0), ()):
            data = fixture(grades)
            data[2][0]["methods"]["m"] = {"hits": []}
            metrics = run_fixture(data)["methods"]["m"]
            self.assertIsNone(metrics["ndcg_at_10"])
            self.assertIsNone(metrics["recall_at_10"])
            self.assertEqual(metrics["no_relevant_judgments_rate"], 1)
            self.assertEqual(metrics["no_result_accuracy"], 1)

    def test_failed_method_counted_zero_not_removed_from_ranking(self):
        for value in (None, {"failed": True, "hits": [{"image_id": "d0"}]}):
            data = fixture()
            data[2][0]["methods"]["m"] = value
            metrics = run_fixture(data)["methods"]["m"]
            self.assertEqual(metrics["failure_rate"], 1)
            self.assertEqual(metrics["ranking_support"], 1)
            self.assertEqual(metrics["recall_at_10"], 0)
            self.assertEqual(metrics["no_result_rate"], 0)

    def test_unjudged_keeps_rank_with_zero_gain(self):
        data = fixture((3,))
        data[2][0]["methods"]["m"]["hits"] = [{"image_id": "unjudged"}, {"image_id": "d0"}]
        metrics = run_fixture(data)["methods"]["m"]
        self.assertEqual(metrics["unjudged_returned_count"], 1)
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        self.assertAlmostEqual(metrics["ndcg_at_10"], 1 / math.log2(3))
        self.assertEqual(metrics["filter_correctness"], 0)

    def test_human_grades_override_conflicting_hit_metadata(self):
        data = fixture((0, 3), human=True)
        data[2][0]["methods"]["m"]["hits"] = [{"image_id": "d0", "business_category": "hotel", "city": "x"}]
        report = run_fixture(data)
        self.assertEqual(report["evidence_class"], "human_declared_qrels_identity_unverified")
        self.assertFalse(report["qrels_validation"]["promotion_eligible_as_human_ground_truth"])
        self.assertTrue(report["qrels_validation"]["human_protocol_complete"])
        self.assertEqual(report["methods"]["m"]["ndcg_at_10"], 0)

    def test_filters_use_catalog_not_forged_returned_metadata(self):
        data = fixture((3,))
        data[3]["documents"][0]["city"] = "y"
        data[3]["corpus_sha256"] = sha(data[3]["documents"])
        data[2][0]["methods"]["m"]["hits"][0]["city"] = "x"
        self.assertEqual(run_fixture(data)["methods"]["m"]["filter_correctness"], 0)

    def test_random_rankings_match_independent_reference_and_are_bounded(self):
        rng = random.Random(71)
        for _ in range(100):
            gold = [rng.randrange(4) for _ in range(20)]
            returned = rng.sample(gold, 8)
            ideal = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(gold, reverse=True)[:5]))
            actual = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(returned[:5]))
            value = ndcg(returned, gold, 5)
            self.assertAlmostEqual(value, actual / ideal)
            self.assertLessEqual(value, 1)


class FailClosedTests(unittest.TestCase):
    def test_hit_only_api_is_rejected(self):
        q, a, r, _ = fixture()
        with self.assertRaisesRegex(ValueError, "full qrels"):
            score_search_results(q, a, r)

    def test_duplicate_identities_rejected(self):
        for target in ("queries", "results", "documents", "qrels", "judgments", "hits"):
            data = fixture()
            rows = {"queries": data[0], "results": data[2], "documents": data[3]["documents"],
                    "qrels": data[3]["queries"], "judgments": data[3]["queries"][0]["judgments"],
                    "hits": data[2][0]["methods"]["m"]["hits"]}[target]
            rows.append(copy.deepcopy(rows[0]))
            with self.subTest(target=target), self.assertRaises(ValueError):
                run_fixture(data)

    def test_missing_extra_query_and_missing_document_rejected(self):
        for target in ("results", "qrels", "judgments"):
            data = fixture()
            rows = {"results": data[2], "qrels": data[3]["queries"], "judgments": data[3]["queries"][0]["judgments"]}[target]
            rows.pop()
            with self.subTest(target=target), self.assertRaises(ValueError):
                run_fixture(data)

    def test_bad_grade_and_bad_k_rejected(self):
        for grade in (-1, 4, 1.5, True, None, "3"):
            data = fixture()
            data[3]["queries"][0]["judgments"][0]["grade"] = grade
            with self.assertRaisesRegex(ValueError, "grade"):
                run_fixture(data)
        for ks in ((), (0,), (True,), (1, 1)):
            with self.assertRaisesRegex(ValueError, "ks"):
                run_fixture(fixture(), ks=ks)

    def test_sha_bindings_rejected_on_mutation(self):
        for key in ("query_sha256", "query_image_sha256", "query_source_sha256"):
            data = fixture()
            data[3]["queries"][0][key] = "c" * 64
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                run_fixture(data)

    def test_false_abstention_flag_and_malformed_hits_rejected(self):
        for row in ({"hits": [{"image_id": "d0"}], "no_result": True}, {"hits": None}):
            data = fixture()
            data[2][0]["methods"]["m"] = row
            with self.assertRaises(ValueError):
                run_fixture(data)

    def test_human_header_alone_and_duplicate_annotator_rejected(self):
        data = fixture(human=True)
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_annotation_protocol(data[0], data[1])
        data[1][0]["annotators"] = ["alice", "alice"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            run_fixture(data)

    def test_human_unresolved_conflict_is_rejected(self):
        data = fixture(human=True)
        data[3]["queries"][0]["judgments"][0]["ratings"][0]["grade"] = 0
        with self.assertRaisesRegex(ValueError, "unresolved"):
            run_fixture(data)

    def test_human_adjudication_and_independence(self):
        data = fixture(human=True)
        judgment = data[3]["queries"][0]["judgments"][0]
        judgment["ratings"][0]["grade"] = 0
        judgment.update(conflict_resolution="adjudicated", adjudicator_id="carol", adjudication_reason="visual evidence reviewed")
        self.assertEqual(run_fixture(data)["evidence_class"], "human_declared_qrels_identity_unverified")
        judgment["adjudicator_id"] = "alice"
        with self.assertRaisesRegex(ValueError, "independent"):
            run_fixture(data)

    def test_human_document_provenance_and_rating_coverage(self):
        data = fixture(human=True)
        data[3]["queries"][0]["judgments"][0]["ratings"].pop()
        with self.assertRaisesRegex(ValueError, "two distinct"):
            run_fixture(data)
        data = fixture(human=True)
        data[3]["documents"][0].pop("source")
        data[3]["corpus_sha256"] = sha(data[3]["documents"])
        with self.assertRaisesRegex(ValueError, "source record"):
            run_fixture(data)

    def test_weak_corpus_cannot_create_human_grades(self):
        q, a, r, gold = fixture(human=True)
        with self.assertRaisesRegex(ValueError, "explicit"):
            score_search_results(q, a, r, corpus=gold["documents"])


if __name__ == "__main__":
    unittest.main()
