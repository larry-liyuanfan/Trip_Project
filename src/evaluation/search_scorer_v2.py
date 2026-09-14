"""Full-qrels retrieval scoring; no inference and no metadata-as-human fallback."""

from __future__ import annotations

import math
import statistics
from typing import Any

from src.evaluation.relevance_evidence import (
    _norm, _required_sha, _required_text, _weak_grade, canonical_json_sha256,
)

SCORER_VERSION = "search_scorer_v2"
WEAK_PROVENANCE = {"weak_programmatic_metadata", "synthetic"}


def require_versioned_inference_protocol(config: dict) -> None:
    """Do not silently apply historic selection gates after changing the scorer."""
    if config.get("search", {}).get("scorer_version") != SCORER_VERSION:
        raise ValueError("historical inference protocol is frozen; offline rescore only, or preregister a new scorer-v2 protocol")


def unique_index(rows: list[dict[str, Any]], key: str, scope: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"{scope} must be a list")
    index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{scope} rows must be objects")
        identity = _required_text(row, key, scope)
        if identity in index:
            raise ValueError(f"duplicate {scope} {key}: {identity}")
        index[identity] = row
    return index


def validate_query_identities(queries: list[dict]) -> dict:
    index = unique_index(queries, "query_id", "queries")
    if not index:
        raise ValueError("query universe must not be empty")
    for query in queries:
        unsigned = {key: value for key, value in query.items() if key != "query_sha256"}
        if query.get("query_sha256") != canonical_json_sha256(unsigned):
            raise ValueError("query_sha256 mismatch")
        if not isinstance(query.get("source"), dict):
            raise ValueError("query source required")
        if query.get("source_record_sha256") != canonical_json_sha256(query["source"]):
            raise ValueError("query source_record_sha256 mismatch")
        _required_sha(query.get("image", {}), "sha256", "query image")
    return index


def build_weak_qrels(queries: list[dict], annotations: list[dict], corpus: list[dict]) -> dict:
    """Materialize rules over the FULL corpus, never over the retrieved pool."""
    from src.evaluation.relevance_evidence import validate_annotation_protocol

    validate_annotation_protocol(queries, annotations)
    validate_query_identities(queries)
    documents = unique_index(corpus, "image_id", "corpus")
    if not documents:
        raise ValueError("full corpus must not be empty")
    annotation_index = unique_index(annotations, "query_id", "annotations")
    rows = []
    for query in queries:
        annotation = annotation_index[query["query_id"]]
        if annotation["label_provenance"] not in WEAK_PROVENANCE:
            raise ValueError("human scoring requires explicit qrels; metadata fallback forbidden")
        rows.append({
            "query_id": query["query_id"],
            "query_sha256": query["query_sha256"],
            "query_image_sha256": query["image"]["sha256"],
            "query_source_sha256": query["source_record_sha256"],
            "label_provenance": annotation["label_provenance"],
            "judgments": [
                {"image_id": identity, "grade": _weak_grade(doc, annotation)}
                for identity, doc in documents.items()
            ],
        })
    return {
        "schema_version": "search_qrels_v2",
        "coverage": "complete_declared_document_universe",
        "document_universe_kind": "full_index_metadata_weak",
        "documents": corpus,
        "queries": rows,
        "annotation_sha256": canonical_json_sha256(annotations),
        "corpus_sha256": canonical_json_sha256(corpus),
        "image_byte_verification": "NOT_PERFORMED_BY_OFFLINE_SCORER",
    }


def _grade(value: Any) -> int:
    if type(value) is not int or value not in range(4):
        raise ValueError("grade must be an integer in [0, 3]")
    return value


def _validate_human_judgment(judgment: dict, declared_annotators: set[str]) -> None:
    ratings = unique_index(judgment.get("ratings"), "annotator_id", "human ratings")
    if len(ratings) < 2 or set(ratings) != declared_annotators:
        raise ValueError("each human query-document pair requires two distinct declared annotators")
    if any(name.startswith("programmatic_") for name in ratings):
        raise ValueError("programmatic annotator cannot claim human")
    grades = {_grade(row.get("grade")) for row in ratings.values()}
    resolution = judgment.get("conflict_resolution")
    if resolution == "none":
        if grades != {judgment["grade"]}:
            raise ValueError("unresolved human conflict or final grade differs from agreement")
    elif resolution == "adjudicated":
        adjudicator = _required_text(judgment, "adjudicator_id")
        if adjudicator in ratings or adjudicator.startswith("programmatic_"):
            raise ValueError("adjudicator must be an independent human identity")
        _required_text(judgment, "adjudication_reason")
    else:
        raise ValueError("human judgment requires agreement or adjudication")


def validate_qrels(queries: list[dict], annotations: list[dict], qrels: dict) -> dict:
    """A complete judgment matrix is relative to its declared document universe."""
    query_index = validate_query_identities(queries)
    annotation_index = unique_index(annotations, "query_id", "annotations")
    if set(annotation_index) != set(query_index):
        raise ValueError("annotation/query support mismatch")
    if not isinstance(qrels, dict) or qrels.get("schema_version") != "search_qrels_v2":
        raise ValueError("explicit search_qrels_v2 is required")
    if qrels.get("coverage") != "complete_declared_document_universe":
        raise ValueError("qrels must declare complete document coverage")
    if qrels.get("document_universe_kind") not in {"full_index_metadata_weak", "judged_pool", "full_index"}:
        raise ValueError("unknown document universe kind")
    if qrels.get("annotation_sha256") != canonical_json_sha256(annotations):
        raise ValueError("qrels annotation_sha256 mismatch")
    documents = unique_index(qrels.get("documents"), "image_id", "documents")
    if qrels.get("corpus_sha256") != canonical_json_sha256(qrels["documents"]):
        raise ValueError("qrels corpus_sha256 mismatch")
    judgments = unique_index(qrels.get("queries"), "query_id", "qrels queries")
    if set(judgments) != set(query_index):
        raise ValueError("qrels/query coverage mismatch")
    human_queries = 0
    for identity, query in query_index.items():
        row = judgments[identity]
        expected = (query["query_sha256"], query["image"]["sha256"], query["source_record_sha256"])
        actual = (row.get("query_sha256"), row.get("query_image_sha256"), row.get("query_source_sha256"))
        if actual != expected:
            raise ValueError("qrels query/image/source SHA mismatch")
        provenance = row.get("label_provenance")
        if provenance != annotation_index[identity].get("label_provenance"):
            raise ValueError("qrels label provenance mismatch")
        if provenance not in WEAK_PROVENANCE | {"human"}:
            raise ValueError("unsupported qrels provenance")
        pairs = unique_index(row.get("judgments"), "image_id", "judgments")
        if set(pairs) != set(documents):
            raise ValueError("incomplete qrels document coverage")
        if provenance == "human":
            human_queries += 1
            if not documents:
                raise ValueError("empty human judgment universe cannot establish ground truth")
            if qrels["document_universe_kind"] == "full_index_metadata_weak":
                raise ValueError("metadata-only universe cannot claim human")
            for doc in documents.values():
                _required_sha(doc, "image_sha256", "human document")
                if not isinstance(doc.get("source"), dict) or not doc["source"]:
                    raise ValueError("human document source record required")
                if doc.get("source_record_sha256") != canonical_json_sha256(doc["source"]):
                    raise ValueError("human document source SHA mismatch")
        for judgment in pairs.values():
            _grade(judgment.get("grade"))
            if provenance == "human":
                _validate_human_judgment(judgment, set(annotation_index[identity]["annotators"]))
    return {
        "status": "PASS", "query_support": len(query_index),
        "document_support": len(documents), "judgment_support": len(query_index) * len(documents),
        "declared_human_query_support": human_queries,
        "human_protocol_complete": human_queries == len(query_index),
        "promotion_eligible_as_human_ground_truth": False,
        "document_universe_kind": qrels["document_universe_kind"],
        "qrels_sha256": canonical_json_sha256(qrels),
        "identity_boundary": "SHA binding checked; human identity authenticity and image bytes require external audit",
    }


def dcg(grades: list[int], k: int) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))


def ndcg(grades: list[int], full_qrels_grades: list[int], k: int) -> float | None:
    ideal = dcg(sorted(full_qrels_grades, reverse=True), k)
    return dcg(grades, k) / ideal if ideal else None


def score(queries: list[dict], annotations: list[dict], results: list[dict], methods: tuple[str, ...],
          qrels: dict, ks: tuple[int, ...] = (5, 10), binary_threshold: int = 2) -> dict:
    from src.evaluation.relevance_evidence import validate_annotation_protocol

    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks must contain unique positive integers")
    if type(binary_threshold) is not int or binary_threshold not in (1, 2, 3):
        raise ValueError("binary threshold must be 1, 2 or 3")
    if not methods or len(set(methods)) != len(methods) or any(not isinstance(m, str) or not m for m in methods):
        raise ValueError("methods must be nonempty and unique")
    validation = validate_annotation_protocol(queries, annotations, qrels=qrels)
    result_index = unique_index(results, "query_id", "results")
    if set(result_index) != {query["query_id"] for query in queries}:
        raise ValueError("search result support must exactly match the query lock")
    if any(not isinstance(row.get("methods"), dict) for row in results):
        raise ValueError("result methods must be an object; empty object denotes all methods failed")
    gold = {row["query_id"]: {j["image_id"]: j["grade"] for j in row["judgments"]} for row in qrels["queries"]}
    documents = {doc["image_id"]: doc for doc in qrels["documents"]}
    metrics = {}
    for method in methods:
        per_query = []
        for query in queries:
            identity = query["query_id"]
            row = result_index[identity].get("methods", {}).get(method)
            if isinstance(row, dict) and type(row.get("failed", False)) is not bool:
                raise ValueError("failed flag must be boolean")
            failed = not isinstance(row, dict) or row.get("failed", False) is True
            if not failed and not isinstance(row.get("hits"), list):
                raise ValueError("successful method result must contain a hits list")
            hits = [] if failed else row["hits"]
            hit_index = unique_index(hits, "image_id", "retrieved hits")
            empty_prediction = not failed and not hits
            flag = None if failed else row.get("no_result")
            if flag is not None and (type(flag) is not bool or flag != empty_prediction):
                raise ValueError("no_result flag contradicts actual empty hits")
            grades = [gold[identity].get(doc_id, 0) for doc_id in hit_index]
            full_grades = list(gold[identity].values())
            relevant_total = sum(g >= binary_threshold for g in full_grades)
            filters = {key: value for key, value in query.get("requested_filters", {}).items()
                       if key in {"city", "business_category", "price_range"}}
            # Filter checks use locked catalog values, not model-supplied hit metadata.
            filter_correct = not failed and all(
                doc_id in documents and all(_norm(documents[doc_id].get(key)) == _norm(value)
                                             for key, value in filters.items())
                for doc_id in hit_index
            )
            metric = {
                "query_id": identity, "slices": query.get("slices", []), "failed": failed,
                "ranking_evaluable": relevant_total > 0, "relevant_total": relevant_total,
                "judged_document_support": len(full_grades),
                "returned_count": len(hits), "unjudged_returned_count": sum(d not in gold[identity] for d in hit_index),
                "predicted_no_result": empty_prediction,
                "no_relevant_judgments": relevant_total == 0,
                "dataset_no_result": "no_result" in query.get("slices", []),
                "no_result_correct": not failed and empty_prediction == (relevant_total == 0),
                "dataset_no_result_correct": not failed and empty_prediction == ("no_result" in query.get("slices", [])),
                "filter_evaluable": bool(filters), "filter_correct": filter_correct,
                "unsupported_constraints_unapplied": not failed and set(query.get("unsupported_constraints", []))
                == set(row.get("unsupported_constraints_unapplied", [])),
            }
            for k in ks:
                retrieved_positive = sum(g >= binary_threshold for g in grades[:k])
                metric[f"relevant_retrieved_at_{k}"] = retrieved_positive
                metric[f"recall_at_{k}"] = retrieved_positive / relevant_total if relevant_total else None
                metric[f"mrr_at_{k}"] = next((1 / rank for rank, g in enumerate(grades[:k], 1)
                                              if g >= binary_threshold), 0.0) if relevant_total else None
                metric[f"dcg_at_{k}"] = dcg(grades, k)
                metric[f"idcg_at_{k}"] = dcg(sorted(full_grades, reverse=True), k)
                metric[f"ndcg_at_{k}"] = ndcg(grades, full_grades, k)
            per_query.append(metric)
        metrics[method] = aggregate(per_query, ks)
        metrics[method]["slices"] = {
            name: aggregate([row for row in per_query if name in row["slices"]], ks)
            for name in sorted({s for row in per_query for s in row["slices"]})
        }
        metrics[method]["per_query"] = per_query
    return {
        "schema_version": SCORER_VERSION, "scope": "business_semantic_relevance",
        "evidence_class": "human_declared_qrels_identity_unverified" if validation["human_protocol_complete"] else "weak_or_synthetic_not_human",
        "document_universe_kind": qrels["document_universe_kind"],
        "no_result_claim_scope": "declared_qrels_universe_only_not_unjudged_full_index",
        "query_support": len(queries), "methods": metrics, "qrels_validation": validation,
        "contract": {
            "ks": list(ks), "binary_threshold": binary_threshold, "graded_gain": "2**grade - 1",
            "idcg": "complete valid qrels sorted by grade, truncated at k",
            "unjudged": "zero_gain_and_nonrelevant_keep_rank_report_count",
            "duplicates": "reject_query_document_and_prediction_duplicates",
            "query_universe": "exact_manifest_support_missing_method_is_failed_zero_not_dropped",
            "no_positive": "Recall/MRR null and excluded; nDCG null only if IDCG=0; report denominators",
            "no_result": "successful empty hits; failures separately counted, not abstentions",
            "filter": "all returned docs satisfy supported filters using locked catalog; empty vacuously correct; macro over filtered queries only",
        },
        "query_manifest_sha256": canonical_json_sha256(queries),
        "annotation_sha256": canonical_json_sha256(annotations),
        "result_sha256": canonical_json_sha256(results), "qrels_sha256": canonical_json_sha256(qrels),
    }


def aggregate(rows: list[dict], ks: tuple[int, ...]) -> dict:
    def ratio(numerator, denominator):
        return {"numerator": numerator, "denominator": denominator,
                "value": numerator / denominator if denominator else None}

    result = {"support": len(rows), "ranking_support": sum(r["ranking_evaluable"] for r in rows)}
    denominators = {}
    for k in ks:
        for name in ("recall", "mrr", "ndcg"):
            key = f"{name}_at_{k}"
            values = [r[key] for r in rows if r[key] is not None]
            result[key] = statistics.fmean(values) if values else None
            denominators[key] = ratio(sum(values), len(values))
    for name, field in (
        ("no_result_rate", "predicted_no_result"),
        ("no_relevant_judgments_rate", "no_relevant_judgments"),
        ("dataset_no_result_slice_rate", "dataset_no_result"),
        ("no_result_accuracy", "no_result_correct"),
        ("dataset_no_result_accuracy", "dataset_no_result_correct"),
        ("failure_rate", "failed"),
        ("unsupported_constraint_disclosure", "unsupported_constraints_unapplied"),
        ("filter_correctness", "filter_correct"),
    ):
        scope = [r for r in rows if r["filter_evaluable"]] if name == "filter_correctness" else rows
        denominators[name] = ratio(sum(r[field] for r in scope), len(scope))
        result[name] = denominators[name]["value"]
    result["denominators"] = denominators
    result["unjudged_returned_count"] = sum(r["unjudged_returned_count"] for r in rows)
    return result
