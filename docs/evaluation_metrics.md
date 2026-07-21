# Week 3 Evaluation Metrics Contract

## Scope

This document defines Stage 4 scoring for persisted Week 3 evaluation records. Scoring is read-only with respect to `data/eval/runs/<run_id>` and does not call a model. Local score artifacts are written to the ignored `data/eval/scores/<run_id>` directory and an existing score directory is never overwritten.

The versioned alias contract is `configs/evaluation/metric_aliases_v1.json`. Any change to that file changes the scoring contract and must be reviewed as such.

## Matching and Missing Values

All comparable strings use the same deterministic pipeline:

1. Unicode NFKC normalization.
2. Unicode-aware case folding.
3. Leading and trailing whitespace removal and internal whitespace collapse.
4. At most one exact, field-specific alias lookup.

There is no fuzzy matching, stemming, token similarity, substring matching, or cross-field alias reuse. Alias chains are invalid. Values absent from the alias file retain their normalized text.

For a Schema-valid output, a missing scalar is normalized to an empty value and therefore does not match a non-empty annotation. A missing or non-array multi-label field is treated as an empty predicted set. Duplicate multi-label values are collapsed after normalization. OCR exact match is the exception: it compares the normalized ordered lists, so order and duplicate occurrences remain significant.

`ocr_ground_truth: null` means that the source provides no OCR reference. For a valid structured output, OCR recall and exact match are `null` and are excluded from their scenario means. An empty OCR reference list is a real reference: an empty prediction scores 1 for both recall and exact match.

For the standardized structured track, if either `json_valid` or `schema_valid` is false, supported structured task metrics for that sample are 0. Scalar gold explicitly marked `unknown` and nullable OCR gold are excluded from their metric denominator, with `<metric>_support_count` recording the actual denominator. The sample remains in format and latency aggregates. The unchanged minimal baseline has no human-coded secondary track: when its natural-language output cannot be deterministically parsed, JSON/Schema compliance and latency remain measured while semantic task metrics are `PENDING` with support count 0.

## Shared Multi-label Rules

For normalized expected set `G` and predicted set `P`:

- `TP = |G intersection P|`
- `FP = |P minus G|`
- `FN = |G minus P|`
- precision is `TP / (TP + FP)`
- recall is `TP / (TP + FN)`
- F1 is the harmonic mean of precision and recall

Empty-set behavior is explicit:

| Expected | Predicted | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| empty | empty | 1 | 1 | 1 |
| non-empty | empty | 0 | 0 | 0 |
| empty | non-empty | 0 | 1 | 0 |

Macro precision, recall, and F1 are arithmetic means of the corresponding sample scores, including invalid samples as zeros. Micro precision, recall, and F1 are calculated after summing TP, FP, and FN across all samples in one scenario. Both forms are exported with `_macro` and `_micro` suffixes; raw aggregate TP, FP, and FN are also retained.

If a field has no label events at all (`TP + FP + FN = 0`), its micro scores are 1 only when every represented sample has valid structured output. If any represented sample is invalid, the micro scores are 0 while the raw counts remain zero; this prevents an invalid empty prediction from receiving empty-set credit without inventing a false label event.

## Scenario Metrics

### Image-to-product

- `business_category_accuracy` and `price_range_accuracy`: normalized exact equality, 1 or 0.
- Style and facility precision, recall, and F1: shared multi-label rules.
- `label_completeness`: `(correct category + correct price + recovered gold style labels + recovered gold facility labels) / (2 + gold style count + gold facility count)`.

### Intelligent after-sales

- `issue_type_accuracy` and `severity_accuracy`: normalized exact equality, 1 or 0.
- Key-information precision, recall, and F1: shared multi-label rules.
- `ocr_recall`: set recall after normalization.
- `ocr_exact_match`: exact equality of normalized ordered OCR lists.

### Multimodal itinerary planning

Hard and soft constraints are typed labels. The same text in the wrong type is not a match.

- `constraint_recognition_accuracy`: Jaccard similarity of the expected and predicted typed constraint sets, `intersection / union`; both empty is 1.
- Hard- and soft-constraint precision, recall, and F1: shared multi-label rules. The required headline metrics are the two recalls.
- Itinerary-element precision, recall, and F1: shared multi-label rules. `itinerary_element_completeness` is the same value as itinerary-element recall.
- `constraint_check_coverage`: expected typed constraints with at least one matching `constraint_check` row divided by expected typed constraints; no expected constraints is 1.
- `constraint_violation_rate`: expected typed constraints with at least one matching `violated` check divided by expected typed constraints; no expected constraints is 0. A missing check is exposed through coverage and is not silently treated as a violation.

## Format, Latency, and Errors

`json_compliance` is 1 only when strict JSON parsing succeeded. `schema_pass` is 1 only when the parsed object passed the complete scenario Schema. Scenario aggregates report their arithmetic means.

Latency aggregates include count, minimum, arithmetic mean, median, nearest-rank p95, and maximum in milliseconds. Only finite, non-negative persisted latency values are included. For nearest-rank p95, the selected sorted index is `ceil(0.95 * n) - 1`.

The error taxonomy is:

- `dry_run`
- `mock_fixture_missing`
- `model_request_error`
- `json_parse_error`
- `schema_validation_error`
- `missing_output`
- `unknown_error`
- `valid`

Mock fixture absence and live request failure remain distinct. Error rows retain traceability fields, raw and parsed output, validity flags, the original error, and the sample metrics.

## Joins and Artifacts

Results join to completed annotations by `sample_id`. Duplicate result IDs, duplicate annotation IDs across manifests, missing annotations, run-ID mismatches, and scenario mismatches are rejected. The scorer does not fabricate or skip a score for any persisted result.

Before reading annotations or creating a score directory, the scorer strictly parses and validates `data/eval/runs/<run_id>/metadata.json`. It requires the runner traceability fields, non-negative integer `selected_count` and `record_count`, a `completed` or `failed` status, and status-consistent error information. `NaN`, `Infinity`, and `-Infinity` are invalid JSON values.

The following cross-file invariants are mandatory:

- `metadata.run_id` equals the requested `--run-id`.
- `metadata.record_count` equals the actual validated row count in `results.jsonl`.
- For a `completed` run, `metadata.selected_count` equals `metadata.record_count`.
- Every result row has the same run ID.

Failed runs are not scored. The scorer rejects `status: failed` with the persisted run error and does not create a partial score directory or silently label the run complete. A successful scoring summary retains `run_status`, `selected_count`, and `record_count` from the validated metadata.

Each new score directory contains:

```text
data/eval/scores/<run_id>/
  sample_scores.jsonl
  aggregate_scores.csv
  error_cases.jsonl
  score_summary.json
```

Both JSONL files use strict JSON serialization and reject `NaN`, `Infinity`, and `-Infinity`. The CSV contains one flat row per represented scenario. Empty or pre-existing score directories are not reused.

The unchanged minimal baseline has no JSON-format instruction. Invalid natural-language output is a format failure, not proof of a semantic failure. Use a simple documented coding or extraction rule for the required semantic metrics; when that cannot be done reliably, store the semantic metric as `null`/`PENDING`. A standardized-vs-baseline paired comparison and bootstrap analysis are not Week 3 requirements.

The completed real baseline supplies format and latency metrics. Its unparsed semantic task metrics remain `PENDING`; standardized metrics report reduced support where frozen gold is `unknown` or not applicable.

After Stage 5 has produced an approved persisted run, strict structured scoring can be invoked with:

```powershell
python scripts/score_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id <existing_run_id>
```

Stage 4 verification uses only synthetic fixtures. This command is documented for the later approved run phase and is not evidence that a real evaluation has occurred.
