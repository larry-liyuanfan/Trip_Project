# Search scorer v2 contract

Scorer v2 is an offline measurement correction, not a new retrieval model or a new independent
test. The formal release, model prompts, weights, historical metrics and Fresh Test 120 are
unchanged. Configuration: `configs/evaluation/search_scorer_v2_contract.json`.

## Metric definitions

- Query universe: every unique ID in the exact query manifest, with query, image and source SHA
  bindings. Result rows must match it exactly. Missing methods / explicit failures receive zero
  on evaluable ranking queries, remain in denominators, and do not count as successful abstentions.
- Binary relevance is grade >= 2 by default; Recall@K divides the number of distinct retrieved
  positives by **all positives in the query's valid qrels**, not a caller-supplied `relevant_total`.
  MRR uses the first binary-positive rank. K is explicit, defaults to 5 and 10.
- Graded DCG uses `(2**grade - 1) / log2(rank + 1)`. IDCG sorts **all valid qrels**, then takes K.
  Binary relevance and graded gain are independent: grade 1 has gain 1 but is not a binary positive.
  Returning only grade 1 when an unreturned grade 3 exists yields nDCG@1 = **1/7**, not 1.
- Recall/MRR are null for queries with no binary-positive judgments; nDCG is null only when
  IDCG is zero. These nulls are excluded from their macro averages; every metric reports its
  own numerator and denominator. A nonempty positive qrels set plus empty predictions scores zero.
- Unknown/unjudged IDs get zero gain and are nonrelevant, keep their rank, and are counted explicitly.
  Duplicate query IDs, document IDs, query-document judgments and returned document IDs are rejected.
- `no_result_rate`: successful empty hit lists / all manifest queries. It is not a boolean supplied
  by the model (a contradictory `no_result` flag is rejected), and not the dataset slice fraction.
  `no_relevant_judgments_rate` and `dataset_no_result_slice_rate` are separate fields.
  `no_result_accuracy` compares actual empty predictions to absence of binary positives in qrels;
  `dataset_no_result_accuracy` compares to the historical slice label. These can disagree.
- Filter correctness uses locked document metadata rather than mutable hit fields. It is macro
  over queries with supported filters (city, business category, price). Successful empty results
  are vacuously filter-correct; this does **not** mean useful search. Unsupported constraints remain
  separately disclosed. All metrics and denominators are also reported per slice.

## Qrels and human provenance

`score_search_results(..., qrels=bundle)` accepts `search_qrels_v2` JSON. The bundle contains:

```text
schema_version: search_qrels_v2
coverage: complete_declared_document_universe
document_universe_kind: full_index | judged_pool | full_index_metadata_weak
documents: [{image_id, ...locked metadata...}]
corpus_sha256: canonical SHA of documents
annotation_sha256: canonical SHA of annotations
queries:
  - query_id, query_sha256, query_image_sha256, query_source_sha256, label_provenance
    judgments: [{image_id, grade, ...human ratings if applicable...}]
```

Every query must have one final grade (integer 0–3) for every declared document. Missing or
unresolved pairs are rejected; they are not filled from metadata. A `judged_pool` is only complete
**within that declared pool**, not the whole index. No-positive judgments in that pool cannot
establish that the full index contains no relevant product. This limitation applies to Recall and
no-result accuracy as well as nDCG.

For `label_provenance=human`, each annotation declares at least two distinct human annotator IDs;
each query-document judgment contains matching `ratings: [{annotator_id, grade}, ...]` and either:

- `conflict_resolution=none`: all grades agree with the final grade; or
- `conflict_resolution=adjudicated`: an independent `adjudicator_id`, a nonempty
  `adjudication_reason`, and the final grade. Unresolved conflicts are rejected.

Human documents additionally require `image_sha256`, a nonempty `source` record and its canonical
`source_record_sha256`. Query/source hashes are verified against the supplied manifest, and all
documents/ratings have unique IDs. These checks validate the **declared protocol**, not the
real-world identity of people or the actual image bytes. Scoring alone never promotes records to
verified human ground truth; authenticity and image-byte verification remain external requirements.

Human identity comparisons (declared annotators, per-pair raters, adjudicator exclusion and the
`programmatic_` prefix) all use the same `NFKC -> strip -> casefold` equivalence. Case, outer-space
or full-width variants cannot establish a second person or an independent adjudicator. This
normalization still does not authenticate the person behind any ID.

Weak/synthetic metadata grading is a separate explicit path:
`score_search_results(..., corpus=full_metadata)`. It constructs qrels across the entire supplied
catalog, hashes both corpus and annotations, and labels the result weak/synthetic. It is never a
fallback for human annotations. An arbitrary supplied catalog is not proof of complete index
coverage; the historical audit CLI additionally binds the formal archive's hash and exact 1000
metadata records. Metadata-only qrels do not prove semantic relevance for facilities, conflict
resolution or no-result queries. They can contradict the dataset's intended slice labels.

## Commands and versioning

```bash
python scripts/run_relevance_evidence.py --config <protocol.json> score-search \
  --results <predictions.jsonl> --qrels <qrels-v2.json> --ks 5 10 --output <new-report.json>
# Explicit weak alternative: replace --qrels with --weak-corpus <full-metadata.jsonl>.

python scripts/audit_search_scorer_v2.py --case v4_final \
  --input-root <ignored-audit-inputs> --retrieval-archive <formal-retrieval.tar.gz> \
  --output <new-versioned-audit.json>
# Also supported: --case v8_validation.
```

Outputs are exclusive-create. The historical audit verifies original query, annotation, prediction
and metric file hashes against committed evidence, and their canonical cross-bindings, before
scoring. It reports old/new denominators, corrected metrics on the old ranking subset, source Git
SHA, scorer SHA/files, original inference hardware and CPU audit hardware. It never loads CLIP,
Milvus, VLM or Fresh Test, changes thresholds, or reconsumes a split.

The old private aggregator was renamed `_aggregate_search_method_v1` solely for integrity checks
of immutable v8 evidence. Its known misnamed rate must not be used for current scoring. Inference
runners reject historical configurations lacking explicit `search.scorer_version=search_scorer_v2`;
a new inference experiment needs new data/gates and a versioned protocol. Do not add that field to
an old locked config to reuse a seen holdout. Historical gate success does not establish a scorer-v2
quality/latency qualification.

Audit evidence and missing records: `reports/scorer_v2_audit_20260914/HANDOFF.md`.
