# Week 3 Evaluation Data Contracts

## Scope and current status

This document defines Phase 1 contracts for three independent Week 3 evaluation sets. Target sizes are design requirements, not completed sample counts. Real images, manifests, sampling logs, and the exclusion registry are local ignored data. A new clone creates the empty local files with the explicit `init` command; no empty runtime JSONL is committed.

| Scenario | Scenario key | Target | Current candidate / annotated / validated / tested |
| --- | --- | ---: | ---: |
| Image-to-product search | `image_product_search` | 200 | 200 / 200 / 200 / 200 |
| Intelligent after-sales | `after_sales` | 150 | 150 / 150 / 150 / 150 |
| Multimodal itinerary planning | `itinerary_planning` | 100 | 100 / 100 / 100 / 100 |

The current counts bind `tested` to completed baseline run `week3_baseline_full_20260721_003`. The frozen labels remain human-authored; `unknown` and empty semantic fields limit metric support rather than structural validation.

## Common JSONL record

Every non-empty line is one UTF-8 JSON object. Paths are repository-relative, missing values use JSON `null`, and unknown semantic labels use the documented `unknown` enum rather than a guessed value.

| Field | Contract |
| --- | --- |
| `sample_id` | Non-empty, unique string generated from scenario and stable source ID. |
| `scenario` | `image_product_search`, `after_sales`, or `itinerary_planning`. |
| `source_type` | Non-empty provenance type such as an approved public source or synthetic fixture. |
| `source_id` | Stable non-empty source identifier used by leakage checks; candidate inputs must be unique by this field. |
| `source_license` | Non-empty license or usage-rights statement. |
| `image_sha256` | Lowercase 64-character SHA-256 of the primary (first) input image. This compatibility field must equal `input.images[0].sha256`. |
| `input` | Inference-ready input independent of `annotation`; required even while annotation is pending. |
| `split` | Always `evaluation`. |
| `dataset_version` | Non-empty version string. |
| `annotation_status` | `pending`, `in_progress`, or `completed`. |
| `annotator` | Human annotator ID when work begins; otherwise `null`. Automated candidate generation never sets this field. |
| `review_status` | Legacy compatibility field: `pending`, `validated`, or `rejected`; it is not a release gate. |
| `reviewer` | Legacy compatibility field. New single-annotator records keep this `null`. |
| `file_status` | `pending`, `valid`, `missing`, or `unreadable`. |
| `annotation` | Scenario object only when annotation is completed; otherwise `null`. |
| `notes` | String or `null`. |
| `sampling_stratum` | Candidate-generation stratum recorded on sampled rows; omitted only for records created outside the sampler. |
| `provenance` | Source URI/version and optional source or synthetic-recipe metadata. |

Lifecycle invariants are enforced: pending records have null annotator and annotation; in-progress records have a human annotator but no completed annotation; completed records require the full scenario annotation. Legacy non-pending review metadata remains structurally validated but does not control release eligibility.

## Inference input

`input` has exactly two fields:

```json
{
  "images": [
    {
      "path": "data/eval/images/example.jpg",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "perceptual_hash": "0000000000000000"
    }
  ],
  "text_constraints": null
}
```

- `images` is a non-empty array. Product-search and after-sales records contain exactly one image; itinerary records may contain multiple reference images.
- Every image path is repository-relative and every image has its own byte SHA-256. Release candidates also carry a 64-bit perceptual hash. Manifest loading and candidate sampling read actual bytes and reject a mismatch, missing file, path traversal, or path escaping the repository through resolution.
- `text_constraints` is a non-empty raw source string for itinerary records and JSON `null` for the two image-only scenarios.
- The input remains available before annotation, so a pending record can later enter inference after human annotation and validation without reconstructing source data.

## Scenario annotations

### Image-to-product search

- `business_category`: `hotel`, `attraction`, `restaurant`, or `unknown`.
- `style_tags`: array of observed style labels.
- `visible_facilities`: array of visible facility labels.
- `price_range`: `budget`, `mid_range`, `premium`, `luxury`, or `unknown`.

### Intelligent after-sales

- `issue_type`: `hygiene_stain`, `facility_damage`, `attraction_closure`, `transport_delay`, `other`, or `unknown`.
- `severity`: `low`, `medium`, `high`, `critical`, or `unknown`.
- `key_information`: array of concise annotated facts.
- `ocr_ground_truth`: array of human-transcribed strings, or `null` when OCR is not applicable.

### Multimodal itinerary planning

- `reference_images`: non-empty array of repository-relative image paths.
- `text_constraints`: array of source text constraints.
- `style_preferences`: array of style preferences.
- `hard_constraints`: array of mandatory constraints.
- `soft_constraints`: array of preferences.
- `required_itinerary_elements`: array of elements required in a valid itinerary.

## Counts

- `target_count`: configured mentor-required size.
- `candidate_count`: records collected into the scenario manifest.
- `annotated_count`: records whose human `annotation_status` is `completed`.
- `validated_count`: completed human annotations with `file_status=valid`.
- `tested_count`: validated records whose sample IDs occur in a persisted completed full live run. Use `python scripts/validate_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id <completed-run-id>` to derive this count from `results.jsonl`; omitting `--run-id` intentionally reports zero because no run evidence was supplied.

## Deterministic sampling

`configs/evaluation_week3.yaml` records a fixed seed and candidate strata for the mentor-required coverage groups. Exact human-gold class balance is not a mentor requirement. The sampler records selected source IDs, preserves the complete top-level `input`, computes or verifies image hashes from local bytes, and resets candidate annotations to pending.

Sampling outputs and audit logs are local run artifacts. They must not be committed until the selected sources, licenses, and intended evaluation use are approved.

The current source composition is explicit:

- Product candidates use Yelp photos joined to explicit OTA business categories. Category precedence is hotel, attraction, then restaurant when Yelp assigns multiple categories. This is only stratification; a human supplies the gold business category.
- The final after-sales set must contain both relevant public-scene samples and business-synthetic samples across the four required problem types. Weak Yelp review/photo pairing may discover candidates but does not replace human relevance checking.
- Itinerary candidates pair unused Yelp reference images with deterministic project-owned text-constraint templates. A human must parse and validate every constraint.

## Evaluation isolation

Every registered evaluation candidate contributes an exclusion row with stable source ID, image SHA-256, scenario, and dataset version. Registry construction rejects duplicate source IDs and image hashes across the manifests. A reusable validator rejects future training candidates that collide by source ID or image hash. Additional deduplication metadata may be retained, but it is not a separate Week 3 deliverable.

## Local workspace lifecycle

`data/eval/images/`, manifests, annotation packets, run/score outputs, sampling logs, and the actual exclusion registry are ignored. Initialization creates only missing directories and empty JSONL files; it never truncates an existing manifest or registry.

## Phase 1 commands

```powershell
python scripts/prepare_week3_evaluation.py --config configs/evaluation_week3.yaml init
python scripts/build_week3_candidate_manifests.py --config configs/evaluation_week3.yaml
python scripts/prepare_week3_evaluation.py --config configs/evaluation_week3.yaml build-exclusion
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3.yaml
python scripts/manage_week3_annotations.py --scenario image_product_search export --output data/eval/codings/image_product_search_annotation_packet.jsonl
python -m unittest tests.test_evaluation_manifests tests.test_evaluation_annotation_workflow -v
```

The builder refuses to overwrite non-empty manifests or registry files. Annotation packets are applied only after the full batch passes validation and before an atomic manifest replacement. See `docs/week3_annotation_guidelines.md`.

Week 3 intentionally provides no browser annotation UI. Human annotation uses
exported JSONL packets and transactional CLI application so the Git deliverable
stays within the approved non-UI scope.

Annotation export may add deterministic, explicitly non-gold suggestions under packet-only `context.deterministic_suggestion`. These hints are derived from source metadata and rules, never from a VLM, and are removed before the submission reaches manifest validation.
