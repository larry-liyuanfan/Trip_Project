# Project Requirements

## Week 1: OTA Multimodal VLM Engineering Foundation

### Goals

- Establish the Python, Docker, FastAPI, and vLLM project foundation.
- Support deterministic fallback tests without requiring a live GPU service.
- Verify a live single-image multimodal request on the local GPU.
- Prepare a small Yelp OTA sample and reproducible experiment records.

### Acceptance Criteria

- API health and image-understanding routes return structured responses.
- Docker serves the selected VLM through an OpenAI-compatible endpoint.
- Sample POIs, reviews, and images are available without committing raw Yelp data.
- Unit tests and experiment logs record the verified Week 1 state.

## Week 2: Yelp Multimodal Dataset Processing Pipeline

### Background
Week 1 established the project scaffold and small Yelp subset workflow. Week 2 builds a reproducible pipeline for parsing, validating, aligning, and reporting Yelp multimodal data.

### Goals
- Normalize Yelp raw, interim, processed, logs, and report directories.
- Consume the Week 1 downloaded/extracted Yelp files from the normalized raw directory.
- Read business, review, and photo JSONL files line by line for the full processing pipeline.
- Validate local photo files and record image metadata.
- Build strong valid-image/non-empty-caption pairs, medium image-business pairs, and weak image-review pairs.
- Provide a reproducible CLIP denoising stage that runs in a dedicated GPU Docker task without adding torch or vLLM to the base data environment.
- Generate `reports/yelp_multimodal_data_processing_report_part1.md`.

### Non-goals
- Do not commit raw Yelp archives, extracted images, or large generated Parquet files.
- Do not require GPU or CLIP for the base pipeline.
- Do not build model training or a UI this week.

### Deliverables
- Config-driven parsing and alignment scripts.
- Reusable `src/data/` modules for archive extraction, JSONL parsing, validation, alignment, statistics, templates, and optional denoising.
- Interim Parquet outputs, processed alignment outputs, statistics JSON, and report draft for the full local Yelp run.
- Tests for parsers, validation, alignment, denoising skip behavior, and report generation.

### Acceptance Criteria
- `python -m unittest discover -s tests -v` passes.
- Week 2 data processing can be installed with `pip install -r requirements-data.txt` without installing `vllm`.
- Parsing script writes business, review, photo, image-index, review-stats, and validation-summary outputs.
- Alignment script writes strong, medium, weak, and dataset-statistics outputs.
- Denoising task writes a row-level denoised table, a summary, and similarity distribution; disabled/dependency-unavailable paths still write an explicit skipped summary.
- Report generator writes the Week 2 report using real statistics or explicit `TODO` markers.

### Risks / Questions
- Official Yelp archives may need manual extraction before parsing unless archive extraction is added.
- The Yelp download/extraction step is treated as a completed prerequisite for this Week 2 processing review.
- Full review/photo data is large, so review output uses chunked table writing to keep memory bounded.
- CLIP requires the `clip-denoising` Docker profile and exclusive GPU access. Stop vLLM before full CLIP inference on the local 8GB GPU.
- Local environments without `pyarrow` will run with a CSV fallback at the configured output path until dependencies are installed.
- The default config sets `processing_limits.max_reviews` to `null` and uses chunked review writes for full-dataset parsing.
- vLLM should not be installed in native Windows Python by default; use Docker or WSL2 for live LLM serving dependencies.

## Week 3: Zero-Shot Business Baselines and Standardized Prompts

### Background

Week 3 measures the unoptimized multimodal model on three OTA business scenarios before prompt optimization or model fine-tuning. The delivery must establish reproducible evaluation data contracts, objective metrics, raw-output traceability, standardized prompts, and backend-compatible JSON Schemas. The report is an engineering and business decision record, not a paper; use concise Markdown tables and evidence.

### Goals

- Define independent evaluation sets for image-to-product search, intelligent after-sales understanding, and multimodal itinerary planning.
- Enforce separation between evaluation samples and any future training candidates.
- Define reproducible scenario metrics, normalization rules, invalid-output handling, and batch scoring.
- Run the simplest possible zero-shot baseline without a role, JSON constraint, chain-of-thought instruction, few-shot example, or prompt optimization.
- Preserve every input, raw model output, latency, parse result, validation result, model version, and prompt version.
- Define a four-layer standardized prompt architecture and one JSON Schema per scenario.
- Report measured weaknesses and evidence-backed prompt or fine-tuning priorities without inventing results.

### Non-goals

- Do not build a training or fine-tuning pipeline.
- Do not implement embedding retrieval, a UI, or an unrequested future-week feature.
- Do not auto-generate completed manual annotations or use model-generated labels as ground truth.
- Do not require a full live-model run when validated data or compute is unavailable; a verified framework, dry-run, and honest partial evaluation are acceptable interim states.
- Do not expose or request private chain-of-thought. Standard prompts may request concise observable evidence and field-level checks only.

### Evaluation Set Targets and Status Counts

Target sizes describe the required evaluation design, not the current completion state:

| Scenario | Target | Required coverage |
| --- | ---: | --- |
| Image-to-product search | 200 images | Hotels, attractions, and restaurants |
| Intelligent after-sales | 150 evidence images | Hygiene stains, facility damage, attraction closure, and transport delay |
| Multimodal itinerary planning | 100 paired samples | Reference style image(s), text constraints, parsed requirements, and itinerary constraints |

Every scenario summary must report these counts separately:

- `target_count`: mentor-required target size.
- `candidate_count`: collected candidates before annotation review.
- `annotated_count`: candidates with completed human annotations.
- `validated_count`: annotations and files that pass validation.
- `tested_count`: validated samples with a persisted inference record for the selected run.

No report may present a target as completed work. Only validated samples may enter inference, and `tested_count` must come from actual run records.

### Annotation and Manifest Requirements

All manifests use versioned JSONL. Each record must include `sample_id`, `scenario`, `source_type`, `source_id`, `source_license`, `image_sha256` where applicable, `split`, `dataset_version`, `annotation_status`, `annotator`, `review_status`, `annotation`, and `notes`. Paths are repository-relative. Missing values use JSON `null`; unknown semantic values use documented enums rather than fabricated labels.

Image-to-product annotations contain business category, style tags, visible core facilities, and price range. After-sales annotations contain issue type, severity, key information, and OCR ground truth when applicable. Itinerary annotations contain reference images, text constraints, style preferences, hard constraints, soft constraints, and required itinerary elements.

Candidate generation may use deterministic stratified sampling with a recorded seed, criteria, and per-stratum counts. Candidate generation must set annotations to pending; only human-reviewed records may become completed. Public or synthetic after-sales sources must record provenance and usage rights.

### Evaluation Isolation

The evaluation registry records stable `source_id` and `image_sha256` identifiers and generates `data/eval/registry/evaluation_exclusion_manifest.jsonl`. A reusable validator must reject any future training candidate whose source ID or image hash appears in the exclusion manifest. Week 3 does not need a training pipeline, but it must provide the exclusion interface and deterministic conflict tests. A registry without enforced collision checking is insufficient.

### Zero-Shot Baseline and Run Records

Create one versioned minimal baseline instruction per scenario. Each instruction states only the recognition task and must not contain a role definition, required fields, JSON or formatting constraint, chain-of-thought instruction, or example. Keep baseline prompts physically and logically separate from standardized prompts, and never overwrite an earlier run.

Each inference record must include `run_id`, `sample_id`, `scenario`, `model_name`, `model_config`, `prompt_version`, normalized input metadata, `raw_output`, `parsed_output`, `json_valid`, `schema_valid`, `latency_ms`, `error`, and `timestamp`. The runner supports mock, dry-run, and live modes. It reads the current verified model and serving settings from configuration and documentation; it must not hard-code a model, download one, or change dependencies merely because a prompt mentions a model.

### Metrics and Scoring Rules

- Image-to-product: business-category and price-range accuracy; style and facility precision, recall, and F1; label completeness; JSON compliance; schema pass rate.
- Intelligent after-sales: issue and severity accuracy; key-information precision, recall, and F1; OCR field recall and exact match; JSON compliance; schema pass rate.
- Multimodal itinerary planning: constraint-recognition accuracy; hard- and soft-constraint recall; itinerary-element completeness; constraint-violation rate; JSON compliance; schema pass rate.

Metric specifications must define text normalization, enum aliases, missing fields, synonym handling, multi-label micro and macro averaging, and invalid JSON behavior. Invalid or unparseable output receives zero for structured task metrics and remains counted in format statistics. Sample-level scores, scenario aggregates, latency statistics, and error taxonomy must be reproducible from persisted records.

### Standardized Prompt Architecture

All standardized scenario prompts use four layers: system role, task instruction, input context, and output constraint. The common role identifies a professional OTA travel-platform assistant and requires Chinese output, domain relevance, explicit unknown values, separation of observation from inference, no fabrication, privacy protection, and route/travel safety.

Scenario prompts cover structured product labels; after-sales anomaly localization, classification, severity, key fields, and OCR; and itinerary preference parsing, hard/soft constraints, itinerary elements, and constraint checking. The model returns only JSON that matches the relevant Schema. `observed_evidence`, `constraint_check`, and confidence fields may contain only concise observable facts or field sources, never long-form reasoning or a narrated internal thought process.

### Deliverables

- Week 3 evaluation configuration, annotation specification, manifest contracts, and count definitions.
- Three minimal baseline prompts, common standardized prompt layers, and three standardized scenario prompts.
- Three JSON Schema files and prompt-rendering/schema-validation tests.
- Config-driven mock, dry-run, and live evaluation interfaces with immutable run outputs.
- Scenario metrics, count aggregation, exclusion validation, and error-case export with tests.
- A concise zero-shot baseline report and prompt architecture specification.
- Updated README, weekly records, experiment notes, and verification evidence.

### Acceptance Criteria

- All three evaluation formats and the five status counts are defined and validated.
- Evaluation samples produce an exclusion manifest; source-ID and image-hash collisions are rejected by tests.
- Baseline and standardized prompts are separately stored and versioned for all scenarios.
- Three JSON Schemas validate representative valid and invalid fixtures.
- The pipeline runs without a GPU in mock or dry-run mode and refuses to overwrite an existing run ID.
- Raw output, latency, model configuration, parse status, and schema status are persisted.
- Scenario metrics and invalid-output rules have deterministic unit tests.
- Reports display target, candidate, annotated, validated, and tested counts separately.
- Any live or partial result is traceable to persisted output; unavailable results are marked `PENDING` rather than fabricated.
- `python -m unittest discover -s tests -v` and task-specific validators pass before delivery.

### Delivery Status Rules and Risks

- `READY`: core tests and dry-run pass, at least one real verifiable output exists, reports match outputs, and no blocking leakage, data, or security issue remains.
- `PARTIAL`: framework and tests pass, but real data, manual annotation, or full evaluation is incomplete; all missing work and five counts are explicit.
- `NOT READY`: a core flow fails, results are untraceable, evaluation leakage is possible, baseline and standardized prompts are mixed, or Git contains prohibited data.

Local GPU capacity may limit full inference. Public after-sales evidence may require licensing review, and human annotation capacity may limit completed counts. Any cloud or third-party runtime must be approved and documented before use. Phase completion requires a Project Control checkpoint; code defects found by Review return to Execution rather than being silently repaired during reporting.
